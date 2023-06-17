use bevy::{
  math::Vec3Swizzles,
  prelude::{
    default, on_event, resource_exists, shape, App, Assets, Bundle,
    Camera2dBundle, Color, Commands, Component, DefaultPlugins, Entity, Handle,
    Image, In, IntoPipeSystem, IntoSystemConfig, Mesh, PluginGroup, Query,
    Reflect, ReflectResource, Res, ResMut, Resource, Transform, Vec2, Vec3,
  },
  sprite::{ColorMaterial, MaterialMesh2dBundle},
  time::Time,
  window::{Window, WindowCloseRequested, WindowPlugin},
};
use bevy_egui::{EguiContexts, EguiPlugin};
use bevy_inspector_egui::{
  prelude::ReflectInspectorOptions, quick::ResourceInspectorPlugin,
  InspectorOptions,
};
use crossbeam::channel::{self, Receiver};
use image::{
  codecs::jpeg::JpegDecoder, DynamicImage, GenericImageView, GrayImage,
  ImageDecoder, Luma,
};
use keyde::KdTree;
use nokhwa::{pixel_format::LumaFormat, CallbackCamera, Camera};

#[derive(Component)]
struct GameCamera;

#[derive(Resource)]
struct CameraDev {
  #[allow(unused)]
  camera: CallbackCamera,
  receiver: Receiver<DynamicImage>,
}

#[derive(Resource, Clone, Default)]
struct LumaGrid(GrayImage);

#[derive(Resource, Clone, Default)]
struct LumaGridPreview(Handle<Image>);

#[derive(Component, Clone, Copy, Default)]
struct Velocity(Vec2);

#[derive(Component, Clone, Copy, Default)]
struct ImageGradient(Vec2);

#[derive(Component, Clone, Copy, Default)]
struct RepelGradient(Vec2);

#[derive(Component, Clone, Copy, Default)]
struct Friction(Vec2);

#[derive(Default, Resource, Bundle, Clone)]
struct CircleBundle {
  #[bundle]
  mesh: MaterialMesh2dBundle<ColorMaterial>,
  velocity: Velocity,
  image_grad: ImageGradient,
  repel_gradient: RepelGradient,
  friction: Friction,
}

#[derive(Default, Resource, Clone, Reflect)]
struct Grid<T> {
  size: (usize, usize),
  value: Vec<T>,
}

#[derive(Default, Resource, Clone)]
struct TrackedCircles {
  circles: Vec<Entity>,
}

#[derive(Resource)]
struct StaticParam {
  pub size: (f32, f32),
  pub circle_grid: (usize, usize),
  pub circle_radius: f32,
}

#[derive(Resource, Reflect, InspectorOptions)]
#[reflect(Resource, InspectorOptions)]
struct DynamicParam {
  #[inspector(min = 0.0, max = 100.0)]
  pub friction_coeff: f32,
  #[inspector(min = 0.0, max = 1000.0)]
  pub repel_coeff: f32,
  #[inspector(min = -1000.0, max = 1000.0)]
  pub gradient_scale: f32,
  #[inspector(min = 0.0, max = 10000.0)]
  pub max_velocity: f32,
}

impl Default for DynamicParam {
  fn default() -> Self {
    Self {
      friction_coeff: 10.0,
      repel_coeff: 100.0,
      gradient_scale: -100.0,
      max_velocity: 1000.0,
    }
  }
}

fn main() {
  let static_param = StaticParam::default();
  let plugins = DefaultPlugins.set(WindowPlugin {
    primary_window: Some(Window {
      title: "Dotcam".into(),
      resolution: static_param.resolution(),
      ..default()
    }),
    ..default()
  });

  App::new()
    .add_plugins(plugins)
    .add_plugin(EguiPlugin)
    .insert_resource(static_param)
    .init_resource::<DynamicParam>()
    .init_resource::<TrackedCircles>()
    .add_plugin(ResourceInspectorPlugin::<DynamicParam>::default())
    .add_startup_system(setup_webcam)
    .add_startup_system(setup_camera)
    .add_startup_system(setup_circle_bundle.pipe(spawn_circles))
    .add_system(save_camera_output.run_if(resource_exists::<CameraDev>()))
    .add_system(update_image_gradient.run_if(resource_exists::<LumaGrid>()))
    .add_system(update_repel_gradient)
    .add_system(
      update_velocity
        .after(update_image_gradient)
        .after(update_repel_gradient),
    )
    .add_system(update_friction.after(update_velocity))
    .add_system(inspect_buffer.run_if(resource_exists::<LumaGridPreview>()))
    .add_system(physics_velocity_system)
    .add_system(stop_webcam.run_if(on_event::<WindowCloseRequested>()))
    .run();
}

fn setup_webcam(mut commands: Commands, param: Res<StaticParam>) {
  use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};
  let index = CameraIndex::Index(0);
  let requested = RequestedFormat::new::<LumaFormat>(
    RequestedFormatType::AbsoluteHighestResolution,
  );

  let camera = Camera::new(index, requested).unwrap();

  let (sender, receiver) = channel::bounded(1);

  let size = [param.width() as u32, param.height() as u32];
  let mut camera = CallbackCamera::with_custom(camera, move |buffer| {
    let image = camera_buffer_to_image(buffer, size);
    sender.send(image).unwrap();
  });
  camera.open_stream().unwrap();

  commands.insert_resource(CameraDev { camera, receiver });
}

fn setup_camera(mut commands: Commands) {
  commands.spawn((Camera2dBundle::default(), GameCamera));
}

fn camera_buffer_to_image(
  buffer: nokhwa::Buffer,
  size: [u32; 2],
) -> DynamicImage {
  use nokhwa::utils::FrameFormat::MJPEG;

  assert!(buffer.source_frame_format() == MJPEG);

  let decoder = JpegDecoder::new(buffer.buffer()).unwrap();
  debug_assert!(decoder.color_type() == image::ColorType::Rgb8);

  let image: DynamicImage = buffer.decode_image::<LumaFormat>().unwrap().into();

  let dim = image.height().min(image.width());
  let image = image
    .crop_imm(
      (image.width() - dim) / 2,
      (image.height() - dim) / 2,
      dim,
      dim,
    )
    .resize_exact(size[0], size[1], image::imageops::FilterType::Nearest)
    .fliph();

  image
}

fn save_camera_output(
  mut commands: Commands,
  camera_dev: Res<CameraDev>,
  mut luma_grid: Option<ResMut<LumaGrid>>,
  mut preview: Option<ResMut<LumaGridPreview>>,
  mut images: ResMut<Assets<Image>>,
) {
  let receiver = &camera_dev.receiver;
  let Ok(image) = receiver.try_recv() else {
    return;
  };

  if let Some(preview) = preview.as_mut() {
    let bevy_image = images.get_mut(&preview.0).unwrap();
    bevy_image.data.copy_from_slice(image.as_bytes());
  } else {
    let data = image.as_bytes().to_vec();
    let width = image.width();
    let height = image.height();
    let format = bevy::render::render_resource::TextureFormat::R8Unorm;
    let bevy_image = Image::new(
      bevy::render::render_resource::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
      },
      bevy::render::render_resource::TextureDimension::D2,
      data,
      format,
    );
    let handle = images.add(bevy_image);
    commands.insert_resource(LumaGridPreview(handle));
  }

  if let Some(luma_grid) = luma_grid.as_mut() {
    luma_grid.0.copy_from_slice(image.as_bytes());
  } else {
    commands.insert_resource(LumaGrid(image.into_luma8()));
  }
}

fn setup_circle_bundle(
  mut commands: Commands,
  mut meshes: ResMut<Assets<Mesh>>,
  mut materials: ResMut<Assets<ColorMaterial>>,
  dimension_info: Res<StaticParam>,
) -> CircleBundle {
  let mesh_handle = meshes
    .add(Mesh::from(shape::Circle {
      radius: dimension_info.circle_radius,
      ..default()
    }))
    .into();

  let material_handle = materials.add(Color::rgb(1.0, 0.7, 0.7).into());
  let bundle = CircleBundle {
    mesh: MaterialMesh2dBundle {
      mesh: mesh_handle,
      material: material_handle,
      ..default()
    },
    velocity: Velocity(Vec2::new(0.0, 0.0)),
    ..default()
  };

  commands.insert_resource(bundle.clone());
  bundle
}

fn spawn_circles(
  In(circle_bundle): In<CircleBundle>,
  dimension_info: Res<StaticParam>,
  mut tracked_circles: ResMut<TrackedCircles>,
  mut commands: Commands,
) {
  for pos in dimension_info.circle_positions() {
    let entity = commands
      .spawn(circle_bundle.clone())
      .insert(Transform::from_translation(pos.extend(0.0)))
      .id();

    tracked_circles.circles.push(entity);
  }
}

fn update_image_gradient(
  dynamic_param: Res<DynamicParam>,
  mut q: Query<(&Transform, &mut ImageGradient)>,
  luma_grid: Res<LumaGrid>,
  param: Res<StaticParam>,
) {
  let luma_grid = &luma_grid.0;
  for (trans, mut grad) in q.iter_mut() {
    let [x, y] = param.translation_to_pixel(&trans.translation);
    let new_grad =
      find_gradient(luma_grid, x, y) * dynamic_param.gradient_scale;

    *grad = ImageGradient(new_grad);
  }
}

fn update_repel_gradient(
  dynamic_param: Res<DynamicParam>,
  all_trans: Query<&Transform>,
  mut q: Query<(&Transform, &mut RepelGradient)>,
  all_circles: Res<TrackedCircles>,
) {
  const RADIUS: f32 = 5.0;
  const ALT_RADIUS: f32 = 30.0;

  let points: Vec<[f32; 2]> = all_trans
    .iter_many(&all_circles.circles)
    .map(|t| t.translation)
    .map(|t| [t.x, t.y])
    .collect();

  let kdtree = KdTree::from_points(&points);

  for (trans, mut grad) in q.iter_mut() {
    let mut new_grad = Vec2::ZERO;
    let trans = trans.translation;
    let point = [trans.x, trans.y];

    // 1% of the time, use a larger radius to avoid getting stuck
    let radius = if rand::random::<f32>() < 0.01 {
      ALT_RADIUS
    } else {
      RADIUS
    };

    for neigh_id in kdtree.point_indices_within(point, radius) {
      let entity = all_circles.circles[neigh_id];
      let neigh_trans = all_trans.get(entity).unwrap().translation;

      let dist_sq = trans.distance_squared(neigh_trans);
      if dist_sq < 0.0001 {
        continue;
      }

      let force = (trans - neigh_trans).normalize() / dist_sq;
      new_grad += force.xy();
    }

    grad.0 = new_grad * dynamic_param.repel_coeff;
  }
}

fn update_friction(
  dyn_param: Res<DynamicParam>,
  mut q: Query<(&Velocity, &mut Friction)>,
) {
  for (vel, mut friction) in q.iter_mut() {
    friction.0 = -vel.0 * dyn_param.friction_coeff;
  }
}

fn update_velocity(
  time: Res<Time>,
  dyn_param: Res<DynamicParam>,
  mut q: Query<(&mut Velocity, &ImageGradient, &RepelGradient, &Friction)>,
) {
  for (mut vel, grad, grad2, grad3) in q.iter_mut() {
    let force = grad.0 + grad2.0 + grad3.0;
    vel.0 += force * time.delta_seconds();
    vel.0 = vel.0.clamp_length_max(dyn_param.max_velocity);
  }
}

fn physics_velocity_system(
  time: Res<Time>,
  dimension_info: Res<StaticParam>,
  mut query: Query<(&mut Transform, &mut Velocity)>,
) {
  let dt = time.delta_seconds();
  let [x_min, x_max, y_min, y_max] = dimension_info.boundary();

  for (mut transform, velocity) in query.iter_mut() {
    let mut new_translation =
      transform.translation + Vec3::new(velocity.0.x, velocity.0.y, 0.0) * dt;
    new_translation.x = wrap_around(new_translation.x, x_min, x_max);
    new_translation.y = wrap_around(new_translation.y, y_min, y_max);
    transform.translation = new_translation;
  }
}

fn inspect_buffer(mut ctx: EguiContexts, preview: Res<LumaGridPreview>) {
  let preview_handle = &preview.0;

  let texture_id = ctx
    .image_id(preview_handle)
    .unwrap_or_else(|| ctx.add_image(preview_handle.clone()));

  bevy_inspector_egui::egui::Window::new("Camera Buffer").show(
    ctx.ctx_mut(),
    |ui| {
      ui.image(texture_id, [100.0, 100.0]);
    },
  );
}

fn stop_webcam(mut _webcam: ResMut<CameraDev>) {
  // it's unable to acquire lock from another thread to stop
  // the camera stream. resulting the program to hang.
  //
  // this hack is simply exit the program.
  std::process::exit(0);
}

impl Default for StaticParam {
  fn default() -> Self {
    Self {
      size: (500.0, 500.0),
      circle_grid: (60, 60),
      circle_radius: 1.0,
    }
  }
}

impl StaticParam {
  fn width(&self) -> f32 {
    self.size.0
  }
  fn height(&self) -> f32 {
    self.size.1
  }

  fn resolution(&self) -> bevy::window::WindowResolution {
    let (w, h) = self.size;
    bevy::window::WindowResolution::new(w, h)
  }

  fn circle_positions(&self) -> impl Iterator<Item = Vec2> {
    let (grid_w, grid_h) = self.circle_grid;
    let [x0, x1, y0, y1] = self.boundary();

    let x_step = (x1 - x0) / (grid_w as f32);
    let y_step = (y1 - y0) / (grid_h as f32);

    let x_start = x0 + x_step / 2.0;
    let y_start = y0 + y_step / 2.0;

    (0..grid_w).flat_map(move |x| {
      (0..grid_h).map(move |y| {
        let x = x_start + x_step * (x as f32);
        let y = y_start + y_step * (y as f32);
        Vec2::new(x, y)
      })
    })
  }

  fn boundary(&self) -> [f32; 4] {
    let x0 = self.size.0 / -2.0;
    let x1 = self.size.0 / 2.0;
    let y0 = self.size.1 / -2.0;
    let y1 = self.size.1 / 2.0;

    [x0, x1, y0, y1]
  }

  fn translation_to_pixel(&self, translation: &Vec3) -> [u32; 2] {
    let [x0, x1, y0, y1] = self.boundary();
    let x = (translation.x - x0) / (x1 - x0) * self.width();
    let y = (1.0 - (translation.y - y0) / (y1 - y0)) * self.height();
    [x as u32, y as u32]
  }
}

fn wrap_around(mut v: f32, min: f32, max: f32) -> f32 {
  if v > min && v < max {
    return v;
  }

  let range = max - min;
  while v < min {
    v += range;
  }
  while v > max {
    v -= range;
  }
  v
}

fn find_gradient(
  luma_grid: &impl GenericImageView<Pixel = Luma<u8>>,
  x: u32,
  y: u32,
) -> Vec2 {
  let mut grad = Vec2::ZERO;
  let x = x.max(1).min(luma_grid.width() - 2);
  let y = y.max(1).min(luma_grid.height() - 2);

  let get_pixel =
    |x: u32, y: u32| -> f32 { luma_grid.get_pixel(x, y).0[0] as f32 };

  grad.x = get_pixel(x + 1, y) - get_pixel(x - 1, y);
  grad.y = get_pixel(x, y + 1) - get_pixel(x, y - 1);

  grad
}
