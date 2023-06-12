use std::io::Read;

use bevy::{
  prelude::{
    default, shape, App, Assets, Bundle, Camera2dBundle, Color, Commands,
    Component, DefaultPlugins, Entity, Handle, Image, In, IntoPipeSystem,
    IntoSystemConfig, Mesh, NonSendMut, PluginGroup, Query, Rect, Reflect,
    ReflectResource, Res, ResMut, Resource, Transform, Vec2, Vec3, World,
  },
  render::render_resource::{Extent3d, TextureDimension, TextureFormat},
  sprite::{ColorMaterial, MaterialMesh2dBundle},
  time::Time,
  window::{Window, WindowPlugin},
};
use bevy_egui::{EguiContexts, EguiPlugin};
use image::{codecs::jpeg::JpegDecoder, ImageDecoder};
use itertools::Itertools;
use nokhwa::{pixel_format::LumaFormat, Camera};

#[derive(Component)]
struct GameCamera;

#[derive(Resource, Reflect, Default)]
#[reflect(Resource)]
struct CameraBuffer {
  image: Handle<Image>,
  width: u32,
  height: u32,
}

#[derive(Component, Clone, Copy, Default)]
struct Velocity(Vec2);

#[derive(Component, Clone, Copy, Default)]
struct Force(Vec2);

#[derive(Default, Resource, Bundle, Clone)]
struct CircleBundle {
  #[bundle]
  mesh: MaterialMesh2dBundle<ColorMaterial>,
  force: Force,
  velocity: Velocity,
}

#[derive(Default, Resource, Clone)]
struct Grid<T> {
  size: (usize, usize),
  value: Vec<T>,
}

#[derive(Default, Resource, Clone)]
struct TrackedCircles {
  circles: Vec<Entity>,
}

#[derive(Resource)]
struct DimensionInfo {
  pub viewport_size: (f32, f32),
  pub circle_grid: (usize, usize),
  pub circle_radius: f32,
  pub boundary: Rect,
}

fn main() {
  let dim = DimensionInfo::new(500.0, 500.0);
  let plugins = DefaultPlugins.set(WindowPlugin {
    primary_window: Some(Window {
      title: "Dotcam".into(),
      resolution: dim.resolution(),
      ..default()
    }),
    ..default()
  });

  App::new()
    .add_plugins(plugins)
    .add_plugin(EguiPlugin)
    .insert_resource(dim)
    .init_resource::<TrackedCircles>()
    .add_startup_system(setup_webcam)
    .add_startup_system(setup_camera)
    .add_startup_system(setup_circle_bundle.pipe(spawn_circles))
    .add_system(update_camera_output)
    .add_system(inspect_buffer)
    .add_system(physics_force_system)
    .add_system(physics_velocity_system.after(physics_force_system))
    .run();
}

fn setup_webcam(world: &mut World) {
  use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};
  let index = CameraIndex::Index(0);
  let requested = RequestedFormat::new::<LumaFormat>(
    RequestedFormatType::AbsoluteHighestResolution,
  );

  let mut camera = Camera::new(index, requested).unwrap();
  let resolution = camera.resolution();
  camera.open_stream().unwrap();
  world.insert_non_send_resource(camera);

  let width = resolution.width();
  let height = resolution.height();
  let extent = Extent3d {
    width,
    height,
    depth_or_array_layers: 1,
  };
  let dimension = TextureDimension::D2;
  let format = TextureFormat::Rgba8UnormSrgb;
  let image = Image::new_fill(extent, dimension, &[100, 0, 0, 255], format);

  let mut images = world.get_resource_mut::<Assets<Image>>().unwrap();

  let image = images.add(image);

  world.insert_resource(CameraBuffer {
    image,
    width,
    height,
  });
}

fn setup_camera(mut commands: Commands) {
  commands.spawn((Camera2dBundle::default(), GameCamera));
}

fn update_camera_output(
  mut camera: Option<NonSendMut<Camera>>,
  mut buffer: Option<ResMut<CameraBuffer>>,
  mut images: ResMut<Assets<Image>>,
) {
  use nokhwa::utils::FrameFormat::MJPEG;

  let camera = match camera.as_mut() {
    Some(camera) => camera,
    None => return,
  };
  let image = match buffer.as_mut() {
    Some(buffer) => images.get_mut(&buffer.image).unwrap(),
    None => return,
  };

  let frame = camera.frame().unwrap();
  debug_assert!(frame.source_frame_format() == MJPEG);

  let decoder = JpegDecoder::new(frame.buffer()).unwrap();

  debug_assert!(decoder.color_type() == image::ColorType::Rgb8);

  decoder
    .into_reader()
    .unwrap()
    .bytes()
    .tuples()
    .zip(image.data.iter_mut().tuples())
    .for_each(|((r, g, b), (dr, dg, db, _da))| {
      *dr = r.unwrap();
      *dg = g.unwrap();
      *db = b.unwrap();
    });
}

fn setup_circle_bundle(
  mut commands: Commands,
  mut meshes: ResMut<Assets<Mesh>>,
  mut materials: ResMut<Assets<ColorMaterial>>,
  dimension_info: Res<DimensionInfo>,
) -> CircleBundle {
  let mesh_handle = meshes
    .add(Mesh::from(shape::Circle {
      radius: dimension_info.circle_radius,
      ..Default::default()
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
    force: Force(Vec2::new(0.0, 0.0)),
  };

  commands.insert_resource(bundle.clone());
  bundle
}

fn spawn_circles(
  In(circle_bundle): In<CircleBundle>,
  dimension_info: Res<DimensionInfo>,
  mut tracked_circles: ResMut<TrackedCircles>,
  mut commands: Commands,
) {
  // random force an each circle
  let mut rng = rand::thread_rng();

  for pos in dimension_info.circle_positions() {
    let entity = commands
      .spawn(circle_bundle.clone())
      .insert(Transform::from_translation(pos.extend(0.0)))
      .insert(Force(Vec2::new(
        rng.gen_range(-5.0..5.0),
        rng.gen_range(-5.0..5.0),
      )))
      .id();

    tracked_circles.circles.push(entity);
  }
}

fn physics_force_system(
  time: Res<Time>,
  mut query: Query<(&mut Force, &mut Velocity)>,
) {
  let dt = time.delta_seconds();

  for (force, mut velocity) in query.iter_mut() {
    velocity.0 += force.0 * dt;
  }
}

fn physics_velocity_system(
  time: Res<Time>,
  dimension_info: Res<DimensionInfo>,
  mut query: Query<(&mut Transform, &mut Velocity)>,
) {
  let dt = time.delta_seconds();
  let [x_min, x_max, y_min, y_max] = dimension_info.boundary();

  for (mut transform, velocity) in query.iter_mut() {
    transform.translation += Vec3::new(velocity.0.x, velocity.0.y, 0.0) * dt;
    transform.translation.x =
      wrap_around(transform.translation.x, x_min, x_max);
    transform.translation.y =
      wrap_around(transform.translation.y, y_min, y_max);
  }
}

fn inspect_buffer(
  mut ctx: EguiContexts,
  mut buffer: Option<ResMut<CameraBuffer>>,
) {
  let buffer = match buffer.as_mut() {
    Some(buffer) => buffer,
    None => return,
  };

  const SCALE: f32 = 0.1;

  let texture_id = ctx
    .image_id(&buffer.image)
    .unwrap_or_else(|| ctx.add_image(buffer.image.clone()));

  bevy_inspector_egui::egui::Window::new("Camera Buffer").show(
    ctx.ctx_mut(),
    |ui| {
      ui.label(format!("Image: {}x{}", buffer.width, buffer.height));
      ui.image(
        texture_id,
        [buffer.width as f32 * SCALE, buffer.height as f32 * SCALE],
      );
    },
  );
}

impl DimensionInfo {
  fn new(width: f32, height: f32) -> Self {
    let viewport_size = (width, height);
    let circle_grid = (10, 10);
    let circle_radius = 10.0;
    let boundary =
      Rect::new(-(width / 2.0), -(height / 2.0), width / 2.0, height / 2.0);

    Self {
      viewport_size,
      circle_grid,
      circle_radius,
      boundary,
    }
  }

  fn resolution(&self) -> bevy::window::WindowResolution {
    let (w, h) = self.viewport_size;
    bevy::window::WindowResolution::new(w, h)
  }

  fn circle_positions(&self) -> impl Iterator<Item = Vec2> {
    let (grid_w, grid_h) = self.circle_grid;
    let Vec2 { x: x0, y: y0 } = self.boundary.min;
    let Vec2 { x: x1, y: y1 } = self.boundary.max;

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
    let Rect { min, max } = self.boundary;
    [min.x, min.y, max.x, max.y]
  }
}

fn wrap_around(v: f32, min: f32, max: f32) -> f32 {
  if v < min {
    max - (min - v)
  } else if v > max {
    min + (v - max)
  } else {
    v
  }
}
