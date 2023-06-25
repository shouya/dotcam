#![feature(portable_simd)]

use bevy::{
  diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
  math::Vec3Swizzles,
  prelude::{
    default, on_event, resource_changed, resource_exists, shape, App, Assets,
    Bundle, Camera2d, Camera2dBundle, Color, Commands, Component, Condition,
    DefaultPlugins, Entity, Handle, Image, In, IntoPipeSystem,
    IntoSystemConfig, Mesh, PluginGroup, Query, Reflect, ReflectResource, Res,
    ResMut, Resource, Transform, Vec2, Vec3,
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
  codecs::jpeg::JpegDecoder,
  imageops::{resize, FilterType},
  GrayImage, ImageBuffer, ImageDecoder, Luma,
};
use image::{DynamicImage, LumaA};
use keyde::KdTree;
use nokhwa::{pixel_format::LumaFormat, CallbackCamera, Camera};

use std::simd::{self, Simd, SimdFloat};

type ScalarField = ImageBuffer<Luma<f32>, Vec<f32>>;
type VectorField = ImageBuffer<LumaA<f32>, Vec<f32>>;

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

#[derive(Resource, Clone, Default)]
struct Preview1(Handle<Image>);

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
      friction_coeff: 0.5,
      repel_coeff: 10.0,
      gradient_scale: 10.0,
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
    .add_plugin(FrameTimeDiagnosticsPlugin)
    .add_plugin(LogDiagnosticsPlugin::default())
    .add_startup_system(setup_webcam)
    .add_startup_system(setup_camera)
    .add_startup_system(setup_circle_bundle.pipe(spawn_circles))
    .add_system(save_camera_output.run_if(resource_exists::<CameraDev>()))
    .add_system(update_image_gradient.run_if(
      resource_exists::<LumaGrid>().and_then(resource_changed::<LumaGrid>()),
    ))
    .add_system(update_repel_gradient)
    .add_system(
      update_velocity
        .after(update_image_gradient)
        .after(update_repel_gradient),
    )
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
  let camera_2d = Camera2d {
    clear_color: bevy::core_pipeline::clear_color::ClearColorConfig::Custom(
      Color::WHITE,
    ),
  };

  commands
    .spawn((Camera2dBundle::default(), GameCamera))
    .insert(camera_2d);
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
    .fliph()
    .into_luma8();

  // image = imageproc::contrast::adaptive_threshold(&image, 10);

  image.into()
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

  let material_handle = materials.add(Color::rgb(0.0, 0.0, 0.0).into());
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

  let [horizontal_gradient, vertical_gradient] =
    calc_image_gradient(luma_grid, 4);

  for (trans, mut grad) in q.iter_mut() {
    let [x, y] = param.translation_to_pixel(&trans.translation);
    let dx = horizontal_gradient.get_pixel(x, y)[0] as f32;
    let dy = vertical_gradient.get_pixel(x, y)[0] as f32;

    let new_grad =
      Vec2::new(dx, dy).normalize_or_zero() * dynamic_param.gradient_scale;

    *grad = ImageGradient(new_grad);
  }
}

fn calc_image_gradient(
  image: &GrayImage,
  n_iter: usize,
) -> [ImageBuffer<Luma<i16>, Vec<i16>>; 2] {
  let (w, h) = (image.width(), image.height());

  let mut horizontal_gradients = Vec::new();
  let mut vertical_gradients = Vec::new();
  let mut image = image.clone();
  for i in 0..n_iter {
    let horz = imageproc::gradients::horizontal_sobel(&image);
    let vert = imageproc::gradients::vertical_sobel(&image);

    horizontal_gradients.push(horz);
    vertical_gradients.push(vert);

    if i == n_iter - 1 {
      // save a resize
      break;
    }

    image = resize(
      &image,
      image.width() / 2,
      image.height() / 2,
      FilterType::Nearest,
    );
  }

  let avg_gradient = |grads: &Vec<ImageBuffer<Luma<i16>, Vec<i16>>>| {
    ImageBuffer::from_fn(w, h, |x, y| {
      let mut grad = 0;
      (0..n_iter).for_each(|i| {
        let new_x = (x / 2_u32.pow(i as u32)).min(grads[i].width() - 1);
        let new_y = (y / 2_u32.pow(i as u32)).min(grads[i].height() - 1);
        grad += grads[i].get_pixel(new_x, new_y)[0];
      });
      Luma([grad / n_iter as i16])
    })
  };
  let horizontal_gradient = avg_gradient(&horizontal_gradients);
  let vertical_gradient = avg_gradient(&vertical_gradients);

  [horizontal_gradient, vertical_gradient]
}

fn update_repel_gradient(
  static_param: Res<StaticParam>,
  dynamic_param: Res<DynamicParam>,
  all_trans: Query<&Transform>,
  mut q: Query<(&Transform, &mut RepelGradient)>,
  all_circles: Res<TrackedCircles>,
) {
  let mut canvas =
    new_scalar_field(static_param.width() as u32, static_param.height() as u32);

  all_trans.iter_many(&all_circles.circles).for_each(|t| {
    let [x, y] = static_param.translation_to_pixel(&t.translation);
    canvas[(x, y)].0[0] -= 0.5;
  });

  const GAUSSIAN: [f32; 9] = [
    1.0 / 16.0,
    2.0 / 16.0,
    1.0 / 16.0,
    2.0 / 16.0,
    4.0 / 16.0,
    2.0 / 16.0,
    1.0 / 16.0,
    2.0 / 16.0,
    1.0 / 16.0,
  ];
  let canvas = filter_3x3_simd(&canvas, &GAUSSIAN);
  // let canvas = filter3x3(&canvas, &GAUSSIAN);
  // let canvas = filter3x3(&canvas, &GAUSSIAN);

  // canvas.save("/tmp/canvas.png").unwrap();

  let gradient = gradient(&canvas);

  // let (a, b) = split_vector_field(&gradient);
  // scalar_field_to_image(&a).save("/tmp/grad_x.png").unwrap();
  // scalar_field_to_image(&b).save("/tmp/grad_y.png").unwrap();

  let mut tot_dx = 0.0;
  let mut tot_dy = 0.0;

  for (trans, mut grad) in q.iter_mut() {
    let [x, y] = static_param.translation_to_pixel(&trans.translation);

    let grad_at_point = gradient.get_pixel(x, y).0;
    let dx = grad_at_point[1];
    let dy = grad_at_point[0];

    tot_dx += dx;
    tot_dy += dy;

    // dbg!(dx, dy);
    let new_grad = -Vec2::new(dx, dy);
    *grad = RepelGradient(new_grad * dynamic_param.repel_coeff);
  }

  dbg!(tot_dx, tot_dy);
}

#[allow(unused)]
fn update_repel_gradient_slow(
  dynamic_param: Res<DynamicParam>,
  all_trans: Query<&Transform>,
  mut q: Query<(&Transform, &mut RepelGradient)>,
  all_circles: Res<TrackedCircles>,
) {
  const RADIUS: f32 = 4.0;

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

    for neigh_id in kdtree.point_indices_within(point, RADIUS) {
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

fn update_velocity(
  time: Res<Time>,
  dyn_param: Res<DynamicParam>,
  mut q: Query<(&mut Velocity, &ImageGradient, &RepelGradient)>,
) {
  for (mut vel, grad, grad2) in q.iter_mut() {
    let force = grad.0 + grad2.0;
    vel.0 += force * time.delta_seconds();
    vel.0 = vel.0.clamp_length_max(dyn_param.max_velocity);
    vel.0 *= (1.0 - dyn_param.friction_coeff).clamp(0.0, 1.0);
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
      circle_grid: (100, 100),
      circle_radius: 2.0,
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

  fn translation_to_pixel_f32(&self, translation: &Vec3) -> [f32; 2] {
    let [x0, x1, y0, y1] = self.boundary();
    let x = (translation.x - x0) / (x1 - x0) * self.width();
    let y = (1.0 - (translation.y - y0) / (y1 - y0)) * self.height();
    [x, y]
  }

  fn translation_to_pixel(&self, translation: &Vec3) -> [u32; 2] {
    let [x, y] = self.translation_to_pixel_f32(translation);
    let x = (x as u32).clamp(0, self.width() as u32 - 1);
    let y = (y as u32).clamp(0, self.height() as u32 - 1);
    [x, y]
  }
}

fn wrap_around<
  N: std::ops::Sub<Output = M>
    + PartialOrd
    + std::ops::AddAssign<M>
    + std::ops::SubAssign<M>
    + Copy,
  M: Copy,
>(
  mut v: N,
  min: N,
  max: N,
) -> N {
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

fn new_scalar_field(w: u32, h: u32) -> ScalarField {
  ImageBuffer::new(w, h)
}

#[allow(unused)]
fn to_scalar_field(image: &GrayImage) -> ScalarField {
  let (w, h) = image.dimensions();
  let vec = image.as_raw().iter().map(|&v| v as f32 / 255.0).collect();
  ImageBuffer::from_vec(w, h, vec).unwrap()
}

fn scalar_field_to_image(buffer: &ScalarField) -> GrayImage {
  let width = buffer.width();
  let height = buffer.height();

  let pixels = buffer
    .pixels()
    .map(|p| (p[0] * 255.0).clamp(0.0, 255.0) as u8)
    .collect();

  ImageBuffer::from_raw(width, height, pixels).unwrap()
}

fn split_vector_field(vec_field: &VectorField) -> (ScalarField, ScalarField) {
  let width = vec_field.width();
  let height = vec_field.height();

  let mut left = new_scalar_field(width, height);
  let mut right = new_scalar_field(width, height);

  for (x, y, pixel) in vec_field.enumerate_pixels() {
    left.put_pixel(x, y, Luma([pixel[0]]));
    right.put_pixel(x, y, Luma([pixel[1]]));
  }

  (left, right)
}

fn gradient(image: &ScalarField) -> VectorField {
  const HORIZONTAL_KERNEL: [f32; 9] =
    [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
  const VERTICAL_KERNEL: [f32; 9] =
    [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];
  let horz = filter_3x3_simd(image, &HORIZONTAL_KERNEL).into_raw();
  let vert = filter_3x3_simd(image, &VERTICAL_KERNEL).into_raw();
  let vec = horz
    .into_iter()
    .zip(vert.into_iter())
    .flat_map(|(x, y)| [x, y])
    .collect();
  ImageBuffer::from_vec(image.width(), image.height(), vec).unwrap()
}

// apply 3x3 filter to an image. wrap around on edges.
fn filter_3x3(image: &ScalarField, kernel: &[f32; 9]) -> ScalarField {
  // The kernel's input positions relative to the current pixel.
  const TAPS: [(isize, isize); 9] = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (0, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
  ];

  let width = image.width() as isize;
  let height = image.height() as isize;

  let mut result = new_scalar_field(width as u32, height as u32);

  for (x, y, _pixel) in image.enumerate_pixels() {
    let mut sum = 0.0;

    for (i, (dx, dy)) in TAPS.iter().enumerate() {
      let px = ((x as isize + dx) % width + width) % width;
      let py = ((y as isize + dy) % height + height) % height;
      let pixel = image.get_pixel(px as u32, py as u32)[0];
      sum += pixel * kernel[i];
    }

    result.put_pixel(x, y, Luma([sum]));
  }

  result
}

fn filter_3x3_simd(image: &ScalarField, kernel: &[f32; 9]) -> ScalarField {
  // The kernel's input positions relative to the current pixel.
  const TAPS_X: Simd<i32, 8> = Simd::from_array([-1, 0, 1, -1, 1, -1, 0, 1]);
  const TAPS_Y: Simd<i32, 8> = Simd::from_array([-1, -1, -1, 0, 0, 1, 1, 1]);

  let width = image.width() as isize;
  let height = image.height() as isize;

  let width_simd = Simd::from_array([width as i32; 8]);
  let height_simd = Simd::from_array([height as i32; 8]);

  let kernel_simd =
    Simd::gather_or_default(kernel, Simd::from_array([0, 1, 2, 3, 5, 6, 7, 8]));
  let kernel_center = kernel[4];

  let mut result = new_scalar_field(width as u32, height as u32);
  let buffer = image.as_raw().as_slice();

  for (x, y, pixel) in image.enumerate_pixels() {
    let mut xs = Simd::from_array([x as i32; 8]) + TAPS_X;
    let mut ys = Simd::from_array([y as i32; 8]) + TAPS_Y;

    xs = (xs % width_simd + width_simd) % width_simd;
    ys = (ys % height_simd + height_simd) % height_simd;

    let indices = (xs + ys * width_simd).cast();

    let pixels: Simd<f32, 8> = Simd::gather_or_default(buffer, indices);
    let sum = (pixels * kernel_simd).reduce_sum() + pixel.0[0] * kernel_center;

    result.put_pixel(x, y, Luma([sum]));
  }

  result
}

#[cfg(disabled)]
fn filter_3x3(image: &ScalarField, kernel: &[f32; 9]) -> ScalarField {
  const TAPS: [(isize, isize); 9] = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
  ]
  .to;

  let width = image.width() as isize;
  let height = image.height() as isize;

  let mut result = image.clone();

  result
    .enumerate_pixels_mut()
    .par_bridge()
    .for_each(|(x, y, pixel)| {
      let mut sum = 0.0;

      for (i, (dx, dy)) in TAPS.iter().enumerate() {
        // simpler wrap_around
        let px = ((x as isize + dx) % width + width) % width;
        let py = ((y as isize + dy) % height + height) % height;
        let pixel = image.get_pixel(px as u32, py as u32)[0];
        sum += pixel * kernel[i];
      }

      *pixel = Luma([sum]);
    });

  result
}
