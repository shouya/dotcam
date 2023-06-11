use std::io::Read;

use bevy::{
  prelude::{
    default, App, Assets, DefaultPlugins, Handle, Image, NonSendMut,
    PluginGroup, Reflect, ReflectResource, ResMut, Resource, World,
  },
  render::render_resource::{Extent3d, TextureDimension, TextureFormat},
  window::{Window, WindowPlugin},
};
use bevy_egui::{EguiContexts, EguiPlugin};
use image::{codecs::jpeg::JpegDecoder, ImageDecoder};
use itertools::Itertools;
use nokhwa::{pixel_format::LumaFormat, Camera};

#[derive(Resource, Reflect, Default)]
#[reflect(Resource)]
struct CameraBuffer {
  image: Handle<Image>,
  width: u32,
  height: u32,
}

fn main() {
  let plugins = DefaultPlugins.set(WindowPlugin {
    primary_window: Some(Window {
      title: "Dotcam".into(),
      resolution: (500., 500.).into(),
      ..default()
    }),
    ..default()
  });

  App::new()
    .add_plugins(plugins)
    .add_plugin(EguiPlugin)
    .register_type::<CameraBuffer>()
    .add_startup_system(setup_camera)
    .add_system(update_camera_output)
    .add_system(inspect_buffer)
    .run();
}

fn setup_camera(world: &mut World) {
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
