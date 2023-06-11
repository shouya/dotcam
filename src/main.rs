use bevy::{
  prelude::{
    default, App, DefaultPlugins, Image, NonSendMut, PluginGroup, ResMut,
    Resource, World,
  },
  render::render_resource::{Extent3d, TextureDimension, TextureFormat},
  window::{Window, WindowPlugin},
};
use bevy_inspector_egui::quick::ResourceInspectorPlugin;
use bevy_reflect::Reflect;
use image::codecs::jpeg::JpegDecoder;
use nokhwa::{
  pixel_format::{LumaFormat, RgbFormat},
  Camera,
};

#[derive(Resource, Reflect, Default)]
struct CameraBuffer {
  image: Image,
  width: u32,
  height: u32,
}

fn main() {
  let plugins = DefaultPlugins.set(WindowPlugin {
    primary_window: Some(Window {
      title: "Window".into(),
      resolution: (500., 500.).into(),
      ..default()
    }),
    ..default()
  });

  App::new()
    .add_plugins(plugins)
    .add_startup_system(setup_camera)
    .add_system(update_camera_output)
    .add_plugin(ResourceInspectorPlugin::<CameraBuffer>::default())
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
  let format = TextureFormat::R8Unorm;
  let image = Image::new_fill(extent, dimension, &[0], format);

  world.insert_resource(CameraBuffer {
    image,
    width,
    height,
  });
}

fn update_camera_output(
  mut camera: Option<NonSendMut<Camera>>,
  mut buffer: Option<ResMut<CameraBuffer>>,
) {
  use nokhwa::utils::FrameFormat::MJPEG;

  let camera = match camera.as_mut() {
    Some(camera) => camera,
    None => return,
  };
  let image = match buffer.as_mut() {
    Some(buffer) => &mut buffer.image,
    None => return,
  };

  let frame = camera.frame().unwrap();
  debug_assert!(frame.source_frame_format() == MJPEG);

  let mut decoder = JpegDecoder::new(&frame.buffer_bytes()).unwrap();

  debug_assert!(decoder.colortype() == image::ColorType: Rgb8);

  decoder
    .into_reader()
    .bytes()
    .chunks(3)
    .zip(image.data.iter_mut())
    .for_each(|(rgb, luma)| {
      let rgb = rgb.unwrap();
      *luma = rgb[0];
    });
}
