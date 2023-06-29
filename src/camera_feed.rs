use bevy::{
  prelude::{
    on_event, App, Assets, DetectChangesMut, FromWorld, Handle, Image,
    IntoSystemConfig, Plugin, Res, ResMut, Resource, World,
  },
  render::render_resource::{Extent3d, TextureDimension, TextureFormat},
  window::WindowCloseRequested,
};
use bevy_egui::EguiContexts;
use crossbeam::channel::{self, Receiver};
use image::{codecs::jpeg::JpegDecoder, DynamicImage, ImageDecoder};
use nokhwa::{pixel_format::LumaFormat, CallbackCamera, Camera};

use crate::StaticParam;

pub struct CameraFeedPlugin {
  // target will be tagged with
  pub inspect_webcam: bool,
}

impl Default for CameraFeedPlugin {
  fn default() -> Self {
    Self {
      inspect_webcam: true,
    }
  }
}

impl Plugin for CameraFeedPlugin {
  fn build(&self, app: &mut App) {
    let app = app
      .init_resource::<CameraDev>()
      .init_resource::<CameraStream>()
      .add_system(save_camera_output)
      .add_system(stop_webcam.run_if(on_event::<WindowCloseRequested>()));

    if self.inspect_webcam {
      app.add_system(preview_output);
    }
  }
}

#[derive(Resource)]
struct CameraDev {
  #[allow(unused)]
  camera: CallbackCamera,
  size: [u32; 2],
  receiver: Receiver<DynamicImage>,
}

#[derive(Resource, Clone)]
pub struct CameraStream(pub Handle<Image>);

impl FromWorld for CameraDev {
  fn from_world(world: &mut World) -> Self {
    use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};
    let param = world.get_resource::<StaticParam>().unwrap();
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

    CameraDev {
      camera,
      receiver,
      size,
    }
  }
}

impl FromWorld for CameraStream {
  fn from_world(world: &mut World) -> Self {
    let camera_dev = world.resource::<CameraDev>();

    let [w, h] = camera_dev.size;
    let format = TextureFormat::R8Unorm;
    let extent = Extent3d {
      width: w,
      height: h,
      depth_or_array_layers: 1,
    };
    let image = Image::new_fill(extent, TextureDimension::D2, &[0], format);

    let mut images = world.resource_mut::<Assets<Image>>();
    let handle = images.add(image);

    CameraStream(handle)
  }
}

fn save_camera_output(
  camera_dev: Res<CameraDev>,
  mut stream: ResMut<CameraStream>,
  mut images: ResMut<Assets<Image>>,
) {
  let receiver = &camera_dev.receiver;
  let Ok(image) = receiver.try_recv() else {
    return;
  };

  let stream_buffer = images.get_mut(&stream.0).unwrap();
  stream_buffer.data.copy_from_slice(image.as_bytes());

  // stream itself (holding a handle) was not mutated, so technically
  // it doesn't count as changed. we need to manually set it to
  // changed.
  stream.set_changed();
}

fn preview_output(mut ctx: EguiContexts, stream: Res<CameraStream>) {
  let texture_id = ctx
    .image_id(&stream.0)
    .unwrap_or_else(|| ctx.add_image(stream.0.clone_weak()));

  bevy_inspector_egui::egui::Window::new("Camera Buffer").show(
    ctx.ctx_mut(),
    |ui| {
      ui.image(texture_id, [100.0, 100.0]);
    },
  );
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

fn stop_webcam(mut _webcam: ResMut<CameraDev>) {
  // it's unable to acquire lock from another thread to stop
  // the camera stream. resulting the program to hang.
  //
  // this hack is simply exit the program.
  std::process::exit(0);
}
