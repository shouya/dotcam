use bevy::{
  prelude::{
    on_event, App, Assets, DetectChangesMut, FromWorld, Handle, Image,
    IntoSystemConfig, Plugin, Res, ResMut, Resource, World,
  },
  render::render_resource::{Extent3d, TextureDimension, TextureFormat},
  window::WindowCloseRequested,
};
#[cfg(feature = "inspector")]
use bevy_egui::EguiContexts;
use crossbeam::channel::{self, Receiver};
use image::{DynamicImage, Luma};
use nokhwa::{
  pixel_format::LumaFormat,
  utils::{frame_formats, CameraFormat, CameraIndex, FrameFormat, Resolution},
  CallbackCamera, Camera, FormatDecoder, NokhwaError,
};

use crate::StaticParam;

pub struct CameraFeedPlugin {
  #[cfg(feature = "inspector")]
  pub inspect_webcam: bool,
}

impl Default for CameraFeedPlugin {
  fn default() -> Self {
    Self {
      #[cfg(feature = "inspector")]
      inspect_webcam: true,
    }
  }
}

impl Plugin for CameraFeedPlugin {
  fn build(&self, app: &mut App) {
    app
      .init_resource::<CameraDev>()
      .init_resource::<CameraStream>()
      .add_system(save_camera_output)
      .add_system(stop_webcam.run_if(on_event::<WindowCloseRequested>()));

    #[cfg(feature = "inspector")]
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

fn pick_camera_format(
  index: CameraIndex,
  target_size: (u32, u32),
) -> Option<CameraFormat> {
  use nokhwa::utils::{RequestedFormat, RequestedFormatType};

  let all_formats = {
    let initial_req_format =
      RequestedFormat::new::<LumaFormat>(RequestedFormatType::None);
    let mut camera = Camera::new(index, initial_req_format).ok()?;
    camera.compatible_camera_formats().ok()?
  };

  all_formats.into_iter().max_by_key(|format| {
    // best format is one that's faster to decode
    let format_score = match format.format() {
      FrameFormat::GRAY => 2,
      FrameFormat::YUYV => 1,
      _ => 0,
    };

    // best resolution is matching resolution
    let resolution_score = {
      let [w, h] = [format.resolution().width(), format.resolution().height()];
      let w_score = (target_size.0 as i32 - w as i32).abs();
      let h_score = (target_size.1 as i32 - h as i32).abs();
      -(w_score + h_score)
    };

    // best frame rate is highest frame rate
    let frame_rate_score = format.frame_rate();

    (format_score, resolution_score, frame_rate_score)
  })
}

impl FromWorld for CameraDev {
  fn from_world(world: &mut World) -> Self {
    use nokhwa::utils::{RequestedFormat, RequestedFormatType};
    let param = world.get_resource::<StaticParam>().unwrap();
    let index = CameraIndex::Index(0);

    let format = pick_camera_format(
      index.clone(),
      (param.width() as u32, param.height() as u32),
    )
    .unwrap();
    let req_format = RequestedFormat::new::<FastLumaDecoder>(
      RequestedFormatType::Exact(format),
    );
    let camera = Camera::new(index, req_format).unwrap();

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

#[cfg(feature = "inspector")]
fn preview_output(mut ctx: EguiContexts, stream: Res<CameraStream>) {
  let texture_id = ctx
    .image_id(&stream.0)
    .unwrap_or_else(|| ctx.add_image(stream.0.clone_weak()));

  bevy_inspector_egui::egui::Window::new("Camera Buffer")
    .show(ctx.ctx_mut(), |ui| ui.image(texture_id, [100.0, 100.0]));
}

fn camera_buffer_to_image(
  buffer: nokhwa::Buffer,
  size: [u32; 2],
) -> DynamicImage {
  let image: DynamicImage =
    buffer.decode_image::<FastLumaDecoder>().unwrap().into();

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

#[derive(Clone, Copy)]
struct FastLumaDecoder;

impl FormatDecoder for FastLumaDecoder {
  type Output = Luma<u8>;

  const FORMATS: &'static [FrameFormat] = frame_formats();

  fn write_output(
    fcc: FrameFormat,
    resolution: Resolution,
    data: &[u8],
  ) -> Result<Vec<u8>, NokhwaError> {
    match fcc {
      FrameFormat::YUYV => {
        Ok(data.chunks_exact(2).map(|s| s[0]).collect::<Vec<u8>>())
      }
      _ => LumaFormat::write_output(fcc, resolution, data),
    }
  }

  fn write_output_buffer(
    fcc: FrameFormat,
    _resolution: Resolution,
    data: &[u8],
    dest: &mut [u8],
  ) -> Result<(), NokhwaError> {
    match fcc {
      FrameFormat::YUYV => {
        data
          .chunks_exact(2)
          .zip(dest.iter_mut())
          .for_each(|(s, d)| {
            *d = s[0];
          });
        Ok(())
      }
      _ => LumaFormat::write_output_buffer(fcc, _resolution, data, dest),
    }
  }
}
