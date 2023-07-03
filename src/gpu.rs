use bevy::{
  prelude::{
    resource_changed, App, Assets, Commands, Image, IntoSystemConfig, Plugin,
    Query, Res, ResMut,
  },
  render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};

use self::downscaler::{Downscaler, DownscalerPlugin};
use crate::{camera_feed::CameraStream, StaticParam};

mod downscaler;
mod gradiator;

#[derive(Default)]
pub struct DotCamPlugin;

impl Plugin for DotCamPlugin {
  fn build(&self, app: &mut App) {
    app
      .add_plugin(DownscalerPlugin::default())
      .add_startup_system(setup_downscaler)
      .add_system(
        copy_camera_stream_to_downscaler
          .run_if(resource_changed::<CameraStream>()),
      );
  }
}

fn setup_downscaler(mut commands: Commands, static_param: Res<StaticParam>) {
  let w = static_param.width() as u32;
  let h = static_param.height() as u32;
  let downscaler = Downscaler::new(4, (w, h));
  commands.spawn(downscaler);
}

fn copy_camera_stream_to_downscaler(
  mut images: ResMut<Assets<Image>>,
  mut q: Query<&mut Downscaler>,
  camera_stream_res: Res<CameraStream>,
) {
  let camera_stream = images.get(&camera_stream_res.0).unwrap();
  let mut downscaler = q.single_mut();

  // still computing, let's wait till next frame
  if downscaler.is_initialized() && !downscaler.is_ready() {
    return;
  }

  let data = vec_u8_to_vec_f32norm(&camera_stream.data);

  if !downscaler.is_initialized() {
    let w = camera_stream.texture_descriptor.size.width;
    let h = camera_stream.texture_descriptor.size.height;
    let size = Extent3d {
      width: w,
      height: h,
      depth_or_array_layers: 1,
    };
    let dimension = TextureDimension::D2;
    let format = TextureFormat::R32Float;
    let image = Image::new(size, dimension, data, format);
    downscaler.init(image, &mut images);
    return;
  };

  if !downscaler.is_ready() {
    return;
  }

  downscaler.set_input(&data, &mut images);
}

// return bytes
pub fn vec_u8_to_vec_f32norm(src: &[u8]) -> Vec<u8> {
  src
    .iter()
    .map(|&byte| byte as f32 / 255.0)
    .flat_map(|f| f.to_ne_bytes())
    .collect()
}
