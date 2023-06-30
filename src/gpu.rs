use bevy::prelude::{App, Assets, Image, Plugin, Res, ResMut};
use safe_transmute::transmute_to_bytes;

use crate::camera_feed::CameraStream;

use self::downscaler::{DownscalerPlugin, ImageDownscaler, TagLike};

mod downscaler;
mod gradiator;

#[derive(Default)]
pub struct DotCamPlugin;

#[derive(Default, Clone)]
struct CameraDownscalerTag;

impl TagLike for CameraDownscalerTag {}

impl Plugin for DotCamPlugin {
  fn build(&self, app: &mut App) {
    let downscaler = DownscalerPlugin::<CameraDownscalerTag>::default();

    app
      .add_plugin(downscaler)
      .add_system(copy_camera_stream_to_downscaler);
  }
}

fn copy_camera_stream_to_downscaler(
  mut images: ResMut<Assets<Image>>,
  downscaler: Res<ImageDownscaler<CameraDownscalerTag>>,
  camera_stream_res: Res<CameraStream>,
) {
  let camera_stream = images.get(&camera_stream_res.0).unwrap();
  let data = vec_u8_to_vec_f32norm(&camera_stream.data);
  let downscaler_input = images.get_mut(&downscaler.textures[0]).unwrap();

  downscaler_input
    .data
    .copy_from_slice(transmute_to_bytes(data.as_slice()));
}

pub fn vec_u8_to_vec_f32norm(src: &[u8]) -> Vec<f32> {
  src.iter().map(|&byte| byte as f32 / 255.0).collect()
}
