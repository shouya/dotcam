use bevy::{
  prelude::{
    resource_changed, App, Assets, Commands, Component, Entity, Image,
    IntoSystemConfig, Plugin, Query, Res, ResMut, With,
  },
  render::render_resource::{
    Extent3d, TextureDimension, TextureFormat, TextureUsages,
  },
};

use self::gradiator::{Gradiator, GradiatorPlugin};
use crate::camera_feed::CameraStream;

mod downscaler;
mod gradiator;

#[derive(Default)]
pub struct DotCamPlugin;

impl Plugin for DotCamPlugin {
  fn build(&self, app: &mut App) {
    app
      .add_plugin(GradiatorPlugin::default())
      .add_startup_system(setup)
      .add_system(
        copy_camera_stream_to_gradiator
          .run_if(resource_changed::<CameraStream>()),
      );
  }
}

#[derive(Component)]
struct CameraPipeline;

fn setup(mut commands: Commands) {
  commands.spawn(CameraPipeline);
}

fn copy_camera_stream_to_gradiator(
  mut commands: Commands,
  mut images: ResMut<Assets<Image>>,
  mut q: Query<(Entity, Option<&mut Gradiator>), With<CameraPipeline>>,
  camera_stream_res: Res<CameraStream>,
) {
  let camera_stream = images.get(&camera_stream_res.0).unwrap();
  let (entity, downscaler) = q.single_mut();

  let data = vec_u8_to_vec_f32norm(&camera_stream.data);

  match downscaler {
    None => {
      // initialize the downscaler
      let w = camera_stream.texture_descriptor.size.width;
      let h = camera_stream.texture_descriptor.size.height;
      let size = Extent3d {
        width: w,
        height: h,
        depth_or_array_layers: 1,
      };
      let dimension = TextureDimension::D2;
      let format = TextureFormat::R32Float;
      let mut image = Image::new(size, dimension, data, format);
      image.texture_descriptor.usage |= TextureUsages::STORAGE_BINDING;
      let handle = images.add(image);
      commands
        .entity(entity)
        .insert(Gradiator::new(handle, &mut images));
    }
    Some(downscaler) => {
      let buffer = &mut images.get_mut(downscaler.input()).unwrap().data;
      buffer.copy_from_slice(&data);
    }
  }
}

// return bytes
pub fn vec_u8_to_vec_f32norm(src: &[u8]) -> Vec<u8> {
  src
    .iter()
    .map(|&byte| byte as f32 / 255.0)
    .flat_map(|f| f.to_ne_bytes())
    .collect()
}
