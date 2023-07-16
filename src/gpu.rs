use bevy::{
  prelude::{
    default, resource_changed, shape, App, Assets, Color, Commands, Component,
    Entity, FromWorld, Image, IntoSystemConfig, Mesh, Plugin, Query, Res,
    ResMut, Resource, Transform, Vec2, With, World,
  },
  render::render_resource::{
    Extent3d, TextureDimension, TextureFormat, TextureUsages,
  },
  sprite::{ColorMaterial, MaterialMesh2dBundle, Mesh2dHandle},
};

use self::gradiator::{Gradiator, GradiatorPlugin};
use crate::{camera_feed::CameraStream, StaticParam};

mod choreographer;
mod downscaler;
mod gradiator;
mod shaders;

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

#[derive(Resource, Clone)]
pub struct Dots {
  locations: Vec<Vec2>,
  velocities: Vec<Vec2>,
}

impl FromWorld for Dots {
  fn from_world(world: &mut World) -> Self {
    let mut dots = Dots {
      locations: Vec::new(),
      velocities: Vec::new(),
    };
    let param = world.resource::<StaticParam>();

    for pos in param.circle_positions() {
      dots.locations.push(pos);
      dots.velocities.push(Vec2::ZERO);
    }

    dots
  }
}

#[derive(Resource, Clone)]
pub struct DotsTracker {
  dots: Vec<Entity>,
}

fn spawn_dots(
  mut commands: Commands,
  param: Res<StaticParam>,
  mut meshes: ResMut<Assets<Mesh>>,
  mut materials: ResMut<Assets<ColorMaterial>>,
) {
  let mut dot_entities = Vec::new();
  let mesh: Mesh2dHandle = meshes
    .add(Mesh::from(shape::Circle {
      radius: param.circle_radius,
      ..default()
    }))
    .into();

  let material = materials.add(Color::rgb(0.0, 0.0, 0.0).into());

  for _ in param.circle_positions() {
    let entity = commands
      .spawn(MaterialMesh2dBundle {
        mesh: mesh.clone(),
        material: material.clone(),
        ..default()
      })
      .id();

    dot_entities.push(entity);
  }

  commands.insert_resource(DotsTracker { dots: dot_entities });
}

fn update_dot_position(
  dots: Res<Dots>,
  tracker: Res<DotsTracker>,
  mut q: Query<&mut Transform>,
) {
  for (i, entity) in tracker.dots.iter().enumerate() {
    let mut transform = q.get_mut(*entity).unwrap();
    transform.translation = dots.locations[i].extend(0.0);
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
