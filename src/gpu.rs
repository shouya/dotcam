use bevy::{
  prelude::{
    default, not, resource_changed, resource_exists, shape, App, Assets, Color,
    Commands, Condition, DetectChangesMut, Entity, EventReader, EventWriter,
    Handle, Image, IntoSystemConfig, Mesh, Plugin, Query, Res, ResMut,
    Resource, Transform, Vec3,
  },
  render::render_resource::{
    Extent3d, TextureDimension, TextureFormat, TextureUsages,
  },
  sprite::{ColorMaterial, MaterialMesh2dBundle, Mesh2dHandle},
};

use self::choreographer::{
  Choreographer, ChoreographerInput, ChoreographerOutput, ChoreographerPlugin,
};
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
      .add_plugin(ChoreographerPlugin)
      .add_startup_system(spawn_dots)
      .add_startup_system(setup)
      .add_startup_system(setup_choreographer_tapouts)
      .add_system(
        initialize_camera_texture.run_if(
          resource_changed::<CameraStream>()
            .and_then(not(resource_exists::<CameraTexture>())),
        ),
      )
      .add_system(
        update_camera_texture.run_if(
          resource_changed::<CameraStream>()
            .and_then(resource_exists::<CameraTexture>()),
        ),
      )
      .add_system(update_dot_transforms)
      .add_system(
        send_choreographer_input.run_if(
          resource_exists::<CameraTexture>()
            .and_then(resource_changed::<CameraTexture>()),
        ),
      );
  }
}

#[derive(Resource)]
pub struct CameraTexture(pub Handle<Image>);

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

fn setup_choreographer_tapouts(
  mut commands: Commands,
  mut images: ResMut<Assets<Image>>,
) {
  let rf32 = TextureFormat::R32Float;
  let rgf32 = TextureFormat::Rg32Float;

  Choreographer::tap(
    &mut commands,
    &mut images,
    "downscaler_camera_1_output",
    (256, 256),
    rf32,
  );

  Choreographer::tap(
    &mut commands,
    &mut images,
    "downscaler_camera_2_output",
    (128, 128),
    rf32,
  );

  Choreographer::tap(
    &mut commands,
    &mut images,
    "gradiator_camera_output",
    (512, 512),
    rgf32,
  );
}

fn update_dot_transforms(
  mut outputs: EventReader<ChoreographerOutput>,
  tracker: Res<DotsTracker>,
  mut q: Query<&mut Transform>,
) {
  let Some(output) = outputs.iter().last() else {
    return;
  };

  for (i, entity) in tracker.dots.iter().enumerate() {
    let mut transform = q.get_mut(*entity).unwrap();
    let translation = Vec3::new(
      output.dot_locations[i].x - 256.0,
      256.0 - output.dot_locations[i].y,
      0.0,
    );
    transform.translation = translation;
  }
}

fn setup(mut _commands: Commands) {}

fn initialize_camera_texture(
  mut commands: Commands,
  mut images: ResMut<Assets<Image>>,
  camera_stream_res: Res<CameraStream>,
) {
  let camera_stream = images.get(&camera_stream_res.0).unwrap();
  let w = camera_stream.texture_descriptor.size.width;
  let h = camera_stream.texture_descriptor.size.height;
  let size = Extent3d {
    width: w,
    height: h,
    depth_or_array_layers: 1,
  };
  let dimension = TextureDimension::D2;
  let format = TextureFormat::R32Float;

  let data = vec_u8_to_vec_f32norm(&camera_stream.data);
  let mut image = Image::new(size, dimension, data, format);
  image.texture_descriptor.usage |= TextureUsages::STORAGE_BINDING;
  let handle = images.add(image);
  commands.insert_resource(CameraTexture(handle));
}

fn update_camera_texture(
  mut images: ResMut<Assets<Image>>,
  camera_stream_res: Res<CameraStream>,
  mut camera_texture: ResMut<CameraTexture>,
) {
  let camera_stream = images.get(&camera_stream_res.0).unwrap();
  let data = vec_u8_to_vec_f32norm(&camera_stream.data);
  let buffer = &mut images.get_mut(&camera_texture.0).unwrap().data;
  buffer.copy_from_slice(&data);

  // set the resource as changed so we can base run conditions on it.
  camera_texture.set_changed();
}

fn send_choreographer_input(
  mut inputs: EventWriter<ChoreographerInput>,
  camera_texture: Res<CameraTexture>,
) {
  let input = ChoreographerInput {
    camera_feed: camera_texture.0.clone(),
  };
  inputs.send(input);
}

// return bytes
pub fn vec_u8_to_vec_f32norm(src: &[u8]) -> Vec<u8> {
  src
    .iter()
    .map(|&byte| byte as f32 / 255.0)
    .flat_map(|f| f.to_ne_bytes())
    .collect()
}
