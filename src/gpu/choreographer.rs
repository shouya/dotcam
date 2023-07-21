use std::mem;

use bevy::{
  prelude::{
    App, Assets, Commands, Component, EventReader, EventWriter, Handle, Image,
    IntoSystemConfig, Plugin, Query, Res, ResMut, Resource, UVec2, Vec2, World,
  },
  render::{
    render_resource::{Extent3d, TextureDimension, TextureFormat},
    texture::TextureFormatPixelInfo,
  },
  time::Time,
};
use bevy_app_compute::prelude::{
  AppComputePlugin, AppComputeWorker, AppComputeWorkerBuilder,
  AppComputeWorkerPlugin, ComputeWorker,
};
use bevy_egui::{egui, EguiContext, EguiUserTextures};

use crate::StaticParam;

use super::shaders;

pub struct ChoreographerPlugin;

impl Plugin for ChoreographerPlugin {
  fn build(&self, app: &mut App) {
    app
      .add_plugin(AppComputePlugin)
      .add_plugin(AppComputeWorkerPlugin::<Choreographer>::default())
      .add_event::<ChoreographerInput>()
      .add_event::<ChoreographerOutput>()
      .add_system(process_output_system)
      .add_system(process_tapout_system.after(process_output_system))
      .add_system(
        process_input_system
          .after(process_output_system)
          .after(process_tapout_system),
      )
      .add_system(inspect_tapout_system.after(process_tapout_system));
  }
}

pub struct ChoreographerInput {
  // texture format: R32Float
  pub camera_feed: Handle<Image>,
}

pub struct ChoreographerOutput {
  pub dot_locations: Vec<Vec2>,
}

#[derive(Resource)]
pub struct Choreographer;

const WG_SIZE: u32 = 8;
const F32_SIZE: u32 = mem::size_of::<f32>() as u32;

// inputs:
//  dt - f32
//  dots_locations - Vec<Vec2>
//  dots_velocities - Vec<Vec2>
//  camera_input - Vec<f32> (WxH)

//
// outputs:
//  dots_new_locations - Vec<Vec2>
//  dots_new_velocities - Vec<Vec2>

#[derive(Component)]
struct Tapout {
  name: String,
  image: Handle<Image>,
}

impl Choreographer {
  pub fn tap(
    commands: &mut Commands,
    images: &mut Assets<Image>,
    name: &str,
    size: (u32, u32),
    format: TextureFormat,
  ) -> Handle<Image> {
    let size = Extent3d {
      width: size.0,
      height: size.1,
      depth_or_array_layers: 1,
    };
    let dimension = TextureDimension::D2;
    let pixel = vec![0; format.pixel_size()];
    let image = Image::new_fill(size, dimension, pixel.as_slice(), format);
    let handle = images.add(image);

    let tapout = Tapout {
      name: name.to_string(),
      image: handle.clone(),
    };

    commands.spawn(tapout);

    handle
  }

  fn build_downscaler(
    builder: &mut AppComputeWorkerBuilder<'_, Self>,
    prefix: &str,
    iterations: usize,
    mut input_size: UVec2,
  ) {
    let input_name = format!("{}_input", prefix);
    let pixel_count = (input_size.x * input_size.y) as u64;
    // builder.add_rw_storage(&input_name, &[0f32; pixel_count]);
    // builder.add_staging(&input_name, &vec![0f32; pixel_count]);
    builder.add_empty_staging(&input_name, pixel_count * 4);

    for i in 1..=iterations {
      let input_name = if i == 1 {
        input_name.clone()
      } else {
        format!("downscaler_{}_{}_output", prefix, i - 1)
      };

      let name = |n| format!("downscaler_{}_{}_{}", prefix, i, n);
      let output_size = input_size / 2;
      let output_bytes = (output_size.x * output_size.y * F32_SIZE) as u64;

      builder.add_storage(&name("input_size"), &input_size);
      builder.add_storage(&name("output_size"), &output_size);
      // builder.add_empty_rw_storage(&name("output"), output_bytes);
      builder.add_empty_staging(&name("output"), output_bytes);

      let wg_count = output_size / UVec2::new(WG_SIZE, WG_SIZE);

      builder.add_pass::<shaders::DownscalerShader>(
        [wg_count.x, wg_count.y, 1],
        &[
          &name("input_size"),
          &name("output_size"),
          &input_name,
          &name("output"),
        ],
      );

      input_size = output_size;
    }
  }

  fn build_gradiator(
    builder: &mut AppComputeWorkerBuilder<'_, Self>,
    prefix: &str,
    iterations: usize,
    input_size: UVec2,
  ) {
    let input_size_name = format!("gradiator_{}_input_size", prefix);
    builder.add_storage(&input_size_name, &input_size);

    let output_name = format!("gradiator_{}_output", prefix);
    let output_bytes = (input_size.x * input_size.y * F32_SIZE * 2) as u64;
    // builder.add_empty_rw_storage(&output_name, output_bytes);
    builder.add_empty_staging(&output_name, output_bytes);

    let downscaler_input_name = format!("{}_input", prefix);
    let mut input_vars =
      vec![input_size_name, output_name, downscaler_input_name];

    for i in 1..iterations {
      let downscaler_output_name =
        format!("downscaler_{}_{}_output", prefix, i);
      input_vars.push(downscaler_output_name);
    }

    let wg_count = input_size / UVec2::new(WG_SIZE, WG_SIZE);

    let input_vars_str =
      input_vars.iter().map(|s| s.as_str()).collect::<Vec<_>>();

    builder.add_pass::<shaders::GradiatorShader>(
      [wg_count.x, wg_count.y, 1],
      &input_vars_str,
    );
  }

  fn build_dotpainter(
    builder: &mut AppComputeWorkerBuilder<'_, Self>,
    radius: u32,
    input_size: UVec2,
    param: &StaticParam,
  ) {
    let pixel_count = (input_size.x * input_size.y) as usize;
    builder.add_staging("painted_dots", &vec![0f32; pixel_count]);
    builder.add_storage("dotpainter_radius", &radius);

    let circle_count = param.circle_positions_pos().count();
    builder.add_pass::<shaders::DotpainterShader>(
      [circle_count as u32 / WG_SIZE, 1, 1],
      &[
        "dotpainter_radius",
        // provided by choreographer
        "input_size",
        // provided by choreographer
        "dots_locations",
        "dots_input",
      ],
    );
  }

  fn build_choreographer(
    builder: &mut AppComputeWorkerBuilder<'_, Self>,
    input_size: UVec2,
    param: &StaticParam,
  ) {
    builder.add_storage("dt", &0.01667f32);
    builder.add_storage("input_size", &input_size);

    let locations: Vec<Vec2> = param.circle_positions_pos().collect();
    builder.add_staging("dots_locations", &locations);

    let circle_count = locations.len();
    builder.add_staging("dots_velocities", &vec![Vec2::ZERO; circle_count]);

    builder.add_staging("dots_new_locations", &vec![Vec2::ZERO; circle_count]);
    builder.add_staging("dots_new_velocities", &vec![Vec2::ZERO; circle_count]);

    builder.add_pass::<shaders::ChoreographerShader>(
      [circle_count as u32 / WG_SIZE, 1, 1],
      &[
        "dt",
        "input_size",
        "gradiator_camera_output",
        "gradiator_dots_output",
        "dots_locations",
        "dots_velocities",
        "dots_new_locations",
        "dots_new_velocities",
      ],
    );
  }
}

impl ComputeWorker for Choreographer {
  fn build(world: &mut World) -> AppComputeWorker<Self> {
    let static_param = (*world.resource::<StaticParam>()).clone();
    let input_width = static_param.width() as u32;
    let input_height = static_param.height() as u32;

    let input_size = UVec2::new(input_width, input_height);

    let mut builder = AppComputeWorkerBuilder::new(world);

    Self::build_dotpainter(&mut builder, 1, input_size, &static_param);

    Self::build_downscaler(&mut builder, "camera", 5, input_size);
    Self::build_gradiator(&mut builder, "camera", 5, input_size);

    Self::build_downscaler(&mut builder, "dots", 5, input_size);
    Self::build_gradiator(&mut builder, "dots", 5, input_size);

    Self::build_choreographer(&mut builder, input_size, &static_param);

    builder.add_swap("dots_locations", "dots_new_locations");
    builder.add_swap("dots_velocities", "dots_new_velocities");

    builder.build()
  }
}

fn process_input_system(
  time: Res<Time>,
  mut inputs: EventReader<ChoreographerInput>,
  images: Res<Assets<Image>>,
  mut worker: ResMut<AppComputeWorker<Choreographer>>,
) {
  let Some(input) = inputs.iter().last() else {
    return;
  };

  let ChoreographerInput { camera_feed } = input;

  let camera_feed = images.get(camera_feed).unwrap();

  if !worker.ready() {
    return;
  }

  worker.write("dt", &time.delta_seconds());
  worker.write_slice("camera_input", &camera_feed.data);
}

fn process_output_system(
  mut outputs: EventWriter<ChoreographerOutput>,
  static_param: Res<StaticParam>,
  mut worker: ResMut<AppComputeWorker<Choreographer>>,
) {
  if !worker.ready() {
    return;
  }
  let new_locations = worker.read_vec("dots_locations");
  // dbg!(new_locations[10]);

  let pixel_count = (static_param.width() * static_param.height()) as usize;
  worker.write_slice("dots_input", &vec![0f32; pixel_count]);

  let output = ChoreographerOutput {
    dot_locations: new_locations,
  };
  outputs.send(output);
}

fn process_tapout_system(
  worker: Res<AppComputeWorker<Choreographer>>,
  mut images: ResMut<Assets<Image>>,
  tapouts_q: Query<&Tapout>,
) {
  if !worker.ready() {
    return;
  }

  for tapout in tapouts_q.iter() {
    let data = worker.read_vec(&tapout.name);
    let image = images.get_mut(&tapout.image).unwrap();
    if image.data.len() != data.len() {
      panic!("Image size mismatch: {}", tapout.name);
    }
    image.data.copy_from_slice(&data);
  }
}

fn inspect_tapout_system(
  mut textures: ResMut<EguiUserTextures>,
  mut ctx: Query<&mut EguiContext>,
  tapouts_q: Query<&Tapout>,
) {
  let mut binding = ctx.single_mut();
  let ctx = binding.get_mut();
  let mut texture_id = |handle| {
    textures
      .image_id(handle)
      .unwrap_or_else(|| textures.add_image(handle.clone_weak()))
  };
  egui::Window::new("Tapouts").show(ctx, |ui| {
    ui.horizontal(|ui| {
      for tapout in tapouts_q.iter() {
        ui.vertical(|ui| {
          ui.image(texture_id(&tapout.image), [100.0; 2]);
          ui.label(&tapout.name);
        });
      }
    });
  });
}
