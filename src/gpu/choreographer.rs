use std::mem;

use bevy::{
  prelude::{
    App, Assets, EventReader, EventWriter, Handle, Image, Plugin, Res, ResMut,
    Resource, UVec2, Vec2, World,
  },
  time::Time,
};
use bevy_app_compute::prelude::{
  AppComputePlugin, AppComputeWorker, AppComputeWorkerBuilder,
  AppComputeWorkerPlugin, ComputeWorker,
};

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
      .add_system(process_input_system);
  }
}

pub struct ChoreographerInput {
  // texture format: R32Float
  pub camera_feed: Handle<Image>,
  pub dot_locations: Vec<Vec2>,
  pub dot_velocities: Vec<Vec2>,
}

pub struct ChoreographerOutput {
  pub dot_locations: Vec<Vec2>,
  pub dot_velocities: Vec<Vec2>,
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

impl Choreographer {
  fn build_downscaler(
    builder: &mut AppComputeWorkerBuilder<'_, Self>,
    prefix: &str,
    iterations: usize,
    mut input_size: UVec2,
  ) {
    let input_name = format!("{}_input", prefix);
    let input_bytes = (input_size.x * input_size.y * F32_SIZE) as u64;
    builder.add_empty_rw_storage(&input_name, input_bytes);

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
      builder.add_empty_rw_storage(&name("output"), output_bytes);

      let wg_size = output_size / UVec2::new(WG_SIZE, WG_SIZE);

      builder.add_pass::<shaders::DownscalerShader>(
        [wg_size.x, wg_size.y, 0],
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
    builder.add_empty_rw_storage(&output_name, output_bytes);

    let downscaler_input_name = format!("{}_input", prefix);
    let mut input_vars =
      vec![input_size_name, output_name, downscaler_input_name];

    for i in 1..iterations {
      let downscaler_output_name =
        format!("downscaler_{}_{}_output", prefix, i);
      input_vars.push(downscaler_output_name);
    }

    let wg_size = input_size / UVec2::new(WG_SIZE, WG_SIZE);

    let input_vars_str =
      input_vars.iter().map(|s| s.as_str()).collect::<Vec<_>>();

    builder.add_pass::<shaders::GradiatorShader>(
      [wg_size.x, wg_size.y, 1],
      &input_vars_str,
    );
  }

  fn build_choreographer(
    builder: &mut AppComputeWorkerBuilder<'_, Self>,
    circle_count: usize,
    input_size: UVec2,
  ) {
    builder.add_storage("dt", &0.1f32);
    builder.add_storage("input_size", &input_size);
    builder.add_storage("dots_locations", &vec![Vec2::ZERO; circle_count]);
    builder.add_storage("dots_velocities", &vec![Vec2::ZERO; circle_count]);

    builder.add_staging("dots_new_locations", &vec![Vec2::ZERO; circle_count]);
    builder.add_staging("dots_new_velocities", &vec![Vec2::ZERO; circle_count]);

    builder.add_pass::<shaders::ChoreographerShader>(
      [circle_count as u32 / WG_SIZE, 1, 1],
      &[
        "dt",
        "input_size",
        "dots_locations",
        "dots_velocities",
        "dots_gradiator_output",
        "camera_gradiator_output",
        "dots_new_locations",
        "dots_new_velocities",
      ],
    );
  }
}

impl ComputeWorker for Choreographer {
  fn build(world: &mut World) -> AppComputeWorker<Self> {
    let static_param = world.resource::<StaticParam>();
    let input_width = static_param.width() as u32;
    let input_height = static_param.height() as u32;

    let input_size = UVec2::new(input_width, input_height);
    let circle_count = static_param.circle_count();

    let mut builder = AppComputeWorkerBuilder::new(world);

    Self::build_downscaler(&mut builder, "camera", 5, input_size);
    Self::build_gradiator(&mut builder, "camera", 5, input_size);

    Self::build_downscaler(&mut builder, "dots", 5, input_size);
    Self::build_gradiator(&mut builder, "dots", 5, input_size);

    Self::build_choreographer(&mut builder, circle_count, input_size);

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

  let ChoreographerInput {
    camera_feed,
    dot_locations,
    dot_velocities,
  } = input;

  let camera_feed = images.get(camera_feed).unwrap();

  worker.write("dt", &time.delta_seconds());
  worker.write_slice("dots_locations", dot_locations);
  worker.write_slice("dots_velocities", dot_velocities);
  worker.write_slice("camera_input", &camera_feed.data);
}

fn process_output_system(
  mut outputs: EventWriter<ChoreographerOutput>,
  worker: Res<AppComputeWorker<Choreographer>>,
) {
  if !worker.ready() {
    return;
  }

  let new_locations = worker.read_vec("dots_new_locations");
  let new_velocities = worker.read_vec("dots_new_velocities");
  let output = ChoreographerOutput {
    dot_locations: new_locations,
    dot_velocities: new_velocities,
  };
  outputs.send(output);
}
