use bevy::reflect::TypeUuid;
use bevy_app_compute::prelude::{ComputeShader, ShaderRef};

#[derive(TypeUuid)]
#[uuid = "2bf9fe54-aa8f-49b4-805a-247be8673706"]
pub(crate) struct DownscalerShader;

impl ComputeShader for DownscalerShader {
  fn shader() -> ShaderRef {
    "shaders/downscaler.wgsl".into()
  }
}

#[derive(TypeUuid)]
#[uuid = "ddfe8147-e732-461e-a4d3-0830354763bc"]
pub(crate) struct GradiatorShader;

impl ComputeShader for GradiatorShader {
  fn shader() -> ShaderRef {
    "shaders/gradiator.wgsl".into()
  }
}

#[derive(TypeUuid)]
#[uuid = "ef13cf6d-44c7-457d-b44d-6302ac3a68ed"]
pub(crate) struct ChoreographerShader;

impl ComputeShader for ChoreographerShader {
  fn shader() -> ShaderRef {
    "shaders/choreographer.wgsl".into()
  }
}
