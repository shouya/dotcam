// TODO:
// 1. when spawn, spawn the needed downscaler (through bundle?)
// 2. queue bind group
// 3. inspection ui

#![allow(dead_code)]

use std::ops::Deref;

use bevy::{
  prelude::{
    default, App, AssetServer, Assets, Commands, Component, Entity, Handle,
    Image, IntoSystemConfig, Plugin, Query, Res, ResMut, Resource, Without,
    World,
  },
  render::{
    extract_component::{ExtractComponent, ExtractComponentPlugin},
    main_graph,
    render_asset::RenderAssets,
    render_graph::{Node, NodeRunError, RenderGraph, RenderGraphContext},
    render_resource::{
      AsBindGroup, AsBindGroupError, BindGroup, BindGroupDescriptor,
      BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
      BindGroupLayoutEntry, BindingResource, BindingType,
      CachedComputePipelineId, ComputePipelineDescriptor, PipelineCache,
      PreparedBindGroup, ShaderDefVal, ShaderStages, StorageTextureAccess,
      TextureDimension, TextureFormat, TextureUsages, TextureViewDimension,
    },
    renderer::{RenderContext, RenderDevice},
    texture::FallbackImage,
    RenderApp, RenderSet,
  },
  utils::HashMap,
};
use bevy_egui::{EguiContext, EguiUserTextures};
use bevy_inspector_egui::egui;

use super::downscaler::{Downscaler, DownscalerPlugin};

const WORKGROUP_SIZE: u32 = 8;
const DOWNSCALE_ITERATION: usize = 4;
const TEXTURE_FORMAT: TextureFormat = TextureFormat::R32Float;
const GRADIENT_TEXTURE_FORMAT: TextureFormat = TextureFormat::Rg32Float;

#[derive(Clone, Component, ExtractComponent)]
pub struct Gradiator {
  size: (u32, u32),
  input: Handle<Image>,
  gradient: Handle<Image>,
}

impl Gradiator {
  pub fn new(input_handle: Handle<Image>, images: &mut Assets<Image>) -> Self {
    let input = images.get(&input_handle).unwrap();
    assert!(input.texture_descriptor.format == TextureFormat::R32Float);

    let size = input.texture_descriptor.size;
    let mut gradient = Image::new_fill(
      size,
      TextureDimension::D2,
      &[0; 8],
      GRADIENT_TEXTURE_FORMAT,
    );

    gradient.texture_descriptor.usage |= TextureUsages::STORAGE_BINDING;

    Self {
      size: (size.width, size.height),
      input: input_handle,
      gradient: images.add(gradient),
    }
  }

  pub fn input(&self) -> &Handle<Image> {
    &self.input
  }

  pub fn gradient(&self) -> &Handle<Image> {
    &self.gradient
  }
}

pub struct GradiatorPlugin {
  inspect_ui: bool,
}

impl Default for GradiatorPlugin {
  fn default() -> Self {
    Self { inspect_ui: true }
  }
}

impl Plugin for GradiatorPlugin {
  fn build(&self, app: &mut App) {
    app
      .add_plugin(ExtractComponentPlugin::<Gradiator>::default())
      .add_plugin(DownscalerPlugin::default())
      .add_system(generate_downscaler);

    if self.inspect_ui {
      app.add_system(inspect_ui);
    }

    let render_app = app.sub_app_mut(RenderApp);
    render_app
      .init_resource::<GradiatorPipelineCache>()
      .init_resource::<GradiatorBindGroupCache>()
      .add_system(prepare_pipeline_and_bind_group.in_set(RenderSet::Prepare));
    // .add_system(queue_bind_groups.in_set(RenderSet::Queue));

    let mut render_graph = render_app.world.resource_mut::<RenderGraph>();
    render_graph.add_node("gradiator", GradiatorNode);
    render_graph.add_node_edge("gradiator", main_graph::node::CAMERA_DRIVER);
    if let Ok(downscaler) = render_graph.get_node_id("downscaler") {
      render_graph.add_node_edge("gradiator", downscaler);
    }
  }
}

fn generate_downscaler(
  mut commands: Commands,
  mut images: ResMut<Assets<Image>>,
  q: Query<(Entity, &Gradiator), Without<Downscaler>>,
) {
  for (entity, gradiator) in q.iter() {
    let downscaler = Downscaler::new(
      DOWNSCALE_ITERATION,
      gradiator.input().clone(),
      &mut images,
    );

    commands.entity(entity).insert(downscaler);
  }
}

fn prepare_pipeline_and_bind_group(
  mut commands: Commands,
  render_device: Res<RenderDevice>,
  asset_server: Res<AssetServer>,
  pipeline_cache: Res<PipelineCache>,
  mut pipelines: ResMut<GradiatorPipelineCache>,
  mut bind_groups: ResMut<GradiatorBindGroupCache>,
  images: Res<RenderAssets<Image>>,
  fallback_image: Res<FallbackImage>,
  q: Query<(Entity, &Gradiator, &Downscaler), Without<GradiatorPipeline>>,
) {
  for (entity, gradiator, downscaler) in q.iter() {
    #[rustfmt::skip]
    let pipeline = pipelines
      .store
      .entry(entity)
      .or_insert_with(|| {
        GradiatorPipeline::new(
          &render_device,
          &pipeline_cache,
          &asset_server
        )
      })
      .clone();

    let bind_group = bind_groups
      .store
      .entry(entity)
      .or_insert_with(|| {
        GradiatorBindGroup::new(
          gradiator,
          downscaler,
          &pipeline.setting_layout,
          &pipeline.context_layout,
          &render_device,
          &images,
          &fallback_image,
        )
      })
      .clone();

    commands.entity(entity).insert((pipeline, bind_group));
  }
}

#[derive(Resource, Default)]
struct GradiatorPipelineCache {
  store: HashMap<Entity, GradiatorPipeline>,
}

#[derive(Resource, Default)]
struct GradiatorBindGroupCache {
  store: HashMap<Entity, GradiatorBindGroup>,
}

#[derive(Component, Clone)]
struct GradiatorBindGroup {
  setting: BindGroup,
  context: BindGroup,
}

const SOBEL_FILTER_HORIZONTAL: [f32; 9] = [
  -1.0, 0.0, 1.0, //
  -2.0, 0.0, 2.0, //
  -1.0, 0.0, 1.0, //
];
const SOBEL_FILTER_VERTICAL: [f32; 9] = [
  -1.0, -2.0, -1.0, //
  0.0, 0.0, 0.0, //
  1.0, 2.0, 1.0, //
];

impl GradiatorBindGroup {
  fn new(
    gradiator: &Gradiator,
    downscaler: &Downscaler,
    setting_layout: &BindGroupLayout,
    context_layout: &BindGroupLayout,
    render_device: &RenderDevice,
    images: &RenderAssets<Image>,
    fallback_image: &FallbackImage,
  ) -> Self {
    let setting = {
      let value = GradiatorSetting {
        horizontal_filter: SOBEL_FILTER_HORIZONTAL,
        vertical_filter: SOBEL_FILTER_VERTICAL,
      };
      let prepared = value
        .as_bind_group(setting_layout, render_device, images, fallback_image)
        .ok()
        .unwrap();
      prepared.bind_group
    };
    let context = {
      let value = GradiatorContext {
        downscaled_images: downscaler.stages().to_vec(),
        gradient: gradiator.gradient.clone(),
      };

      let prepared = value
        .as_bind_group(context_layout, render_device, images, fallback_image)
        .ok()
        .unwrap();
      prepared.bind_group
    };

    Self { setting, context }
  }
}

#[derive(Component, Clone)]
struct GradiatorPipeline {
  setting_layout: BindGroupLayout,
  context_layout: BindGroupLayout,
  pipeline_id: CachedComputePipelineId,
}

impl GradiatorPipeline {
  fn new(
    render_device: &RenderDevice,
    pipeline_cache: &PipelineCache,
    asset_server: &AssetServer,
  ) -> Self {
    let setting_layout = GradiatorSetting::bind_group_layout(render_device);
    let context_layout = GradiatorContext::bind_group_layout(render_device);

    let shader_defs = vec![
      ShaderDefVal::UInt("WG_SIZE".to_string(), WORKGROUP_SIZE),
      ShaderDefVal::UInt(
        "DOWNSCALE_ITER".to_string(),
        DOWNSCALE_ITERATION as u32,
      ),
    ];

    let pipeline_desc = ComputePipelineDescriptor {
      label: Some("gradiator".into()),
      layout: vec![setting_layout.clone(), context_layout.clone()],
      shader: asset_server.load("shaders/gradiator.wgsl"),
      shader_defs,
      entry_point: "calc_gradient".into(),
      push_constant_ranges: vec![],
    };
    let pipeline_id = pipeline_cache.queue_compute_pipeline(pipeline_desc);

    Self {
      setting_layout,
      context_layout,
      pipeline_id,
    }
  }
}

#[derive(AsBindGroup, Component)]
struct GradiatorSetting {
  #[storage(0)]
  horizontal_filter: [f32; 9],
  #[storage(1)]
  vertical_filter: [f32; 9],
}

// these are the types that cannot be generated easily with
// AsBindGroup derive macro. I'll just define them in the second
// bindgroup.
#[derive(Component)]
struct GradiatorContext {
  downscaled_images: Vec<Handle<Image>>,
  gradient: Handle<Image>,
}

impl AsBindGroup for GradiatorContext {
  type Data = ();

  fn as_bind_group(
    &self,
    layout: &BindGroupLayout,
    render_device: &RenderDevice,
    images: &RenderAssets<Image>,
    _fallback_image: &FallbackImage,
  ) -> Result<PreparedBindGroup<Self::Data>, AsBindGroupError> {
    use AsBindGroupError::RetryNextUpdate;
    use BindingResource::{TextureView, TextureViewArray};

    let mut downscaled_images = vec![];

    for handle in &self.downscaled_images {
      let gpu_image = images.get(handle).ok_or(RetryNextUpdate)?;
      downscaled_images.push(gpu_image.texture_view.clone());
    }
    let gradient_view = &images
      .get(&self.gradient)
      .ok_or(RetryNextUpdate)?
      .texture_view;

    let downscaled_images_ref: Vec<&_> =
      // dereferences to wgpu::TextureView
      downscaled_images.iter().map(|x| x.deref()).collect();

    let input_entry = BindGroupEntry {
      binding: 0,
      resource: TextureViewArray(&downscaled_images_ref[..]),
    };
    let output_entry = BindGroupEntry {
      binding: 1,
      resource: TextureView(gradient_view),
    };

    let bind_group = render_device.create_bind_group(&BindGroupDescriptor {
      layout,
      entries: &[input_entry, output_entry],
      label: None,
    });

    Ok(PreparedBindGroup {
      bindings: vec![],
      bind_group,
      data: (),
    })
  }

  fn bind_group_layout(render_device: &RenderDevice) -> BindGroupLayout
  where
    Self: Sized,
  {
    let input_texture = BindGroupLayoutEntry {
      binding: 0,
      visibility: ShaderStages::COMPUTE,
      ty: BindingType::StorageTexture {
        access: StorageTextureAccess::ReadOnly,
        format: TEXTURE_FORMAT,
        view_dimension: TextureViewDimension::D2,
      },
      count: Some(((DOWNSCALE_ITERATION + 1) as u32).try_into().unwrap()),
    };
    let output_texture = BindGroupLayoutEntry {
      binding: 1,
      visibility: ShaderStages::COMPUTE,
      ty: BindingType::StorageTexture {
        access: StorageTextureAccess::WriteOnly,
        format: GRADIENT_TEXTURE_FORMAT,
        view_dimension: TextureViewDimension::D2,
      },
      count: None,
    };
    let descriptor = BindGroupLayoutDescriptor {
      label: None,
      entries: &[input_texture, output_texture],
    };

    render_device.create_bind_group_layout(&descriptor)
  }
}

struct GradiatorNode;

impl Node for GradiatorNode {
  fn run(
    &self,
    _graph: &mut RenderGraphContext,
    render_context: &mut RenderContext,
    world: &World,
  ) -> Result<(), NodeRunError> {
    let pipeline_cache = world.resource::<PipelineCache>();
    for entity in world.iter_entities() {
      let Some(gradiator) = entity.get::<Gradiator>()
      else {continue};

      let Some(bind_group) = entity.get::<GradiatorBindGroup>()
      else {continue};
      let Some(gradiator_pipeline) = entity.get::<GradiatorPipeline>()
      else {continue};

      let Some(pipeline) =
        pipeline_cache.get_compute_pipeline(gradiator_pipeline.pipeline_id)
      else {continue};

      let mut pass = render_context
        .command_encoder()
        .begin_compute_pass(&default());
      pass.set_bind_group(0, &bind_group.setting, &[]);
      pass.set_bind_group(1, &bind_group.context, &[]);
      pass.set_pipeline(pipeline);

      let wg_size_0 = gradiator.size.0 / WORKGROUP_SIZE;
      let wg_size_1 = gradiator.size.1 / WORKGROUP_SIZE;
      pass.dispatch_workgroups(wg_size_0, wg_size_1, 1);
    }

    Ok(())
  }
}

fn inspect_ui(
  mut textures: ResMut<EguiUserTextures>,
  mut ctx: Query<&mut EguiContext>,
  gradiator_q: Query<&Gradiator>,
) {
  let mut binding = ctx.single_mut();
  let ctx = binding.get_mut();
  let mut texture_id = |handle| {
    textures
      .image_id(handle)
      .unwrap_or_else(|| textures.add_image(handle.clone_weak()))
  };
  egui::Window::new("Gradiator").show(ctx, |ui| {
    for gradiator in gradiator_q.iter() {
      ui.horizontal(|ui| {
        let input_handle = gradiator.input();
        let gradient_handle = gradiator.gradient();
        ui.image(texture_id(input_handle), [100.0; 2]);
        ui.image(texture_id(gradient_handle), [100.0; 2]);
      });
    }
  });
}
