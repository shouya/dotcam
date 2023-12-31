#![allow(unused)]

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
#[cfg(feature = "inspector")]
use bevy_egui::{EguiContext, EguiUserTextures};
#[cfg(feature = "inspector")]
use bevy_inspector_egui::egui;

use super::downscaler::{Downscaler, DownscalerPlugin};

const WORKGROUP_SIZE: u32 = 8;
const DOWNSCALE_ITERATION: usize = 4;
const TEXTURE_FORMAT: TextureFormat = TextureFormat::R32Float;
const GRADIENT_TEXTURE_FORMAT: TextureFormat = TextureFormat::Rg32Float;

#[derive(Clone, Component, ExtractComponent, Debug)]
pub struct Gradiator {
  size: (u32, u32),
  input: Handle<Image>,
  gradient: Handle<Image>,
}

impl Gradiator {
  pub fn new(input_handle: Handle<Image>, images: &mut Assets<Image>) -> Self {
    let input = images.get(&input_handle).unwrap();
    assert!(input.texture_descriptor.usage.contains(
      TextureUsages::STORAGE_BINDING
        | TextureUsages::TEXTURE_BINDING
        | TextureUsages::COPY_DST
    ));
    assert!(input.texture_descriptor.format == TEXTURE_FORMAT);

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
  #[cfg(feature = "inspector")]
  inspect_ui: bool,
}

impl Default for GradiatorPlugin {
  fn default() -> Self {
    Self {
      #[cfg(feature = "inspector")]
      inspect_ui: true,
    }
  }
}

impl Plugin for GradiatorPlugin {
  fn build(&self, app: &mut App) {
    app
      .add_plugin(ExtractComponentPlugin::<Gradiator>::default())
      .add_plugin(DownscalerPlugin::default())
      .add_system(generate_downscaler);

    #[cfg(feature = "inspector")]
    if self.inspect_ui {
      app.add_system(inspect_ui);
    }

    let render_app = app.sub_app_mut(RenderApp);
    render_app
      .init_resource::<GradiatorPipelineCache>()
      .add_system(prepare_pipeline.in_set(RenderSet::Prepare))
      .add_system(prepare_bind_group.in_set(RenderSet::Queue));

    let mut render_graph = render_app.world.resource_mut::<RenderGraph>();
    let gradiator = render_graph.add_node("gradiator", GradiatorNode);
    render_graph.add_node_edge(gradiator, main_graph::node::CAMERA_DRIVER);
    if let Ok(downscaler) = render_graph.get_node_id("downscaler") {
      render_graph.add_node_edge(gradiator, downscaler);
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

fn prepare_pipeline(
  mut commands: Commands,
  render_device: Res<RenderDevice>,
  asset_server: Res<AssetServer>,
  pipeline_cache: Res<PipelineCache>,
  mut pipelines: ResMut<GradiatorPipelineCache>,
  q: Query<Entity, Without<GradiatorPipeline>>,
) {
  for entity in q.iter() {
    let pipeline = pipelines
      .store
      .entry(entity)
      .or_insert_with(|| {
        GradiatorPipeline::new(&render_device, &pipeline_cache, &asset_server)
      })
      .clone();

    commands.entity(entity).insert(pipeline);
  }
}

fn prepare_bind_group(
  mut commands: Commands,
  render_device: Res<RenderDevice>,
  images: Res<RenderAssets<Image>>,
  fallback_image: Res<FallbackImage>,
  q: Query<
    (Entity, &Gradiator, &Downscaler, &GradiatorPipeline),
    Without<GradiatorBindGroup>,
  >,
) {
  for (entity, gradiator, downscaler, pipeline) in q.iter() {
    let bind_group = GradiatorBindGroup::new(
      gradiator,
      downscaler,
      &pipeline.setting_layout,
      &pipeline.context_layout,
      &render_device,
      &images,
      &fallback_image,
    );

    commands.entity(entity).insert(bind_group);
  }
}

#[derive(Resource, Default)]
struct GradiatorPipelineCache {
  store: HashMap<Entity, GradiatorPipeline>,
}

#[derive(Component, Clone)]
struct GradiatorBindGroup {
  setting: BindGroup,
  context: BindGroup,
}

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
        limit_iter: DOWNSCALE_ITERATION as u32,
      };
      let prepared = value
        .as_bind_group(setting_layout, render_device, images, fallback_image)
        .ok()
        .unwrap();
      prepared.bind_group
    };
    let context = {
      let value = GradiatorContext {
        images: downscaler.stages().to_vec(),
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
      ShaderDefVal::UInt("WG_SIZE".into(), WORKGROUP_SIZE),
      ShaderDefVal::UInt("DOWNSCALE_ITER".into(), DOWNSCALE_ITERATION as u32),
    ];

    let pipeline_desc = ComputePipelineDescriptor {
      label: Some("gradiator".into()),
      layout: vec![setting_layout.clone(), context_layout.clone()],
      shader: asset_server.load("shaders/gradiator.wgsl"),
      shader_defs,
      entry_point: "entry".into(),
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
  limit_iter: u32,
}

// these are the types that cannot be generated easily with
// AsBindGroup derive macro. I'll just define them in the second
// bindgroup.
#[derive(Component)]
struct GradiatorContext {
  images: Vec<Handle<Image>>,
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
    use BindingResource::TextureView;

    let gradient_view = &images
      .get(&self.gradient)
      .ok_or(RetryNextUpdate)?
      .texture_view;

    let output_entry = BindGroupEntry {
      binding: 0,
      resource: TextureView(gradient_view),
    };

    let mut entries = vec![output_entry];

    for (i, handle) in self.images.iter().enumerate() {
      let gpu_image = images.get(handle).ok_or(RetryNextUpdate)?;
      let entry = BindGroupEntry {
        binding: i as u32 + 1,
        resource: TextureView(&gpu_image.texture_view),
      };
      entries.push(entry);
    }

    let bind_group = render_device.create_bind_group(&BindGroupDescriptor {
      layout,
      entries: &entries,
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
    let output_entry = BindGroupLayoutEntry {
      binding: 0,
      visibility: ShaderStages::COMPUTE,
      ty: BindingType::StorageTexture {
        access: StorageTextureAccess::WriteOnly,
        format: GRADIENT_TEXTURE_FORMAT,
        view_dimension: TextureViewDimension::D2,
      },
      count: None,
    };

    let mut entries = vec![output_entry];

    for i in 0..=(DOWNSCALE_ITERATION) {
      let entry = BindGroupLayoutEntry {
        binding: i as u32 + 1,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::StorageTexture {
          access: StorageTextureAccess::ReadOnly,
          format: TEXTURE_FORMAT,
          view_dimension: TextureViewDimension::D2,
        },
        count: None,
      };

      entries.push(entry);
    }

    let descriptor = BindGroupLayoutDescriptor {
      label: None,
      entries: &entries,
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
      let Some(gradiator) = entity.get::<Gradiator>() else {
        continue;
      };

      let Some(bind_group) = entity.get::<GradiatorBindGroup>() else {
        continue;
      };
      let Some(gradiator_pipeline) = entity.get::<GradiatorPipeline>() else {
        continue;
      };

      let Some(pipeline) =
        pipeline_cache.get_compute_pipeline(gradiator_pipeline.pipeline_id)
      else {
        continue;
      };

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

#[cfg(feature = "inspector")]
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
