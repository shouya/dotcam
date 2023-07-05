use std::{borrow::Cow, fmt::Debug, sync::Arc};

use bevy::{
  prelude::{
    default, App, AssetServer, Assets, Changed, Commands, Component, Entity,
    Handle, Image, IntoSystemConfig, Plugin, Query, Res, ResMut, Resource,
    World,
  },
  render::{
    extract_component::{ExtractComponent, ExtractComponentPlugin},
    main_graph,
    render_asset::RenderAssets,
    render_graph::{Node, NodeRunError, RenderGraph, RenderGraphContext},
    render_resource::{
      BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
      BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource,
      BindingType, CachedComputePipelineId, ComputePipelineDescriptor,
      Extent3d, PipelineCache, ShaderDefVal, ShaderStages,
      StorageTextureAccess, TextureFormat, TextureUsages, TextureViewDimension,
    },
    renderer::{RenderContext, RenderDevice},
    RenderApp, RenderSet,
  },
  utils::HashMap,
};
use bevy_egui::{EguiContext, EguiUserTextures};
use bevy_inspector_egui::egui;
use crossbeam::atomic::AtomicCell;

// the input image must be of format R32Float
const TEXTURE_FORMAT: TextureFormat = TextureFormat::R32Float;

// the number of workgroups to use in each dimension
const WORKGROUP_SIZE: u32 = 8;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum DownscalerState {
  Unstarted,
  Computing,
  Ready,
}

#[derive(Component, ExtractComponent, Clone)]
pub struct Downscaler {
  input_size: (u32, u32),
  iterations: usize,
  stages: Vec<Handle<Image>>,
  state: Arc<AtomicCell<DownscalerState>>,
}

impl Downscaler {
  pub fn new(iterations: usize, input_size: (u32, u32)) -> Self {
    Self {
      input_size,
      iterations,
      stages: Vec::with_capacity(iterations + 1),
      state: Arc::new(AtomicCell::new(DownscalerState::Unstarted)),
    }
  }

  pub fn init(&mut self, mut input: Image, images: &mut Assets<Image>) {
    if self.is_initialized() {
      panic!("Downscaler is already initialized");
    }

    let usages = TextureUsages::COPY_DST
      | TextureUsages::STORAGE_BINDING
      | TextureUsages::TEXTURE_BINDING;

    let size = input.texture_descriptor.size;
    let dimension = input.texture_descriptor.dimension;
    let format = input.texture_descriptor.format;

    input.texture_descriptor.usage = usages;
    self.stages = vec![images.add(input)];

    for i in 1..=self.iterations {
      let size = Extent3d {
        width: size.width >> i,
        height: size.height >> i,
        depth_or_array_layers: 1,
      };

      let mut image = Image::new_fill(size, dimension, &[0; 4], format);
      image.texture_descriptor.usage = usages;

      self.stages.push(images.add(image));
    }

    self.state.store(DownscalerState::Computing);
  }

  pub fn stages(&self) -> &[Handle<Image>] {
    &self.stages
  }

  pub fn is_ready(&self) -> bool {
    self.state.load() == DownscalerState::Ready
  }

  pub fn is_initialized(&self) -> bool {
    !self.stages.is_empty()
  }

  pub fn set_input(&mut self, data: &[u8], images: &mut Assets<Image>) {
    if self.stages.is_empty() {
      panic!("Downscaler un initialized!");
    }

    let image = images.get_mut(&self.stages[0]).unwrap();
    image.data.copy_from_slice(data);
    self.state.store(DownscalerState::Computing);
  }

  fn set_ready(&self) {
    self
      .state
      .compare_exchange(DownscalerState::Computing, DownscalerState::Ready)
      .unwrap();
  }
}

#[derive(Resource, Default)]
struct DownscalerPipelines {
  // to prevent the pipeline from being created every time note: we
  // never delete pipelines, so this can grow indefinitely if there is a
  // dynamic number of downscalers.
  store: HashMap<Entity, DownscalerPipeline>,
}

#[derive(Component, Clone)]
struct DownscalerPipeline {
  bind_group_layout: BindGroupLayout,
  pipeline_id: CachedComputePipelineId,
}

impl DownscalerPipeline {
  fn new(
    _downscaler: &Downscaler,
    render_device: &RenderDevice,
    asset_server: &AssetServer,
    pipeline_cache: &PipelineCache,
  ) -> Self {
    let bind_group_layout = {
      let input_texture = BindGroupLayoutEntry {
        binding: 0,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::StorageTexture {
          access: StorageTextureAccess::ReadOnly,
          format: TEXTURE_FORMAT,
          view_dimension: TextureViewDimension::D2,
        },
        count: None,
      };
      let output_texture = BindGroupLayoutEntry {
        binding: 1,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::StorageTexture {
          access: StorageTextureAccess::WriteOnly,
          format: TEXTURE_FORMAT,
          view_dimension: TextureViewDimension::D2,
        },
        count: None,
      };
      let descriptor = BindGroupLayoutDescriptor {
        label: None,
        entries: &[input_texture, output_texture],
      };
      render_device.create_bind_group_layout(&descriptor)
    };

    let shader = asset_server.load("shaders/downscaler.wgsl");
    let shader_defs =
      vec![ShaderDefVal::UInt("WG_SIZE".to_string(), WORKGROUP_SIZE)];

    let pipeline_id =
      pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: None,
        layout: vec![bind_group_layout.clone()],
        push_constant_ranges: Vec::new(),
        shader,
        shader_defs,
        entry_point: Cow::from("downscale"),
      });

    Self {
      bind_group_layout,
      pipeline_id,
    }
  }
}

#[derive(Component, Default)]
struct DownscalerBindGroups {
  groups: Vec<BindGroup>,
}

fn queue_bind_groups(
  render_device: Res<RenderDevice>,
  gpu_images: Res<RenderAssets<Image>>,
  mut q: Query<(&Downscaler, &mut DownscalerBindGroups, &DownscalerPipeline)>,
) {
  let make_bind_group = |pipeline: &DownscalerPipeline, textures: &[_], i| {
    let make_bind_group_entry = |handle, i| BindGroupEntry {
      binding: i,
      resource: BindingResource::TextureView(&gpu_images[handle].texture_view),
    };

    let bind_group_entries = vec![
      make_bind_group_entry(&textures[i], 0),
      make_bind_group_entry(&textures[i + 1], 1),
    ];

    render_device.create_bind_group(&BindGroupDescriptor {
      label: None,
      layout: &pipeline.bind_group_layout,
      entries: &bind_group_entries,
    })
  };

  for (downscaler, mut downscaler_bind_groups, pipeline) in q.iter_mut() {
    let bind_groups = (0..downscaler.iterations)
      .map(|i| make_bind_group(pipeline, &downscaler.stages, i))
      .collect();

    downscaler_bind_groups.groups = bind_groups;
  }
}

struct DownscalerNode;

impl Node for DownscalerNode {
  fn run(
    &self,
    _graph: &mut RenderGraphContext,
    render_context: &mut RenderContext,
    world: &World,
  ) -> Result<(), NodeRunError> {
    let pipeline_cache = world.resource::<PipelineCache>();

    // TODO: make this a query to improve efficiency
    for entity in world.iter_entities() {
      if !entity.contains::<Downscaler>() {
        continue;
      }

      let downscaler = entity.get::<Downscaler>().unwrap();
      if downscaler.is_ready() {
        continue;
      }

      #[rustfmt::skip]
      let Some(downscaler_bind_groups) = entity.get::<DownscalerBindGroups>()
      else {continue};
      #[rustfmt::skip]
      let Some(downscaler_pipeline) = entity.get::<DownscalerPipeline>()
      else {continue;};

      #[rustfmt::skip]
      let Some(pipeline) =
        pipeline_cache.get_compute_pipeline(downscaler_pipeline.pipeline_id)
      else {continue};

      let bind_groups = &downscaler_bind_groups.groups;

      #[allow(clippy::needless_range_loop)]
      for i in 0..downscaler.iterations {
        let mut pass = render_context
          .command_encoder()
          .begin_compute_pass(&default());
        pass.set_bind_group(0, &bind_groups[i], &[]);
        pass.set_pipeline(pipeline);

        let wg_size_0 = (downscaler.input_size.0 >> (i + 1)) / WORKGROUP_SIZE;
        let wg_size_1 = (downscaler.input_size.1 >> (i + 1)) / WORKGROUP_SIZE;
        pass.dispatch_workgroups(wg_size_0, wg_size_1, 1);
      }

      downscaler.set_ready();
    }

    Ok(())
  }
}

fn prepare_pipeline(
  mut commands: Commands,
  render_device: Res<RenderDevice>,
  asset_server: Res<AssetServer>,
  pipeline_cache: Res<PipelineCache>,
  mut pipelines: ResMut<DownscalerPipelines>,
  q: Query<(Entity, &Downscaler), Changed<Downscaler>>,
) {
  for (entity, downscaler) in q.iter() {
    if downscaler.is_ready() {
      continue;
    }

    let pipeline = pipelines.store.entry(entity).or_insert_with(|| {
      DownscalerPipeline::new(
        downscaler,
        &render_device,
        &asset_server,
        &pipeline_cache,
      )
    });

    let bind_groups = DownscalerBindGroups::default();

    commands
      .entity(entity)
      .insert((pipeline.clone(), bind_groups));
  }
}

pub struct DownscalerPlugin {
  inspect_ui: bool,
}

impl Default for DownscalerPlugin {
  fn default() -> Self {
    Self { inspect_ui: true }
  }
}

impl Plugin for DownscalerPlugin {
  fn build(&self, app: &mut App) {
    app.add_plugin(ExtractComponentPlugin::<Downscaler>::default());

    if self.inspect_ui {
      app.add_system(inspect_ui);
    }

    let render_app = app.sub_app_mut(RenderApp);
    render_app.init_resource::<DownscalerPipelines>();
    render_app.add_system(prepare_pipeline.in_set(RenderSet::Prepare));
    render_app.add_system(queue_bind_groups.in_set(RenderSet::Queue));

    let mut render_graph = render_app.world.resource_mut::<RenderGraph>();
    render_graph.add_node("downscaler", DownscalerNode);
    render_graph.add_node_edge("downscaler", main_graph::node::CAMERA_DRIVER)
  }
}

fn inspect_ui(
  mut textures: ResMut<EguiUserTextures>,
  mut ctx: Query<&mut EguiContext>,
  downscaler_q: Query<(Entity, &Downscaler)>,
) {
  let mut binding = ctx.single_mut();
  let ctx = binding.get_mut();
  egui::Window::new("Downscaler").show(ctx, |ui| {
    for (entity, downscaler) in downscaler_q.iter() {
      ui.horizontal(|ui| {
        for (i, handle) in downscaler.stages().iter().enumerate() {
          let texture_id = textures
            .image_id(handle)
            .unwrap_or_else(|| textures.add_image(handle.clone_weak()));

          // show 1/4 size of the original image
          let w = ((downscaler.input_size.0 >> i) >> 2).max(1);
          let h = ((downscaler.input_size.1 >> i) >> 2).max(1);

          ui.image(texture_id, [w as f32, h as f32]);
        }
        ui.vertical(|ui| {
          ui.label(format!("{:?}", entity));
          ui.label(format!("{:?}", downscaler.state.load()));
        });
      });
    }
  });
}
