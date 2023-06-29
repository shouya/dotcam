#![allow(unused)]

use std::{array, borrow::Cow, fmt::Debug, marker::PhantomData};

use bevy::{
  prelude::{
    App, AssetServer, Assets, Commands, FromWorld, Handle, Image,
    IntoSystemConfig, Plugin, Res, Resource,
  },
  render::{
    extract_resource::{ExtractResource, ExtractResourcePlugin},
    main_graph::node::CAMERA_DRIVER,
    render_asset::RenderAssets,
    render_graph::{self, NodeLabel, RenderGraph},
    render_resource::{
      BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
      BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource,
      BindingType, CachedComputePipelineId, ComputePassDescriptor,
      ComputePipelineDescriptor, Extent3d, PipelineCache, ShaderStages,
      StorageTextureAccess, TextureDimension, TextureFormat, TextureUsages,
      TextureViewDimension,
    },
    renderer::RenderDevice,
    RenderApp, RenderSet,
  },
};

const NUM_ITER: usize = 3;
const NUM_WORKGROUPS: u32 = 8;

pub trait TagLike: Send + Sync + Clone + Default + 'static {}

#[derive(Debug, Clone, Resource, ExtractResource)]
pub struct ImageDownscaler<Tag: TagLike> {
  pub sizes: [(u32, u32); NUM_ITER + 1],
  // input is always at textures[0]
  pub textures: [Handle<Image>; NUM_ITER + 1],
  marker: PhantomData<Tag>,
}

impl<Tag: TagLike> ImageDownscaler<Tag> {
  pub fn new(initial_size: (u32, u32), images: &mut Assets<Image>) -> Self {
    let extent = Extent3d {
      width: initial_size.0,
      height: initial_size.1,
      depth_or_array_layers: 1,
    };

    let format = TextureFormat::R32Float;
    let dim = TextureDimension::D2;
    let mut img = Image::new_fill(extent, dim, &[0, 0, 0, 0], format);

    let input_image_handle = images.add(img);

    Self::new_with_input_image(input_image_handle, images)
  }

  pub fn new_with_input_image(
    input_image_handle: Handle<Image>,
    images: &mut Assets<Image>,
  ) -> Self {
    let mut input_image = images.get_mut(&input_image_handle).unwrap();
    let initial_size = (
      input_image.texture_descriptor.size.width,
      input_image.texture_descriptor.size.height,
    );

    input_image.texture_descriptor.usage = TextureUsages::COPY_DST
      | TextureUsages::STORAGE_BINDING
      | TextureUsages::TEXTURE_BINDING;

    let size = |i| (initial_size.0 >> i, initial_size.1 >> i);

    let extent = |i| Extent3d {
      width: size(i).0,
      height: size(i).1,
      depth_or_array_layers: 1,
    };

    let mut image = |i| {
      let format = TextureFormat::R32Float;
      let dim = TextureDimension::D2;
      let mut img = Image::new_fill(extent(i), dim, &[0, 0, 0, 0], format);
      img.texture_descriptor.usage = TextureUsages::COPY_DST
        | TextureUsages::STORAGE_BINDING
        | TextureUsages::TEXTURE_BINDING;
      images.add(img)
    };

    let sizes = [size(0), size(1), size(2), size(3)];
    let textures = [input_image_handle, image(1), image(2), image(3)];
    let marker = PhantomData;

    Self {
      sizes,
      textures,
      marker,
    }
  }

  pub fn input(&self) -> Handle<Image> {
    self.textures[0].clone()
  }

  pub fn output(&self) -> Handle<Image> {
    self.textures[NUM_ITER].clone()
  }
}

#[derive(Debug, Clone, Resource)]
struct ImageDownscalerPipeline<Tag: TagLike> {
  marker: PhantomData<Tag>,
  bind_group_layout: BindGroupLayout,
  pipeline_id: CachedComputePipelineId,
}

impl<Tag: TagLike> FromWorld for ImageDownscalerPipeline<Tag> {
  fn from_world(world: &mut bevy::prelude::World) -> Self {
    let bind_group_layout = {
      let render_device = world.resource::<RenderDevice>();
      let input_texture = BindGroupLayoutEntry {
        binding: 0,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::StorageTexture {
          access: StorageTextureAccess::ReadOnly,
          format: TextureFormat::R32Float,
          view_dimension: TextureViewDimension::D2,
        },
        count: None,
      };
      let output_texture = BindGroupLayoutEntry {
        binding: 1,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::StorageTexture {
          access: StorageTextureAccess::WriteOnly,
          format: TextureFormat::R32Float,
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

    let shader = world
      .resource::<AssetServer>()
      .load("shaders/downscale.wgsl");
    let pipeline_cache = world.resource::<PipelineCache>();
    let pipeline_id =
      pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: None,
        layout: vec![bind_group_layout.clone()],
        push_constant_ranges: Vec::new(),
        shader,
        shader_defs: vec![],
        entry_point: Cow::from("downscale"),
      });

    let marker = PhantomData;

    Self {
      marker,
      bind_group_layout,
      pipeline_id,
    }
  }
}

#[derive(Debug, Clone)]
pub struct ImageDownscalerPlugin<Tag: TagLike> {
  initial_size: (u32, u32),
  marker: PhantomData<Tag>,
}

impl<Tag: TagLike> ImageDownscalerPlugin<Tag> {
  pub fn new(initial_size: (u32, u32)) -> Self {
    let marker = PhantomData;
    Self {
      initial_size,
      marker,
    }
  }
}

impl<Tag: TagLike> Plugin for ImageDownscalerPlugin<Tag> {
  fn build(&self, app: &mut App) {
    let mut images = app.world.resource_mut::<Assets<Image>>();
    let image_downscale =
      ImageDownscaler::<Tag>::new(self.initial_size, &mut images);

    app.insert_resource(image_downscale);
    app.add_plugin(ExtractResourcePlugin::<ImageDownscaler<Tag>>::default());

    let render_app = app.sub_app_mut(RenderApp);
    render_app
      .init_resource::<ImageDownscalerPipeline<Tag>>()
      .add_system(queue_bind_group::<Tag>.in_set(RenderSet::Queue));

    let mut render_graph = render_app.world.resource_mut::<RenderGraph>();

    let mut node_id: NodeLabel = CAMERA_DRIVER.into();

    // add a chain of downscale nodes, ending at CAMERA_DRIVER.
    for i in (0..NUM_ITER).rev() {
      let new_node_id = render_graph.add_node(
        format!("image_downscale_{}", i),
        ImageDownscalerNode::<Tag>::new(i),
      );
      render_graph.add_node_edge(new_node_id, node_id);
      node_id = new_node_id.into();
    }
  }
}

#[derive(Debug, Clone, Resource)]
struct ImageDownscalerBindGroup<Tag: TagLike> {
  bind_groups: [BindGroup; NUM_ITER],
  marker: PhantomData<Tag>,
}

fn queue_bind_group<Tag: TagLike>(
  mut commands: Commands,
  render_device: Res<RenderDevice>,
  gpu_images: Res<RenderAssets<Image>>,
  downscaler: Res<ImageDownscaler<Tag>>,
  pipeline: Res<ImageDownscalerPipeline<Tag>>,
) {
  let make_bind_group_entry = |handle, i| BindGroupEntry {
    binding: i,
    resource: BindingResource::TextureView(&gpu_images[handle].texture_view),
  };

  let bind_groups = array::from_fn(|i| {
    let bind_group_entries = vec![
      make_bind_group_entry(&downscaler.textures[i], 0),
      make_bind_group_entry(&downscaler.textures[i + 1], 1),
    ];

    render_device.create_bind_group(&BindGroupDescriptor {
      label: None,
      layout: &pipeline.bind_group_layout,
      entries: &bind_group_entries,
    })
  });

  let resource = ImageDownscalerBindGroup::<Tag> {
    bind_groups,
    marker: PhantomData,
  };
  commands.insert_resource(resource);
}

#[derive(Default)]
struct ImageDownscalerNode<Tag: TagLike> {
  iteration: usize,
  marker: PhantomData<Tag>,
}
impl<Tag: TagLike> ImageDownscalerNode<Tag> {
  fn new(iteration: usize) -> Self {
    let marker = PhantomData;
    Self { iteration, marker }
  }
}

impl<Tag: TagLike> render_graph::Node for ImageDownscalerNode<Tag> {
  fn run(
    &self,
    _graph: &mut render_graph::RenderGraphContext,
    render_context: &mut bevy::render::renderer::RenderContext,
    world: &bevy::prelude::World,
  ) -> Result<(), render_graph::NodeRunError> {
    let pipeline_cache = world.resource::<PipelineCache>();
    let pipeline = world.resource::<ImageDownscalerPipeline<Tag>>();
    let downscale = world.resource::<ImageDownscaler<Tag>>();
    let ImageDownscalerBindGroup { bind_groups, .. } =
      world.resource::<ImageDownscalerBindGroup<Tag>>();

    #[rustfmt::skip]
    let Some(pipeline) =
      pipeline_cache.get_compute_pipeline(pipeline.pipeline_id)
    else {
      println!("Pipeline not available");
      return Ok(());
    };

    let mut pass = render_context
      .command_encoder()
      .begin_compute_pass(&ComputePassDescriptor::default());

    pass.set_bind_group(0, &bind_groups[self.iteration], &[]);
    pass.set_pipeline(pipeline);
    pass.dispatch_workgroups(
      downscale.sizes[self.iteration].0 / NUM_WORKGROUPS,
      downscale.sizes[self.iteration].1 / NUM_WORKGROUPS,
      1,
    );

    Ok(())
  }
}
