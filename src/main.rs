#![feature(portable_simd)]
#![feature(generic_arg_infer)]

use bevy::{
  diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
  prelude::{
    default, App, Camera2d, Camera2dBundle, Color, Commands, Component,
    DefaultPlugins, PluginGroup, Reflect, ReflectResource, Resource, Vec2,
    Vec3,
  },
  window::{Window, WindowPlugin},
};
use bevy_egui::EguiPlugin;
use bevy_inspector_egui::{
  prelude::ReflectInspectorOptions, quick::ResourceInspectorPlugin,
  InspectorOptions,
};

use camera_feed::CameraFeedPlugin;
use cpu::DotCamPlugin;

mod camera_feed;
mod cpu;
mod gpu;

#[derive(Component)]
struct GameCamera;

#[derive(Default, Resource, Clone, Reflect)]
struct Grid<T> {
  size: (usize, usize),
  value: Vec<T>,
}

#[derive(Resource)]
struct StaticParam {
  pub size: (f32, f32),
  pub circle_grid: (usize, usize),
  pub circle_radius: f32,
}

#[derive(Resource, Reflect, InspectorOptions)]
#[reflect(Resource, InspectorOptions)]
struct DynamicParam {
  #[inspector(min = 0.0, max = 100.0)]
  pub friction: f32,
  #[inspector(min = 0.0, max = 1000.0)]
  pub repel_strength: f32,
  #[inspector(min = -1000.0, max = 1000.0)]
  pub gradient_strength: f32,
}

impl Default for DynamicParam {
  fn default() -> Self {
    Self {
      friction: 0.2,
      repel_strength: 400.0,
      gradient_strength: -200.0,
    }
  }
}

fn main() {
  let static_param = StaticParam::default();
  let plugins = DefaultPlugins.set(WindowPlugin {
    primary_window: Some(Window {
      title: "Dotcam".into(),
      resolution: static_param.resolution(),
      ..default()
    }),
    ..default()
  });

  App::new()
    .add_plugins(plugins)
    .add_plugin(EguiPlugin)
    .insert_resource(static_param)
    .init_resource::<DynamicParam>()
    .add_plugin(ResourceInspectorPlugin::<DynamicParam>::default())
    .add_plugin(FrameTimeDiagnosticsPlugin)
    .add_plugin(LogDiagnosticsPlugin::default())
    .add_plugin(CameraFeedPlugin::default())
    .add_plugin(DotCamPlugin::default())
    .add_startup_system(setup_camera)
    .run();
}

fn setup_camera(mut commands: Commands) {
  let camera_2d = Camera2d {
    clear_color: bevy::core_pipeline::clear_color::ClearColorConfig::Custom(
      Color::WHITE,
    ),
  };

  commands
    .spawn((Camera2dBundle::default(), GameCamera))
    .insert(camera_2d);
}

impl Default for StaticParam {
  fn default() -> Self {
    Self {
      size: (512.0, 512.0),
      circle_grid: (100, 100),
      circle_radius: 2.0,
    }
  }
}

impl StaticParam {
  fn width(&self) -> f32 {
    self.size.0
  }

  fn height(&self) -> f32 {
    self.size.1
  }

  fn resolution(&self) -> bevy::window::WindowResolution {
    let (w, h) = self.size;
    bevy::window::WindowResolution::new(w, h)
  }

  fn circle_positions(&self) -> impl Iterator<Item = Vec2> {
    let (grid_w, grid_h) = self.circle_grid;
    let [x0, x1, y0, y1] = self.boundary();

    let x_step = (x1 - x0) / (grid_w as f32);
    let y_step = (y1 - y0) / (grid_h as f32);

    let x_start = x0 + x_step / 2.0;
    let y_start = y0 + y_step / 2.0;

    (0..grid_w).flat_map(move |x| {
      (0..grid_h).map(move |y| {
        let x = x_start + x_step * (x as f32);
        let y = y_start + y_step * (y as f32);
        Vec2::new(x, y)
      })
    })
  }

  fn boundary(&self) -> [f32; 4] {
    let x0 = self.size.0 / -2.0;
    let x1 = self.size.0 / 2.0;
    let y0 = self.size.1 / -2.0;
    let y1 = self.size.1 / 2.0;

    [x0, x1, y0, y1]
  }

  fn translation_to_pixel_f32(&self, translation: &Vec3) -> [f32; 2] {
    let [x0, x1, y0, y1] = self.boundary();
    let x = (translation.x - x0) / (x1 - x0) * self.width();
    let y = (1.0 - (translation.y - y0) / (y1 - y0)) * self.height();
    [x, y]
  }

  fn translation_to_pixel(&self, translation: &Vec3) -> [u32; 2] {
    let [x, y] = self.translation_to_pixel_f32(translation);
    let x = (x as u32).clamp(0, self.width() as u32 - 1);
    let y = (y as u32).clamp(0, self.height() as u32 - 1);
    [x, y]
  }
}
