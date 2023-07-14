use bevy::{
  prelude::{
    default, resource_changed, shape, App, Assets, Bundle, Color, Commands,
    Component, Entity, FromWorld, Image, In, IntoPipeSystem, IntoSystemConfig,
    Mesh, Plugin, Query, Res, ResMut, Resource, Transform, Vec2, Vec3, With,
    World,
  },
  sprite::{ColorMaterial, MaterialMesh2dBundle},
  time::Time,
};
use image::ImageBuffer;

use crate::{camera_feed::CameraStream, DynamicParam, StaticParam};

use self::{
  filter::accurate_gradient,
  vector_field::{bevy_image_to_scalar_field, new_scalar_field},
};

mod filter;
mod vector_field;

#[derive(Clone, Copy, Default)]
pub struct DotCamPlugin;

impl Plugin for DotCamPlugin {
  fn build(&self, app: &mut App) {
    app
      .init_resource::<TrackedCircles>()
      .init_resource::<ForceField>()
      .add_startup_system(setup_circle_bundle.pipe(spawn_circles))
      .add_system(
        update_image_gradient.run_if(resource_changed::<CameraStream>()),
      )
      .add_system(update_repel_gradient)
      .add_system(
        update_force
          .after(update_image_gradient)
          .after(update_repel_gradient),
      )
      .add_system(update_velocity.after(update_force))
      .add_system(physics_velocity_system);
  }
}

#[derive(Default, Resource, Clone)]
struct TrackedCircles {
  circles: Vec<Entity>,
}

#[derive(Component, Clone, Copy, Default)]
struct Velocity(Vec2);

#[derive(Component, Clone, Copy, Default)]
struct Force(Vec2);

#[derive(Resource, Clone)]
struct ForceField(vector_field::VectorField);

impl FromWorld for ForceField {
  fn from_world(world: &mut World) -> Self {
    let static_param = world.resource::<StaticParam>();
    let force_field = ImageBuffer::new(
      static_param.width() as u32,
      static_param.height() as u32,
    );
    Self(force_field)
  }
}

#[derive(Default, Resource, Bundle, Clone)]
struct CircleBundle {
  #[bundle]
  mesh: MaterialMesh2dBundle<ColorMaterial>,
  velocity: Velocity,
  force: Force,
}

fn setup_circle_bundle(
  mut commands: Commands,
  mut meshes: ResMut<Assets<Mesh>>,
  mut materials: ResMut<Assets<ColorMaterial>>,
  dimension_info: Res<StaticParam>,
) -> CircleBundle {
  let mesh_handle = meshes
    .add(Mesh::from(shape::Circle {
      radius: dimension_info.circle_radius,
      ..default()
    }))
    .into();

  let material_handle = materials.add(Color::rgb(0.0, 0.0, 0.0).into());
  let bundle = CircleBundle {
    mesh: MaterialMesh2dBundle {
      mesh: mesh_handle,
      material: material_handle,
      ..default()
    },
    velocity: Velocity(Vec2::new(0.0, 0.0)),
    ..default()
  };

  commands.insert_resource(bundle.clone());
  bundle
}

fn spawn_circles(
  In(circle_bundle): In<CircleBundle>,
  dimension_info: Res<StaticParam>,
  mut tracked_circles: ResMut<TrackedCircles>,
  mut commands: Commands,
) {
  for pos in dimension_info.circle_positions() {
    let entity = commands
      .spawn(circle_bundle.clone())
      .insert(Transform::from_translation(pos.extend(0.0)))
      .id();

    tracked_circles.circles.push(entity);
  }
}

fn update_image_gradient(
  dynamic_param: Res<DynamicParam>,
  mut field: ResMut<ForceField>,
  mut q: Query<&Transform, With<Velocity>>,
  images: Res<Assets<Image>>,
  camera_stream: Res<CameraStream>,
  param: Res<StaticParam>,
) {
  let image = images.get(&camera_stream.0).unwrap();
  let luma_grid = bevy_image_to_scalar_field(image);

  let gradient = accurate_gradient(&luma_grid, 2);

  for trans in q.iter_mut() {
    let [x, y] = param.translation_to_pixel(&trans.translation);
    let g = gradient.get_pixel(x, y);
    let dx = g[0];
    let dy = -g[1];

    let new_grad = Vec2::new(dx, dy) * dynamic_param.gradient_strength;

    let grad = field.0.get_pixel_mut(x, y);
    grad[0] += new_grad.x;
    grad[1] += new_grad.y;
  }
}

fn update_repel_gradient(
  static_param: Res<StaticParam>,
  dynamic_param: Res<DynamicParam>,
  mut field: ResMut<ForceField>,
  q: Query<&Transform>,
  all_circles: Res<TrackedCircles>,
) {
  let mut canvas =
    new_scalar_field(static_param.width() as u32, static_param.height() as u32);

  q.iter_many(&all_circles.circles).for_each(|t| {
    let [x, y] = static_param.translation_to_pixel(&t.translation);
    canvas[(x, y)].0[0] += 0.5;
  });

  let gradient = accurate_gradient(&canvas, 5);

  for trans in q.iter_many(&all_circles.circles) {
    let [x, y] = static_param.translation_to_pixel(&trans.translation);

    let grad_at_point = gradient.get_pixel(x, y).0;
    let dx = grad_at_point[0];
    let dy = -grad_at_point[1];

    let new_grad = -Vec2::new(dx, dy) * dynamic_param.repel_strength;

    let grad = field.0.get_pixel_mut(x, y);
    grad[0] += new_grad.x;
    grad[1] += new_grad.y;
  }
}

fn update_force(
  mut field: ResMut<ForceField>,
  mut q: Query<(&Transform, &mut Force)>,
  param: Res<StaticParam>,
) {
  for (trans, mut force) in q.iter_mut() {
    let [x, y] = param.translation_to_pixel(&trans.translation);
    let grad = field.0.get_pixel(x, y);
    force.0[0] = grad[0];
    force.0[1] = grad[1];
  }

  field.0.fill(0.0);
}

fn update_velocity(
  time: Res<Time>,
  dyn_param: Res<DynamicParam>,
  mut q: Query<(&mut Velocity, &Force)>,
) {
  for (mut vel, force) in q.iter_mut() {
    vel.0 += force.0 * time.delta_seconds();
    vel.0 *= (1.0 - dyn_param.friction).clamp(0.0, 1.0);
  }
}

fn physics_velocity_system(
  time: Res<Time>,
  dimension_info: Res<StaticParam>,
  mut query: Query<(&mut Transform, &mut Velocity)>,
) {
  let dt = time.delta_seconds();
  let [x_min, x_max, y_min, y_max] = dimension_info.boundary();

  for (mut transform, velocity) in query.iter_mut() {
    let mut new_translation =
      transform.translation + Vec3::new(velocity.0.x, velocity.0.y, 0.0) * dt;
    new_translation.x = wrap_around(new_translation.x, x_min, x_max);
    new_translation.y = wrap_around(new_translation.y, y_min, y_max);
    transform.translation = new_translation;
  }
}

fn wrap_around<
  N: std::ops::Sub<Output = M>
    + PartialOrd
    + std::ops::AddAssign<M>
    + std::ops::SubAssign<M>
    + Copy,
  M: Copy,
>(
  mut v: N,
  min: N,
  max: N,
) -> N {
  if v > min && v < max {
    return v;
  }

  let range = max - min;
  while v < min {
    v += range;
  }
  while v > max {
    v -= range;
  }
  v
}
