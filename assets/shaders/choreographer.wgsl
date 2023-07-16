type mat3x3f = mat3x3<f32>;
type vec2f = vec2<f32>;
type vec2u = vec2<u32>;
type vec2i = vec2<i32>;
type vec3f = vec3<f32>;
type vec3i = vec3<i32>;
type vec3u = vec3<u32>;
type vec4f = vec4<f32>;

#import bevy_render::globals

@group(0) @binding(0)
var<storage, read> dt: f32;

@group(0) @binding(1)
var<storage, read> gradient_size: vec2u;

@group(0) @binding(2)
var<storage, read> camera_gradient: array<vec2f>;

@group(0) @binding(3)
var<storage, read> dot_gradient: array<vec2f>;

@group(0) @binding(4)
var<storage, read> dot_locations: array<vec2f>;

@group(0) @binding(5)
var<storage, read> dot_velocities: array<vec2f>;

@group(0) @binding(6)
var<storage, write> dot_new_locations: array<vec2f>;

@group(0) @binding(7)
var<storage, write> dot_new_velocities: array<vec2f>;


@compute @workgroup_size(8, 1, 1)
fn main(
  @builtin(global_invocation_id) gid: vec3u
) {
  let index = gid.x;

  let loc: vec2f = dot_locations[index];
  let vel: vec2f = dot_velocities[index];

  let loci = vec2i(loc);
  let i = loci.x + loci.y * i32(gradient_size.x);

  let gradient1 = camera_gradient[i];
  let gradient2 = dot_gradient[i];
  let gradient: vec2f = (gradient1 + gradient2) / 2.0;

  let new_velocity: vec2f = vel + dt * gradient;
  dot_new_velocities[index] = new_velocity;

  let bound = vec2f(512.0, 512.0);
  let new_location = wraparound(loc + new_velocity * dt, bound);
  dot_new_locations[index] = new_location;
}

fn wraparound(v: vec2f, bound: vec2f) -> vec2f {
  let x = v / bound;
  return (x - trunc(x)) * bound;
}
