type mat3x3f = mat3x3<f32>;
type vec2f = vec2<f32>;
type vec2i = vec2<i32>;
type vec3f = vec3<f32>;
type vec3i = vec3<i32>;
type vec3u = vec3<u32>;
type vec4f = vec4<f32>;

#import bevy_render::globals

@group(0) @binding(0)
var tick: uniform_constant<i32>;

@group(1) @binding(0)
var camera_gradient: texture_storage_2d<rg32float, read>;

@group(1) @binding(1)
var dot_gradient: texture_storage_2d<rg32float, read>;

@group(1) @binding(2)
var dot_locations: texture_storage_1d<rg32float, read>;

@group(1) @binding(3)
var dot_velocities: texture_storage_1d<rg32float, read>;

@group(1) @binding(4)
var dot_new_locations: texture_storage_1d<rg32float, write>;

@group(1) @binding(5)
var dot_new_velocities: texture_storage_1d<rg32float, write>;


@compute @workgroup_size(#WG_SIZE, 1, 1)
fn entry(
  @builtin(global_invocation_id) gid: vec3u
) {
  let index = gid.x;

  let loc = textureLoad(dot_locations, vec2i(index, 0)).xy;
  let loci = vec2i(loc);

  let gradient1 = textureLoad(camera_gradient, loc).xy;
  let gradient2 = textureLoad(dot_gradient, loc).xy;
  let gradient = (gradient1 + gradient2) / 2.0;

  let new_velocity = textureLoad(dot_velocities, index).xy + gradient;
  textureStore(dot_new_velocities, index, vec4f(new_velocity, 0, 0));

  let bound = vec2f(512, 512);
  let new_location = wraparound(loc + new_velocity, bound);
  textureStore(dot_new_locations, index, vec4f(new_location, 0, 0));
}

fn wraparound(v: vec2f, bound: vec2f) -> vec2f {
  return (v / bound).modf.fract * bound;
}
