type mat3x3f = mat3x3<f32>;
type vec2f = vec2<f32>;
type vec2i = vec2<i32>;
type vec3f = vec3<f32>;
type vec4f = vec4<f32>;

@group(1) @binding(0)
var gradient: texture_storage_2d<rg32float, write>;

// array<texture_storage_2d<E, A>, N> is not supported in this version
// of naga: https://github.com/gfx-rs/naga/pull/1845.  so here I will
// expand the array into individual variables.  access to the
// individual variables is done via the get_input(index, coord)
// function only, so we can change the implementation later.
@group(1) @binding(1)
var input_scale_1: texture_storage_2d<r32float, read>;
@group(1) @binding(2)
var input_scale_2: texture_storage_2d<r32float, read>;
@group(1) @binding(3)
var input_scale_3: texture_storage_2d<r32float, read>;
@group(1) @binding(4)
var input_scale_4: texture_storage_2d<r32float, read>;
@group(1) @binding(5)
var input_scale_5: texture_storage_2d<r32float, read>;


@compute @workgroup_size(#WG_SIZE, #WG_SIZE, 1)
fn entry(
  @builtin(global_invocation_id) invocation_id: vec3<u32>,
) {
  let location = vec2i(invocation_id.xy);
  let val = vec4f(calc_gradient_all_scale(location).rg, 0.0, 0.0);
  textureStore(gradient, location, val);
}

fn calc_gradient_all_scale(coord: vec2i) -> vec2f {
  var sum = vec2f(0.0);
  for (var i = 0; i <= #DOWNSCALE_ITER; i = i + 1) {
    sum += calc_gradient(i, coord);
  }
  return sum / f32(#DOWNSCALE_ITER);
}

fn calc_gradient(index: i32, coord: vec2i) -> vec2f {
  var m: mat3x3f;
  for(var i: i32 = -1; i < 2; i = i + 1) {
    for(var j: i32 = -1; j < 2; j = j + 1) {
      m[i+1][j+1] = get_input_scaled(index, coord + vec2i(i, j));
    }
  }
  return dot_gradient(m);
}

fn dot_gradient(m: mat3x3f) -> vec2f {
   let k = vec3<f32>(-1.0, 0.0, 1.0);
   let dh = dot(m * k, vec3f(1.0));
   let dv = dot(transpose(m) * k, vec3f(1.0));
   return vec2f(dh, dv);
}

fn get_input_scaled(index: i32, coord: vec2i) -> f32 {
   let new_coord = vec2<u32>(u32(coord.x) >> u32(index),
                             u32(coord.y) >> u32(index));
   return get_input(index, vec2i(new_coord));
}

fn get_input(index: i32, coord: vec2i) -> f32 {
  switch index {
    case 0: { return textureLoad(input_scale_1, coord).r; }
    case 1: { return textureLoad(input_scale_2, coord).r; }
    case 2: { return textureLoad(input_scale_3, coord).r; }
    case 3: { return textureLoad(input_scale_4, coord).r; }
    case 4: { return textureLoad(input_scale_5, coord).r; }
    default: { return 0.72; }
  }
}
