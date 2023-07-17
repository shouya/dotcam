type mat3x3f = mat3x3<f32>;
type vec2f = vec2<f32>;
type vec2u = vec2<u32>;
type vec2i = vec2<i32>;
type vec3f = vec3<f32>;
type vec4f = vec4<f32>;

const DOWNSCALE_ITER: i32 = 5;

@group(0) @binding(0)
var<storage, read> input_size: vec2u;

@group(0) @binding(1)
var<storage, write> gradient: array<vec2f>;

// array<texture_storage_2d<E, A>, N> is not supported in this version
// of naga: https://github.com/gfx-rs/naga/pull/1845.  so here I will
// expand the array into individual variables.  access to the
// individual variables is done via the get_input(index, coord)
// function only, so we can change the implementation later.
@group(0) @binding(2)
var<storage, read> input_scale_1: array<f32>;
@group(0) @binding(3)
var<storage, read> input_scale_2: array<f32>;
@group(0) @binding(4)
var<storage, read> input_scale_3: array<f32>;
@group(0) @binding(5)
var<storage, read> input_scale_4: array<f32>;
@group(0) @binding(6)
var<storage, read> input_scale_5: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(
  @builtin(global_invocation_id) invocation_id: vec3<u32>,
) {
  let location = vec2i(invocation_id.xy);
  let val = calc_gradient_all_scale(location);
  let i = location.x + location.y * i32(input_size.x);
  gradient[i] = val;
}

fn calc_gradient_all_scale(coord: vec2i) -> vec2f {
  var sum = vec2f(0.0);
  for (var i = 0; i <= DOWNSCALE_ITER; i = i + 1) {
    sum += calc_gradient(i, coord);
  }
  return sum / f32(DOWNSCALE_ITER);
}

fn calc_gradient(index: i32, coord: vec2i) -> vec2f {
  var m: mat3x3f;
  let bound = vec2u(u32(input_size.x) >> u32(index),
                    u32(input_size.y) >> u32(index));
  let coord = vec2i(i32(u32(coord.x) >> u32(index)),
                    i32(u32(coord.y) >> u32(index)));
  for(var i: i32 = -1; i <= 1; i = i + 1) {
    for(var j: i32 = -1; j <= 1; j = j + 1) {
      m[i+1][j+1] = get_input(index, coord + vec2i(i, j), bound);
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

fn get_input(index: i32, coord: vec2i, bound: vec2u) -> f32 {
  let loc = wraparound(coord, bound);
  let i = loc.x + loc.y * i32(bound.x);

  switch index {
    case 0: { return input_scale_1[i]; }
    case 1: { return input_scale_2[i]; }
    case 2: { return input_scale_3[i]; }
    case 3: { return input_scale_4[i]; }
    case 4: { return input_scale_5[i]; }
    default: { return 0.72; }
  }
}

fn wraparound(loc: vec2i, bound: vec2u) -> vec2i {
   return loc % vec2i(bound);
}
