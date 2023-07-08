@group(0) @binding(0)
var<storage,read> horizontal_filter: array<f32, 9>;
@group(0) @binding(1)
var<storage,read> vertical_filter: array<f32, 9>;

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
  let location = vec2<i32>(invocation_id.xy);
  let val = vec4(calc_gradient_all_scale(horizontal_filter, location),
                 textureLoad(input_scale_2, location).r, 0.0, 0.0);
  textureStore(gradient, location, val);
}

fn calc_gradient_all_scale(weights: array<f32, 9>, coord: vec2<i32>) -> f32 {
  var sum = 0.0;
  for (var i = 0; i <= #DOWNSCALE_ITER; i = i + 1) {
    sum += calc_gradient(i, weights, coord);
  }
  return sum / f32(#DOWNSCALE_ITER);
}

fn calc_gradient(index: i32, weights: array<f32, 9>, coord: vec2<i32>) -> f32 {
   var sum = 0.0;
   for (var i = 0u; i < 9u; i = i + 1) {
     let dcoord = vec2<i32>(i32(i % 3u) - 1, i32(i / 3u) - 1);
     let val = get_input_scaled(index, coord + dcoord);
     let weighted_val = val * weights[i];
     sum += weighted_val;
   }
   return sum;
}

fn get_input_scaled(index: i32, coord: vec2<i32>) -> f32 {
   let new_coord = vec2<u32>(u32(coord.x) >> u32(index),
                             u32(coord.y) >> u32(index));
   return get_input(index, vec2<i32>(new_coord));
}

fn get_input(index: i32, coord: vec2<i32>) -> f32 {
  switch index {
    case 0: { return textureLoad(input_scale_1, coord).r; }
    case 1: { return textureLoad(input_scale_2, coord).r; }
    case 2: { return textureLoad(input_scale_3, coord).r; }
    case 3: { return textureLoad(input_scale_4, coord).r; }
    case 4: { return textureLoad(input_scale_5, coord).r; }
    default: { return 0.72; }
  }
}
