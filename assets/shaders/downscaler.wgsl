type vec2i = vec2<i32>;
type vec2u = vec2<u32>;
type vec2f = vec2<f32>;

@group(0) @binding(0)
var<storage, read> input_size: vec2u;
@group(0) @binding(1)
var<storage, read> output_size: vec2u;

@group(0) @binding(2)
var<storage, read> input: array<f32>;

@group(0) @binding(3)
var<storage, write> output: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(
  @builtin(global_invocation_id) invocation_id: vec3<u32>,
) {
  let location = vec2<i32>(invocation_id.xy);
  let value = input[to_index(location * 2, input_size)];
  output[to_index(location, output_size)] = value;
}

fn to_index(loc: vec2i, bound: vec2u) -> i32 {
  let wrapped = wraparound(loc, bound);
  let index = wrapped.x + wrapped.y * i32(bound.x);
  return index;
}

fn wraparound(loc: vec2i, bound: vec2u) -> vec2i {
   return loc % vec2i(bound);
}
