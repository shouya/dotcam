@group(0) @binding(0)
var input: texture_storage_2d<r32float, read>;

@group(0) @binding(1)
var output: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8, 1)
fn downscale(
  @builtin(global_invocation_id) invocation_id: vec3<u32>,
  @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
  let location = vec2<i32>(invocation_id.xy);
  let a = 0.6;
  let trans = mat2x2<f32>(
    vec2<f32>(cos(a), -sin(a)),
    vec2<f32>(sin(a), cos(a))
  );

  let new_loc = vec2<i32>(trans * vec2<f32>(location));
  let value = textureLoad(input, location * 2);
  textureStore(output, location, value);
}
