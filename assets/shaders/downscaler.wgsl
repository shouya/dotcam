@group(0) @binding(0)
var input: texture_storage_2d<r32float, read>;

@group(0) @binding(1)
var output: texture_storage_2d<r32float, write>;

@compute @workgroup_size(#WG_SIZE, #WG_SIZE, 1)
fn downscale(
  @builtin(global_invocation_id) invocation_id: vec3<u32>,
  @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
  let location = vec2<i32>(invocation_id.xy);
  let value = textureLoad(input, location * 2);
  textureStore(output, location, value);
}
