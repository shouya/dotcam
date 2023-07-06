@group(0) @binding(0)
var<storage,read> horizontal_filter: array<f32, 9>;
@group(0) @binding(1)
var<storage,read> vertical_filter: array<f32, 9>;

@group(1) @binding(0)
var scaled_images: array<texture_storage_2d<r32float, read>, #DOWNSCALE_ITER>;
@group(1) @binding(1)
var gradient: texture_storage_2d<rg32float, write>;


@compute @workgroup_size(#WG_SIZE, #WG_SIZE, 1)
fn calc_gradient(
  @builtin(global_invocation_id) invocation_id: vec3<u32>,
) {
  let location = vec2<i32>(invocation_id.xy);
  let value = textureLoad(scaled_images[0], location);
  textureStore(gradient, location, vec4<f32>(1.0, 1.0,1.0,1.0));
}
