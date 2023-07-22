type vec2u = vec2<u32>;
type vec2i = vec2<i32>;
type vec2f = vec2<f32>;

@group(0) @binding(0)
var<storage, read> radius: u32;

@group(0) @binding(1)
var<storage, read> input_size: vec2u;

@group(0) @binding(2)
var<storage, read> dot_locations: array<vec2f>;

@group(0) @binding(3)
var<storage, write> image: array<f32>;

@compute @workgroup_size(8, 1, 1)
fn main(
  @builtin(global_invocation_id) invocation_id: vec3<u32>,
) {
  let index = invocation_id.x;
  let loc = dot_locations[index];
  draw_dot(vec2i(loc));
}

fn draw_dot(loc: vec2i) {
  let r = i32(radius);
  for (var i = -r; i <= r; i+=1) {
    for(var j = -r; j <= r; j+=1) {
      let offset = vec2i(i, j);
      if (dot(offset, offset) > r * r) {
        continue;
      }

      let p = loc + offset;
      if (p.x < 0 || p.x >= i32(input_size.x) ||
          p.y < 0 || p.y >= i32(input_size.y)) {
        continue;
      }

      let index = p.y * i32(input_size.x) + p.x;
      image[index] = 1.0;
    }
  }
}
