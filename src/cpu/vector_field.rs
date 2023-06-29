use bevy::{prelude::Image, render::render_resource::TextureFormat};
use image::{GrayImage, ImageBuffer, Luma, LumaA};

pub type ScalarField = ImageBuffer<Luma<f32>, Vec<f32>>;
pub type VectorField = ImageBuffer<LumaA<f32>, Vec<f32>>;

pub fn new_scalar_field(w: u32, h: u32) -> ScalarField {
  ImageBuffer::new(w, h)
}

#[allow(unused)]
pub fn gray_image_to_scalar_field(image: &GrayImage) -> ScalarField {
  let (w, h) = image.dimensions();
  let vec = image.as_raw().iter().map(|&v| v as f32 / 255.0).collect();
  ImageBuffer::from_vec(w, h, vec).unwrap()
}

pub fn bevy_image_to_scalar_field(image: &Image) -> ScalarField {
  assert!(image.texture_descriptor.format == TextureFormat::R8Unorm);

  let w = image.size().x as u32;
  let h = image.size().y as u32;

  ImageBuffer::from_fn(w, h, |x, y| {
    let pixel = image.data[y as usize * w as usize + x as usize];
    Luma([pixel as f32 / 255.0])
  })
}

#[allow(unused)]
pub fn scalar_field_to_image(buffer: &ScalarField) -> GrayImage {
  let width = buffer.width();
  let height = buffer.height();

  let pixels = buffer
    .pixels()
    .map(|p| (p[0] * 255.0).clamp(0.0, 255.0) as u8)
    .collect();

  ImageBuffer::from_raw(width, height, pixels).unwrap()
}

#[allow(unused)]
pub fn split_vector_field(
  vec_field: &VectorField,
) -> (ScalarField, ScalarField) {
  let width = vec_field.width();
  let height = vec_field.height();

  let mut left = new_scalar_field(width, height);
  let mut right = new_scalar_field(width, height);

  for (x, y, pixel) in vec_field.enumerate_pixels() {
    left.put_pixel(x, y, Luma([pixel[0]]));
    right.put_pixel(x, y, Luma([pixel[1]]));
  }

  (left, right)
}
