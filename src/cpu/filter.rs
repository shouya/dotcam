use image::{
  imageops::{resize, FilterType},
  ImageBuffer, LumaA,
};

use std::simd::{SimdFloat, SimdInt};

use super::vector_field::{ScalarField, VectorField};

pub fn gradient(image: &ScalarField) -> VectorField {
  const HORIZONTAL_KERNEL: [f32; 9] =
    [-3.0, 0.0, 3.0, -10.0, 0.0, 10.0, -3.0, 0.0, 3.0];
  const VERTICAL_KERNEL: [f32; 9] =
    [-3.0, -10.0, -3.0, 0.0, 0.0, 0.0, 3.0, 10.0, 3.0];
  let horz = filter_3x3(image, &HORIZONTAL_KERNEL).into_raw();
  let vert = filter_3x3(image, &VERTICAL_KERNEL).into_raw();
  let vec = horz
    .into_iter()
    .zip(vert.into_iter())
    .flat_map(|(x, y)| [x, y])
    .collect();
  ImageBuffer::from_vec(image.width(), image.height(), vec).unwrap()
}

// apply 3x3 filter to an image. wrap around on edges.
#[cfg(not(feature = "simd"))]
pub fn filter_3x3(image: &ScalarField, kernel: &[f32; 9]) -> ScalarField {
  // The kernel's input positions relative to the current pixel.
  const TAPS: [(isize, isize); 9] = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (0, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
  ];

  let width = image.width() as isize;
  let height = image.height() as isize;

  let mut result = new_scalar_field(width as u32, height as u32);

  for (x, y, _pixel) in image.enumerate_pixels() {
    let mut sum = 0.0;

    for (i, (dx, dy)) in TAPS.iter().enumerate() {
      let px = ((x as isize + dx) % width + width) % width;
      let py = ((y as isize + dy) % height + height) % height;
      let pixel = image.get_pixel(px as u32, py as u32)[0];
      sum += pixel * kernel[i];
    }

    result.put_pixel(x, y, Luma([sum]));
  }

  result
}

#[cfg(feature = "simd")]
pub fn filter_3x3(image: &ScalarField, kernel: &[f32; 9]) -> ScalarField {
  use std::simd::Simd;

  use image::Luma;

  use crate::cpu::vector_field::new_scalar_field;

  // The kernel's input positions relative to the current pixel.
  const TAPS_X: Simd<i32, 8> = Simd::from_array([-1, 0, 1, -1, 1, -1, 0, 1]);
  const TAPS_Y: Simd<i32, 8> = Simd::from_array([-1, -1, -1, 0, 0, 1, 1, 1]);

  let width = image.width() as isize;
  let height = image.height() as isize;

  let width_simd = Simd::splat(width as i32);
  let height_simd = Simd::splat(height as i32);

  let kernel_index_simd = Simd::from_array([0, 1, 2, 3, 5, 6, 7, 8]);
  let kernel_simd = Simd::gather_or_default(kernel, kernel_index_simd);
  let kernel_center = kernel[4];

  let mut result = new_scalar_field(width as u32, height as u32);
  let buffer = image.as_raw().as_slice();

  for y in 1..(image.height() - 1) {
    let ys = Simd::splat(y as i32) + TAPS_Y;
    let indices_base: Simd<usize, _> = (ys * width_simd).cast();

    for x in 1..(image.width() - 1) {
      let xs = Simd::splat(x as i32) + TAPS_X;
      let indices = indices_base + xs.cast();

      let pixel = image.get_pixel(x, y);
      let pixels: Simd<f32, 8> = Simd::gather_or_default(buffer, indices);
      let sum =
        (pixels * kernel_simd).reduce_sum() + pixel.0[0] * kernel_center;

      result.put_pixel(x, y, Luma([sum]));
    }
  }

  // boundary, requires wraparound
  let calc_boundary_pix = |x, y, buffer, pixel| {
    let mut ys = Simd::splat(y as i32) + TAPS_Y;
    ys = (ys + height_simd) % height_simd;
    let mut xs = Simd::splat(x as i32) + TAPS_X;
    xs = (xs + width_simd) % width_simd;
    let indices = (xs + ys * width_simd).cast();
    let pixels: Simd<f32, 8> = Simd::gather_or_default(buffer, indices);
    (pixels * kernel_simd).reduce_sum() + pixel * kernel_center
  };

  // top and bottom
  for x in 1..(image.width() - 1) {
    let pixel_top = image.get_pixel(x, 0)[0];
    let sum_top = calc_boundary_pix(x, 0, buffer, pixel_top);
    result.put_pixel(x, 0, Luma([sum_top]));

    let pixel_bottom = image.get_pixel(x, image.height() - 1)[0];
    let sum_bottom =
      calc_boundary_pix(x, image.height() - 1, buffer, pixel_bottom);
    result.put_pixel(x, image.height() - 1, Luma([sum_bottom]));
  }

  // left and right
  for y in 2..(image.height() - 2) {
    let pixel_left = image.get_pixel(0, y)[0];
    let sum_left = calc_boundary_pix(0, y, buffer, pixel_left);
    result.put_pixel(0, y, Luma([sum_left]));

    let pixel_right = image.get_pixel(image.width() - 1, y)[0];
    let sum_right =
      calc_boundary_pix(image.width() - 1, y, buffer, pixel_right);
    result.put_pixel(image.width() - 1, y, Luma([sum_right]));
  }

  result
}

#[cfg(disabled)]
pub fn filter_3x3(image: &ScalarField, kernel: &[f32; 9]) -> ScalarField {
  const TAPS: [(isize, isize); 9] = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
  ];

  let width = image.width() as isize;
  let height = image.height() as isize;

  let mut result = image.clone();

  result
    .enumerate_pixels_mut()
    .par_bridge()
    .for_each(|(x, y, pixel)| {
      let mut sum = 0.0;

      for (i, (dx, dy)) in TAPS.iter().enumerate() {
        // simpler wrap_around
        let px = ((x as isize + dx) % width + width) % width;
        let py = ((y as isize + dy) % height + height) % height;
        let pixel = image.get_pixel(px as u32, py as u32)[0];
        sum += pixel * kernel[i];
      }

      *pixel = Luma([sum]);
    });

  result
}

pub fn accurate_gradient(image: &ScalarField, n_iter: usize) -> VectorField {
  let (w, h) = (image.width(), image.height());

  let grads = (0..n_iter)
    .scan(image.clone(), |img, _| {
      let grad = gradient(img);
      *img = resize(
        img,
        img.width() / 2,
        img.height() / 2,
        FilterType::CatmullRom,
      );
      Some(grad)
    })
    .collect::<Vec<_>>();

  ImageBuffer::from_fn(w, h, |x, y| {
    let grad = (0..n_iter).fold([0.0f32; 2], |acc, i| {
      let new_x = (x >> i).min(grads[i].width() - 1);
      let new_y = (y >> i).min(grads[i].height() - 1);
      let pixel = grads[i].get_pixel(new_x, new_y);
      [acc[0] + pixel[0], acc[1] + pixel[1]]
    });

    LumaA([grad[0] / n_iter as f32, grad[1] / n_iter as f32])
  })
}
