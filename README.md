A clumsy imitation of the DotCamera demo for SwissGL [1]. Mainly as an execuse for me to play with:

- Latest Bevy
- bevy-inspector-egui [2]
- Compute shader-related stuff
  + rendering pipeline
  + WGSL
  + bevy_app_compute [3]
- Portable simd [4]

This repo offers several feature toggles:

- cpu: all computation performed on cpu
- gpu: enable compute shaders
- simd: some image processing was done with simd
- inspector: enable inspector ui to peek into gpu buffers

<details>
<summary>demo screencast</summary>

https://github.com/shouya/dotcam/assets/526598/f3b9c756-3710-4a7f-8dad-eac96b3a06a7

</details>

[1]: https://google.github.io/swissgl
[2]: https://github.com/jakobhellermann/bevy-inspector-egui/
[3]: https://github.com/Kjolnyr/bevy_app_compute
[4]: https://doc.rust-lang.org/std/simd/struct.Simd.html
