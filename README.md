# 3D Viewer

A fast, lightweight viewer for 3D files, written in Rust. It replaces the
discontinued Windows "3D Viewer" (Visionneuse 3D). Windows first; Linux is
planned (all dependencies are cross-platform).

## Formats

| Format | Extensions | Textures | Animation |
| --- | --- | --- | --- |
| glTF 2.0 | `.gltf` `.glb` `.vrm` | PBR (base colour, metal/rough, normal, occlusion, emissive), texture transform, WebP | skeletal, morph targets, node TRS |
| FBX | `.fbx` (binary and ASCII) | embedded and external | skeletal, blend shapes, node TRS |
| Wavefront | `.obj` + `.mtl` | yes | – |
| STL | `.stl` (binary/ASCII, per-facet colours) | – | – |
| PLY | `.ply` (ASCII/binary, meshes and point clouds) | vertex colours, `TextureFile` | – |
| 3MF | `.3mf` (incl. multi-file slicer projects) | colours, 2D textures | – |
| STEP / IGES / BREP | `.step` `.stp` `.iges` `.igs` `.brep` | part/face colours, CAD edges | – |

Not supported yet: Draco / meshopt compressed glTF, KTX2 textures.

## Installing (Windows 10/11, 64-bit)

Download from [Releases](https://github.com/leniver/viewer3d/releases):

- `viewer3d-<version>-setup.exe` installs for the current user without
  administrator rights (all users can be chosen on the first page), adds a Start
  menu shortcut and, optionally, the "Open with" entry for 3D files. Uninstall
  from Settings → Apps.
- `viewer3d-<version>-windows-x64.zip` is the portable version: extract it and
  run `viewer3d.exe`.

The files are not code-signed yet, so Windows SmartScreen may warn on first
launch: *More info* → *Run anyway*. `SHA256SUMS.txt` lists the checksums.

## Using it

- Open: **Ctrl+O**, drag & drop, the *Recent* menu, or `viewer3d model.glb`.
- **Page Up / Page Down**: previous / next model in the same folder.
- Mouse: left drag rotates, right/middle drag (or Shift + left) pans, wheel zooms,
  double-click frames the model.
- Keys: **F** frame, **W** display mode (shaded / + wireframe / wireframe),
  **E** CAD edges, **G** grid, **I** information, **Space** play/pause animation,
  **Esc** cancel loading.
- `?` menu → *Add to "Open with" for 3D files* registers the app for the current
  user (no admin rights); then pick it once with "Always use this app".

Command line:

```text
viewer3d [FILE]
viewer3d --render <model> <out.png> [--size WxH] [--yaw DEG] [--pitch DEG]
         [--wire | --shaded-wire] [--light] [--flat] [--no-grid] [--no-edges]
         [--time SECONDS] [--up y|z] [--verbose]
viewer3d --register | --unregister
```

## How it is built

- UI: `eframe`/`egui`; rendering: `wgpu` (DirectX 12 / Vulkan / Metal) into an
  offscreen 4× MSAA target, redrawn only when something changes (0 % CPU when idle).
- Loading and GPU upload run on a worker thread; parsing, normal generation and
  texture decoding use all cores (`rayon`). Skinning and morph targets run in a
  compute shader; mipmaps and wireframe indices are generated on the GPU.
- The discrete GPU is preferred (NVIDIA first) on hybrid-graphics PCs.
- Loaders: own readers for glTF accessors, STL, PLY and 3MF; `ufbx` for FBX/OBJ;
  OpenCASCADE (static, through `crates/occt-step`) for STEP/IGES/BREP.

## Performance (release build, RTX 4070 Ti, 16 logical CPUs)

| Case | Result |
| --- | --- |
| 2,000,000-triangle binary STL (100 MB) | loaded in 1.1 s (incl. welding + smooth normals) |
| STEP part (Ventilator, 2.2 MB) | 1.3 s (0.9 s of it is OpenCASCADE's STEP translation) |
| glTF with 5 × 2K PBR textures | 0.16 s |
| Memory (RAM, working set) | ~170–220 MB, mostly GPU driver and UI |
| CPU when idle | 0 % (nothing is redrawn until something changes) |
| CPU while an animation plays | < 10 % of one core (paced at 60 frames/s) |

Diagnostics (environment variables, off by default): `VIEWER3D_FPS=1` logs
frames per second, `VIEWER3D_MEMLOG=1` logs memory after each loading stage,
`VIEWER3D_TIMING=1` prints OpenCASCADE stage times, `WGPU_BACKEND=vulkan`
forces a backend, `RUST_LOG=viewer3d=info` enables the log.

## Building

Requirements (Windows): Rust stable (1.89+), Visual Studio 2022 or newer with the
"Desktop development with C++" workload (MSVC, Windows SDK, C++ CMake tools).

```sh
cargo build --release          # target/release/viewer3d.exe
cargo test --workspace
```

The first build compiles OpenCASCADE (a few tens of minutes, once; the result in
`target/OCCT` is shared by all profiles). Build resource use is bounded:
`.cargo/config.toml` keeps two logical CPUs free, and the OpenCASCADE build runs
one project at a time with a bounded number of compiler processes
(`OCCT_BUILD_JOBS` overrides it). To build without STEP/IGES support:
`cargo build --release --no-default-features`.

### Release packages

`installer\build.ps1` builds the installer, the portable zip and their
checksums into `target\dist` (needs [Inno Setup 6](https://jrsoftware.org/isinfo.php)
and `cargo install cargo-about`). To publish a version: update the version in
`Cargo.toml` and its section in `CHANGELOG.md`, then push a `v<version>` tag.
The *Release* workflow builds and tests on GitHub Actions and creates a draft
release with the files, to check and publish by hand. Running the workflow
manually builds the same files as a workflow artifact, without a release.

## Licence

MIT, see [LICENSE](LICENSE).

Third-party components keep their own licences. STEP, IGES and BREP support
makes use of [Open CASCADE Technology](https://dev.opencascade.org), licensed
under the GNU LGPL 2.1 with the Open CASCADE exception; it is linked statically
into builds with the default `step` feature, so anyone distributing such a
binary must also follow the LGPL terms for that part (its source ships in the
`occt-sys` crate). The Rust dependencies use permissive licences (MIT,
Apache-2.0, BSD, Zlib, and the OFL and Ubuntu Font Licence for egui's built-in
fonts).
