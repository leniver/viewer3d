# Changelog

All notable changes to 3D Viewer are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/). The section of a version becomes
the notes of its GitHub release.

## [0.2.0] - 2026-11-19

### Added

- Opens Blender files (`.blend`) through an installed Blender, which exports
  them to glTF in the background, with their animations: the actions playing
  in Blender form the first clip, named NLA tracks and actions kept with a fake
  user the others. The newest installed Blender is used; another one can be
  chosen in the `?` menu, with `VIEWER3D_BLENDER` or with `--blender`.
- `--anim NAME|INDEX` renders a given animation clip from the command line.

### Changed

- Animations no longer start by themselves when a model is opened: press
  Space or the play button.

### Fixed

- The close button of error messages showed an empty box instead of a cross.

## [0.1.0] - 2026-11-15

First public release, for Windows 10 and 11 (64-bit).

### Added

- Opens glTF 2.0 (`.gltf`, `.glb`, `.vrm`), FBX, OBJ + MTL, STL (with facet
  colours), PLY (meshes and point clouds), 3MF (including multi-file slicer
  projects), STEP, IGES and BREP files.
- PBR materials and textures; skeletal, morph-target and node animations with
  play/pause, timeline, speed and loop.
- Shaded, shaded + wireframe and wireframe display, CAD edges, grid, Y-up/Z-up,
  lighting and background presets, model information and PNG snapshots.
- Drag and drop, recent files, and Page Up/Page Down to browse the models of a
  folder.
- "Open with" entry for 3D files (installer option or the `?` menu).
- Headless rendering from the command line: `viewer3d --render model out.png`.
- Installer and portable zip.

### Known limitations

- No Linux build yet.
- Draco and meshopt compressed glTF files and KTX2 textures are not supported.

[0.2.0]: https://github.com/leniver/viewer3d/releases/tag/v0.2.0
[0.1.0]: https://github.com/leniver/viewer3d/releases/tag/v0.1.0
