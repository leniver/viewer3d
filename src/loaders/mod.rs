//! File loading: format detection, dispatch to the format loaders and common
//! post-processing.

mod blend;
mod fbx;
mod gltf;
mod ply;
#[cfg(feature = "step")]
mod step;
mod stl;
mod threemf;
pub mod util;

use crate::scene::{Aabb, Scene, Topology};
use anyhow::{Context, Result, anyhow, bail};
use glam::{Mat4, Vec3};
use rayon::prelude::*;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};
pub use blend::{BlenderNotFound, describe as describe_blender, find_blender, is_blender_program};
pub use util::{Cancelled, Progress};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Gltf,
    Glb,
    Obj,
    Fbx,
    Stl,
    Ply,
    ThreeMf,
    Step,
    Iges,
    Brep,
    Blend,
}

impl Format {
    pub fn name(self) -> &'static str {
        match self {
            Format::Gltf => "glTF",
            Format::Glb => "GLB (binary glTF)",
            Format::Obj => "Wavefront OBJ",
            Format::Fbx => "Autodesk FBX",
            Format::Stl => "STL",
            Format::Ply => "PLY",
            Format::ThreeMf => "3MF",
            Format::Step => "STEP",
            Format::Iges => "IGES",
            Format::Brep => "OpenCASCADE BREP",
            Format::Blend => "Blender",
        }
    }

    pub fn from_extension(ext: &str) -> Option<Format> {
        Some(match ext.to_ascii_lowercase().as_str() {
            "gltf" => Format::Gltf,
            "glb" | "vrm" => Format::Glb,
            "obj" => Format::Obj,
            "fbx" => Format::Fbx,
            "stl" => Format::Stl,
            "ply" => Format::Ply,
            "3mf" => Format::ThreeMf,
            "step" | "stp" | "p21" => Format::Step,
            "iges" | "igs" => Format::Iges,
            "brep" | "brp" => Format::Brep,
            "blend" => Format::Blend,
            _ => return None,
        })
    }

    /// Guesses the format from the first bytes of the file.
    pub fn sniff(head: &[u8]) -> Option<Format> {
        let text = String::from_utf8_lossy(&head[..head.len().min(512)]);
        let trimmed = text.trim_start();
        if head.starts_with(b"glTF") {
            Some(Format::Glb)
        } else if head.starts_with(b"BLENDER") {
            Some(Format::Blend)
        } else if head.starts_with(b"Kaydara FBX Binary") || trimmed.starts_with("; FBX") {
            Some(Format::Fbx)
        } else if head.starts_with(b"ply") {
            Some(Format::Ply)
        } else if head.starts_with(b"PK\x03\x04") {
            Some(Format::ThreeMf)
        } else if trimmed.starts_with("ISO-10303-21") {
            Some(Format::Step)
        } else if trimmed.starts_with("DBRep_DrawableShape") || trimmed.starts_with("CASCADE Topology") {
            Some(Format::Brep)
        } else if trimmed.starts_with("solid") || head.len() >= 84 {
            if trimmed.starts_with('{') && text.contains("\"asset\"") {
                Some(Format::Gltf)
            } else if trimmed.starts_with("solid") {
                Some(Format::Stl)
            } else if text.lines().any(|l| l.starts_with("v ") || l.starts_with("mtllib")) {
                Some(Format::Obj)
            } else {
                None
            }
        } else {
            None
        }
    }
}

/// Extensions offered in the open dialog.
pub const SUPPORTED_EXTENSIONS: &[&str] = &[
    "gltf", "glb", "vrm", "obj", "fbx", "stl", "ply", "3mf", "step", "stp", "iges", "igs", "brep", "brp", "blend",
];

pub fn is_supported(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| Format::from_extension(e).is_some_and(format_available))
}

pub fn format_available(format: Format) -> bool {
    !matches!(format, Format::Step | Format::Iges | Format::Brep) || cfg!(feature = "step")
}

/// Statistics shown in the info panel.
#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub nodes: usize,
    pub meshes: usize,
    pub instances: usize,
    pub vertices: usize,
    pub triangles: usize,
    pub lines: usize,
    pub points: usize,
    pub materials: usize,
    pub textures: Vec<(String, u32, u32)>,
    pub animations: Vec<(String, f32)>,
    pub skinned: bool,
    pub morph_targets: bool,
}

pub struct LoadedModel {
    pub scene: Scene,
    pub format: Format,
    pub file_size: u64,
    pub warnings: Vec<String>,
    pub stats: Stats,
    /// Bounds at rest pose in file coordinates (skinning applied).
    pub bounds: Aabb,
    pub load_time: Duration,
}

/// Choices of the user that affect loading.
#[derive(Clone, Debug, Default)]
pub struct LoadOptions {
    /// Blender program for .blend files (None: `VIEWER3D_BLENDER`, else the
    /// newest installed Blender).
    pub blender: Option<PathBuf>,
}

/// Collects warnings from any thread.
pub struct LoadCtx<'a> {
    pub path: &'a Path,
    pub dir: &'a Path,
    pub progress: &'a Progress,
    pub options: &'a LoadOptions,
    warnings: Mutex<Vec<String>>,
}

impl LoadCtx<'_> {
    pub fn warn(&self, msg: impl Into<String>) {
        util::push_warning(&self.warnings, msg.into());
    }

    pub fn warnings(&self) -> &Mutex<Vec<String>> {
        &self.warnings
    }
}

/// Loads any supported file. Runs on a worker thread; panics inside the
/// format parsers are turned into errors so a bad file never crashes the app.
pub fn load_file(path: &Path, progress: &Progress, options: &LoadOptions) -> Result<LoadedModel> {
    let start = Instant::now();
    let meta = std::fs::metadata(path).with_context(|| format!("cannot open '{}'", path.display()))?;
    if !meta.is_file() {
        bail!("'{}' is not a file", path.display());
    }
    let format = detect_format(path)?;
    if !format_available(format) {
        bail!("{} support is not included in this build", format.name());
    }

    let dir = path.parent().unwrap_or(Path::new("."));
    let ctx = LoadCtx { path, dir, progress, options, warnings: Mutex::new(Vec::new()) };
    progress.set_stage("Reading file");
    progress.set(0.0);

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<Scene> {
        match format {
            Format::Gltf | Format::Glb => gltf::load(&ctx),
            Format::Obj | Format::Fbx => fbx::load(&ctx, format),
            Format::Stl => stl::load(&ctx),
            Format::Ply => ply::load(&ctx),
            Format::ThreeMf => threemf::load(&ctx),
            Format::Blend => blend::load(&ctx),
            #[cfg(feature = "step")]
            Format::Step | Format::Iges | Format::Brep => step::load(&ctx, format),
            #[cfg(not(feature = "step"))]
            Format::Step | Format::Iges | Format::Brep => unreachable!(),
        }
    }));
    let mut scene = match result {
        Ok(r) => r?,
        Err(panic) => {
            let msg = panic
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "unknown error".into());
            return Err(anyhow!("The file could not be parsed ({msg})"));
        }
    };
    progress.check()?;
    progress.set_stage("Preparing");
    finalize(&mut scene, &ctx);
    let bounds = rest_bounds(&scene);
    if bounds.is_empty() {
        bail!("The file contains no visible geometry");
    }
    let stats = compute_stats(&scene);
    let warnings = ctx.warnings.into_inner().unwrap_or_default();
    Ok(LoadedModel {
        scene,
        format,
        file_size: meta.len(),
        warnings,
        stats,
        bounds,
        load_time: start.elapsed(),
    })
}

fn detect_format(path: &Path) -> Result<Format> {
    if let Some(f) = path.extension().and_then(|e| e.to_str()).and_then(Format::from_extension) {
        return Ok(f);
    }
    let mut head = vec![0u8; 1024];
    let n = std::fs::File::open(path)?.read(&mut head)?;
    head.truncate(n);
    Format::sniff(&head).ok_or_else(|| anyhow!("Unrecognised file format"))
}

/// Common clean-up: drop invalid indices and empty primitives, fix attribute
/// lengths so the GPU never reads out of bounds.
fn finalize(scene: &mut Scene, ctx: &LoadCtx) {
    let material_count = scene.materials.len();
    let dropped: usize = scene
        .meshes
        .par_iter_mut()
        .map(|mesh| {
            let mut dropped = 0;
            for p in &mut mesh.primitives {
                let n = p.positions.len();
                match p.topology {
                    Topology::Triangles => dropped += util::sanitize_indices(&mut p.indices, n, 3),
                    Topology::Lines => dropped += util::sanitize_indices(&mut p.indices, n, 2),
                    Topology::Points => p.indices.clear(),
                }
                if p.normals.len() != n {
                    if p.topology == Topology::Triangles && !p.indices.is_empty() {
                        let s = util::smooth_normals(&p.positions, &p.indices, None, 40.0);
                        remap_attributes(p, &s);
                    } else {
                        p.normals.clear();
                    }
                }
                if p.uvs.len() != p.positions.len() {
                    p.uvs.clear();
                }
                if p.colors.len() != p.positions.len() {
                    p.colors.clear();
                }
                if !p.is_skinned() {
                    p.joints.clear();
                    p.weights.clear();
                }
                let n = p.positions.len();
                p.targets.retain_mut(|t| {
                    if t.positions.len() != n {
                        t.positions.clear();
                    }
                    if t.normals.len() != n {
                        t.normals.clear();
                    }
                    !(t.positions.is_empty() && t.normals.is_empty())
                });
                if p.material.is_some_and(|m| m >= material_count) {
                    p.material = None;
                }
            }
            mesh.primitives.retain(|p| p.element_count() > 0);
            dropped
        })
        .sum();
    if dropped > 0 {
        ctx.warn(format!("{dropped} faces referenced missing vertices and were skipped"));
    }

    // Skins referring to missing nodes are dropped.
    let node_count = scene.nodes.len();
    for skin in &mut scene.skins {
        if skin.joints.iter().any(|&j| j >= node_count) {
            skin.joints.clear();
        }
        skin.inverse_bind.resize(skin.joints.len(), Mat4::IDENTITY);
    }
    let skin_count = scene.skins.len();
    let mesh_count = scene.meshes.len();
    for node in &mut scene.nodes {
        if node.skin.is_some_and(|s| s >= skin_count || scene.skins[s].joints.is_empty()) {
            node.skin = None;
        }
        if node.mesh.is_some_and(|m| m >= mesh_count) {
            node.mesh = None;
        }
    }
    for anim in &mut scene.animations {
        anim.channels.retain(|c| c.node < node_count && !c.times.is_empty());
        anim.duration = anim
            .channels
            .iter()
            .filter_map(|c| c.times.last().copied())
            .fold(anim.duration.max(0.0), f32::max);
    }
    scene.animations.retain(|a| !a.channels.is_empty());
}

/// Rebuilds a primitive's attributes after [`util::smooth_normals`].
pub fn remap_attributes(p: &mut crate::scene::Primitive, s: &util::Smoothed) {
    let src_vertex: Vec<u32> = s.source_corner.iter().map(|&c| p.indices[c as usize]).collect();
    let pick = |len: usize| len == p.positions.len();
    if pick(p.uvs.len()) {
        p.uvs = src_vertex.iter().map(|&v| p.uvs[v as usize]).collect();
    }
    if pick(p.colors.len()) {
        p.colors = src_vertex.iter().map(|&v| p.colors[v as usize]).collect();
    }
    if pick(p.joints.len()) && pick(p.weights.len()) {
        p.joints = src_vertex.iter().map(|&v| p.joints[v as usize]).collect();
        p.weights = src_vertex.iter().map(|&v| p.weights[v as usize]).collect();
    }
    for t in &mut p.targets {
        if pick(t.positions.len()) {
            t.positions = src_vertex.iter().map(|&v| t.positions[v as usize]).collect();
        }
        if pick(t.normals.len()) {
            t.normals = src_vertex.iter().map(|&v| t.normals[v as usize]).collect();
        }
    }
    p.positions = s.positions.clone();
    p.normals = s.normals.clone();
    p.indices = s.indices.clone();
}

/// Scene bounds at rest pose, with skinning applied on the CPU (once).
pub fn rest_bounds(scene: &Scene) -> Aabb {
    let globals = scene.rest_globals();
    let order = scene.traversal_order();
    order
        .par_iter()
        .map(|&n| {
            let node = &scene.nodes[n];
            let Some(mesh) = node.mesh.map(|m| &scene.meshes[m]) else { return Aabb::EMPTY };
            let skin = node.skin.map(|s| &scene.skins[s]);
            let mut b = Aabb::EMPTY;
            for p in &mesh.primitives {
                match skin.filter(|_| p.is_skinned()) {
                    Some(skin) => {
                        let joints: Vec<Mat4> = skin
                            .joints
                            .iter()
                            .zip(&skin.inverse_bind)
                            .map(|(&j, ibm)| globals[j] * *ibm)
                            .collect();
                        let pb = p
                            .positions
                            .par_iter()
                            .zip(p.joints.par_iter().zip(p.weights.par_iter()))
                            .fold(
                                || Aabb::EMPTY,
                                |mut acc, (pos, (j, w))| {
                                    let v = Vec3::from_array(*pos);
                                    let mut out = Vec3::ZERO;
                                    let mut total = 0.0;
                                    for k in 0..4 {
                                        if w[k] > 0.0
                                            && let Some(m) = joints.get(j[k] as usize) {
                                                out += m.transform_point3(v) * w[k];
                                                total += w[k];
                                            }
                                    }
                                    acc.extend_point(if total > 0.0 { out / total } else { v });
                                    acc
                                },
                            )
                            .reduce(|| Aabb::EMPTY, |a, b| a.union(&b));
                        b = b.union(&pb);
                    }
                    None => {
                        b = b.union(&Aabb::from_points(&p.positions).transformed(&globals[n]));
                    }
                }
            }
            b
        })
        .reduce(|| Aabb::EMPTY, |a, b| a.union(&b))
}

fn compute_stats(scene: &Scene) -> Stats {
    let mut s = Stats {
        nodes: scene.nodes.len(),
        meshes: scene.meshes.len(),
        materials: scene.materials.len(),
        ..Default::default()
    };
    for &n in &scene.traversal_order() {
        let Some(m) = scene.nodes[n].mesh else { continue };
        s.instances += 1;
        for p in &scene.meshes[m].primitives {
            s.vertices += p.positions.len();
            match p.topology {
                Topology::Triangles => s.triangles += p.indices.len() / 3,
                Topology::Lines => s.lines += p.indices.len() / 2,
                Topology::Points => s.points += p.positions.len(),
            }
            s.skinned |= p.is_skinned() && scene.nodes[n].skin.is_some();
            s.morph_targets |= !p.targets.is_empty();
        }
    }
    s.textures = scene.images.iter().map(|i| (i.name.clone(), i.width, i.height)).collect();
    s.animations = scene.animations.iter().map(|a| (a.name.clone(), a.duration)).collect();
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extension_detection() {
        assert_eq!(Format::from_extension("GLB"), Some(Format::Glb));
        assert_eq!(Format::from_extension("stp"), Some(Format::Step));
        assert_eq!(Format::from_extension("3mf"), Some(Format::ThreeMf));
        assert_eq!(Format::from_extension("Blend"), Some(Format::Blend));
        assert_eq!(Format::from_extension("blend1"), None);
        assert_eq!(Format::from_extension("txt"), None);
    }

    #[test]
    fn content_sniffing() {
        assert_eq!(Format::sniff(b"glTF\x02\0\0\0"), Some(Format::Glb));
        assert_eq!(Format::sniff(b"ply\nformat ascii 1.0\n"), Some(Format::Ply));
        assert_eq!(Format::sniff(b"ISO-10303-21;\nHEADER;"), Some(Format::Step));
        assert_eq!(Format::sniff(b"solid cube\n facet normal 0 0 1\n"), Some(Format::Stl));
        assert_eq!(Format::sniff(b"Kaydara FBX Binary  \0"), Some(Format::Fbx));
        assert_eq!(Format::sniff(b"BLENDER-v502REND"), Some(Format::Blend));
    }
}
