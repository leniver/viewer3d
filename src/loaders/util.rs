//! Helpers shared by the loaders: normals generation, welding, image decoding,
//! texture lookup and progress reporting.

use crate::scene::Image;
use anyhow::{Context, Result, bail};
use glam::Vec3;
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

/// Shared between the loader thread and the UI: progress fraction, current
/// stage, and a cancel flag.
pub struct Progress {
    fraction: AtomicU32,
    cancelled: AtomicBool,
    stage: Mutex<String>,
    /// Part of the bar that `set(0.0..=1.0)` fills (see [`Progress::set_range`]).
    range_start: AtomicU32,
    range_end: AtomicU32,
}

impl Default for Progress {
    fn default() -> Self {
        Self {
            fraction: AtomicU32::new(0),
            cancelled: AtomicBool::new(false),
            stage: Mutex::new(String::new()),
            range_start: AtomicU32::new(0f32.to_bits()),
            range_end: AtomicU32::new(1f32.to_bits()),
        }
    }
}

impl Progress {
    pub fn set(&self, fraction: f32) {
        let start = f32::from_bits(self.range_start.load(Ordering::Relaxed));
        let end = f32::from_bits(self.range_end.load(Ordering::Relaxed));
        let value = start + fraction.clamp(0.0, 1.0) * (end - start);
        self.fraction.store(value.to_bits(), Ordering::Relaxed);
    }

    /// Makes later `set(0.0..=1.0)` calls fill only `start..end` of the bar,
    /// so a loader can run another one as one of its steps.
    pub fn set_range(&self, start: f32, end: f32) {
        self.range_start.store(start.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
        self.range_end.store(end.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
    }

    pub fn fraction(&self) -> f32 {
        f32::from_bits(self.fraction.load(Ordering::Relaxed))
    }

    pub fn set_stage(&self, stage: &str) {
        if let Ok(mut s) = self.stage.lock() {
            s.clear();
            s.push_str(stage);
        }
    }

    pub fn stage(&self) -> String {
        self.stage.lock().map(|s| s.clone()).unwrap_or_default()
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }

    pub fn check(&self) -> Result<()> {
        if self.is_cancelled() {
            bail!(Cancelled);
        }
        Ok(())
    }
}

/// Error marker for a user-cancelled load.
#[derive(Debug)]
pub struct Cancelled;

impl std::fmt::Display for Cancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("loading cancelled")
    }
}

impl std::error::Error for Cancelled {}

// ---------------------------------------------------------------------------
// Normals
// ---------------------------------------------------------------------------

/// Result of [`smooth_normals`]: new vertices referencing the input.
pub struct Smoothed {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    /// For every output vertex, the input corner (index into `indices`) it came from.
    pub source_corner: Vec<u32>,
    pub indices: Vec<u32>,
}

/// Computes normals with a crease angle: faces whose normals differ by more
/// than the angle get split vertices (sharp edges), others are smoothed
/// (area-weighted). `corner_key` optionally prevents corners with different
/// keys (e.g. different colour or UV) from sharing an output vertex.
pub fn smooth_normals(
    positions: &[[f32; 3]],
    indices: &[u32],
    corner_key: Option<&[u32]>,
    crease_degrees: f32,
) -> Smoothed {
    let tri_count = indices.len() / 3;
    let vcount = positions.len();
    let cos_crease = crease_degrees.to_radians().cos();

    // Area-weighted face normals.
    let face_normals: Vec<Vec3> = (0..tri_count)
        .into_par_iter()
        .map(|t| {
            let a = Vec3::from_array(positions[indices[3 * t] as usize]);
            let b = Vec3::from_array(positions[indices[3 * t + 1] as usize]);
            let c = Vec3::from_array(positions[indices[3 * t + 2] as usize]);
            let n = (b - a).cross(c - a);
            if n.is_finite() { n } else { Vec3::ZERO }
        })
        .collect();
    let face_units: Vec<Vec3> = face_normals.par_iter().map(|n| n.normalize_or_zero()).collect();

    // Vertex -> corners adjacency (CSR).
    let mut offsets = vec![0u32; vcount + 1];
    for &i in indices {
        offsets[i as usize + 1] += 1;
    }
    for i in 0..vcount {
        offsets[i + 1] += offsets[i];
    }
    let mut fill = offsets.clone();
    let mut corners = vec![0u32; indices.len()];
    for (corner, &i) in indices.iter().enumerate() {
        let slot = &mut fill[i as usize];
        corners[*slot as usize] = corner as u32;
        *slot += 1;
    }

    // Per-corner normal: sum of adjacent faces within the crease angle.
    let corner_normals: Vec<Vec3> = (0..indices.len())
        .into_par_iter()
        .map(|corner| {
            let face = corner / 3;
            let own = face_units[face];
            let v = indices[corner] as usize;
            let mut sum = Vec3::ZERO;
            for &other in &corners[offsets[v] as usize..offsets[v + 1] as usize] {
                let f = other as usize / 3;
                if own == Vec3::ZERO || face_units[f].dot(own) >= cos_crease {
                    sum += face_normals[f];
                }
            }
            let n = sum.normalize_or_zero();
            if n == Vec3::ZERO { if own == Vec3::ZERO { Vec3::Y } else { own } } else { n }
        })
        .collect();

    // Per vertex, group its corners into output vertices (same key, ~same normal).
    let groups: Vec<Vec<(u32, Vec3, u32)>> = (0..vcount)
        .into_par_iter()
        .map(|v| {
            let mut out: Vec<(u32, Vec3, u32)> = Vec::new(); // (first corner, normal, key)
            for &corner in &corners[offsets[v] as usize..offsets[v + 1] as usize] {
                let n = corner_normals[corner as usize];
                let key = corner_key.map_or(0, |k| k[corner as usize]);
                if !out.iter().any(|(_, gn, gk)| *gk == key && gn.dot(n) > 0.9999) {
                    out.push((corner, n, key));
                }
            }
            out
        })
        .collect();

    let mut base = Vec::with_capacity(vcount + 1);
    let mut total = 0u32;
    for g in &groups {
        base.push(total);
        total += g.len() as u32;
    }

    let mut out_positions = Vec::with_capacity(total as usize);
    let mut out_normals = Vec::with_capacity(total as usize);
    let mut source_corner = Vec::with_capacity(total as usize);
    for (v, g) in groups.iter().enumerate() {
        for (corner, n, _) in g {
            out_positions.push(positions[v]);
            out_normals.push(n.to_array());
            source_corner.push(*corner);
        }
    }

    let out_indices: Vec<u32> = (0..indices.len())
        .into_par_iter()
        .map(|corner| {
            let v = indices[corner] as usize;
            let n = corner_normals[corner];
            let key = corner_key.map_or(0, |k| k[corner]);
            let local = groups[v]
                .iter()
                .position(|(_, gn, gk)| *gk == key && gn.dot(n) > 0.9999)
                .unwrap_or(0);
            base[v] + local as u32
        })
        .collect();

    Smoothed { positions: out_positions, normals: out_normals, source_corner, indices: out_indices }
}

/// Welds identical positions. Returns unique positions and one index per input
/// point. Parallel sort-based, deterministic.
pub fn weld_positions(points: &[[f32; 3]], keys: Option<&[u32]>) -> (Vec<[f32; 3]>, Vec<u32>) {
    let canon = |v: f32| if v == 0.0 { 0u32 } else { v.to_bits() };
    let mut order: Vec<u32> = (0..points.len() as u32).collect();
    let key_of = |i: u32| {
        let p = points[i as usize];
        (canon(p[0]), canon(p[1]), canon(p[2]), keys.map_or(0, |k| k[i as usize]))
    };
    order.par_sort_unstable_by_key(|&i| key_of(i));

    let mut unique = Vec::new();
    let mut remap = vec![0u32; points.len()];
    let mut last = None;
    for &i in &order {
        let k = key_of(i);
        if last != Some(k) {
            unique.push(points[i as usize]);
            last = Some(k);
        }
        remap[i as usize] = unique.len() as u32 - 1;
    }
    (unique, remap)
}

/// Removes triangles/lines referencing vertices out of range.
pub fn sanitize_indices(indices: &mut Vec<u32>, vertex_count: usize, arity: usize) -> usize {
    if indices.iter().all(|&i| (i as usize) < vertex_count) {
        indices.truncate(indices.len() / arity * arity);
        return 0;
    }
    let before = indices.len() / arity;
    let kept: Vec<u32> = indices
        .chunks_exact(arity)
        .filter(|c| c.iter().all(|&i| (i as usize) < vertex_count))
        .flatten()
        .copied()
        .collect();
    *indices = kept;
    before - indices.len() / arity
}

// ---------------------------------------------------------------------------
// Images
// ---------------------------------------------------------------------------

/// Decodes an encoded image (PNG, JPEG, TGA, ...) into RGBA8.
pub fn decode_image(bytes: &[u8], name: &str) -> Result<Image> {
    let format = image::guess_format(bytes).ok();
    let decoded = match format {
        Some(f) => image::load_from_memory_with_format(bytes, f),
        // TGA has no magic number: fall back on the file extension.
        None => {
            let ext = Path::new(name).extension().and_then(|e| e.to_str()).unwrap_or("");
            match image::ImageFormat::from_extension(ext) {
                Some(f) => image::load_from_memory_with_format(bytes, f),
                None => image::load_from_memory(bytes),
            }
        }
    }
    .with_context(|| format!("cannot decode image '{name}'"))?;
    let rgba = decoded.into_rgba8();
    let (width, height) = rgba.dimensions();
    Ok(Image { name: name.to_owned(), width, height, pixels: Arc::new(rgba.into_raw()) })
}

/// Source of an image to decode.
pub enum ImageSource {
    Bytes(Vec<u8>),
    File(PathBuf),
}

/// Decodes many images in parallel. Failures become warnings and `None`.
pub fn decode_images(sources: Vec<(String, ImageSource)>, warnings: &Mutex<Vec<String>>) -> Vec<Option<Image>> {
    sources
        .into_par_iter()
        .map(|(name, source)| {
            let bytes = match source {
                ImageSource::Bytes(b) => b,
                ImageSource::File(path) => match std::fs::read(&path) {
                    Ok(b) => b,
                    Err(e) => {
                        push_warning(warnings, format!("Texture '{}' could not be read: {e}", path.display()));
                        return None;
                    }
                },
            };
            match decode_image(&bytes, &name) {
                Ok(img) => Some(img),
                Err(e) => {
                    push_warning(warnings, format!("{e:#}"));
                    None
                }
            }
        })
        .collect()
}

pub fn push_warning(warnings: &Mutex<Vec<String>>, msg: String) {
    if let Ok(mut w) = warnings.lock()
        && !w.contains(&msg) {
            w.push(msg);
        }
}

/// Finds a texture referenced by a model file. Exporters often store absolute
/// paths from the author's machine, so also try the file name next to the
/// model and in common texture folders, case-insensitively.
pub fn resolve_texture_path(model_dir: &Path, candidates: &[&str]) -> Option<PathBuf> {
    let mut names: Vec<String> = Vec::new();
    for c in candidates.iter().filter(|c| !c.is_empty()) {
        let normalized = c.replace('\\', "/");
        let p = Path::new(&normalized);
        let full = if p.is_absolute() { p.to_path_buf() } else { model_dir.join(p) };
        if full.is_file() {
            return Some(full);
        }
        if let Some(file) = p.file_name().and_then(|f| f.to_str())
            && !names.iter().any(|n| n == file) {
                names.push(file.to_owned());
            }
    }
    let dirs = [
        model_dir.to_path_buf(),
        model_dir.join("textures"),
        model_dir.join("Textures"),
        model_dir.join("texture"),
        model_dir.join("images"),
        model_dir.join("maps"),
        model_dir.join("..").join("textures"),
        model_dir.join("..").join("Textures"),
    ];
    for dir in &dirs {
        for name in &names {
            let p = dir.join(name);
            if p.is_file() {
                return Some(p);
            }
        }
    }
    // Case-insensitive match (matters on Linux).
    for dir in &dirs {
        let Ok(entries) = std::fs::read_dir(dir) else { continue };
        for entry in entries.flatten() {
            let file_name = entry.file_name();
            let file_name = file_name.to_string_lossy();
            if names.iter().any(|n| n.eq_ignore_ascii_case(&file_name)) {
                return Some(entry.path());
            }
        }
    }
    None
}

/// Percent-decodes a URI path component (glTF stores relative URIs).
pub fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len()
            && let Ok(v) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                out.push(v);
                i += 3;
                continue;
            }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn progress_range() {
        let p = Progress::default();
        p.set(0.5);
        assert_eq!(p.fraction(), 0.5);
        p.set_range(0.6, 1.0);
        p.set(0.5);
        assert!((p.fraction() - 0.8).abs() < 1e-6);
        p.set(2.0);
        assert_eq!(p.fraction(), 1.0);
    }

    #[test]
    fn cube_keeps_sharp_edges() {
        // Unit cube, 8 shared vertices, 12 triangles.
        let p = [
            [0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.],
            [0., 0., 1.], [1., 0., 1.], [1., 1., 1.], [0., 1., 1.],
        ];
        let idx: Vec<u32> = vec![
            0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4,
            2, 3, 7, 2, 7, 6, 1, 2, 6, 1, 6, 5, 0, 4, 7, 0, 7, 3,
        ];
        let s = smooth_normals(&p, &idx, None, 40.0);
        // Each cube corner splits into 3 vertices (one per adjacent face).
        assert_eq!(s.positions.len(), 24);
        for n in &s.normals {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!((len - 1.0).abs() < 1e-5);
            assert!(n.iter().filter(|c| c.abs() > 0.999).count() == 1, "axis-aligned normal expected");
        }
        // Smoothing everything merges back to 8 vertices.
        let s = smooth_normals(&p, &idx, None, 180.0);
        assert_eq!(s.positions.len(), 8);
    }

    #[test]
    fn weld_merges_duplicates() {
        let pts = [[0., 0., 0.], [1., 0., 0.], [0., 0., 0.], [-0.0, 0., 0.]];
        let (u, remap) = weld_positions(&pts, None);
        assert_eq!(u.len(), 2);
        assert_eq!(remap[0], remap[2]);
        assert_eq!(remap[0], remap[3]);
        assert_ne!(remap[0], remap[1]);
    }

    #[test]
    fn sanitize_drops_bad_triangles() {
        let mut idx = vec![0, 1, 2, 0, 9, 2, 1, 2, 0];
        assert_eq!(sanitize_indices(&mut idx, 3, 3), 1);
        assert_eq!(idx, vec![0, 1, 2, 1, 2, 0]);
    }

    #[test]
    fn percent_decoding() {
        assert_eq!(percent_decode("my%20texture.png"), "my texture.png");
        assert_eq!(percent_decode("a%zz"), "a%zz");
    }
}
