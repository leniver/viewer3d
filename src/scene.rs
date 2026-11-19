//! Format-independent scene description produced by every loader.

use glam::{Mat4, Quat, Vec3};
use std::sync::Arc;

#[derive(Default)]
pub struct Scene {
    pub nodes: Vec<Node>,
    pub roots: Vec<usize>,
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
    pub images: Vec<Image>,
    pub textures: Vec<Texture>,
    pub skins: Vec<Skin>,
    pub animations: Vec<Animation>,
    /// Up axis of the file's coordinate system.
    pub up_axis: UpAxis,
    /// Length unit, when the format defines one (e.g. "mm").
    pub unit: Option<String>,
    /// Program that converted the file for display (e.g. "Blender 5.2.3").
    pub converted_with: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum UpAxis {
    #[default]
    Y,
    Z,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self { translation: Vec3::ZERO, rotation: Quat::IDENTITY, scale: Vec3::ONE }
    }
}

impl Transform {
    pub fn matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }

    pub fn from_matrix(m: &Mat4) -> Self {
        let (scale, rotation, translation) = m.to_scale_rotation_translation();
        let rotation = if rotation.is_finite() { rotation.normalize() } else { Quat::IDENTITY };
        Self { translation, rotation, scale }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Node {
    pub name: String,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    /// Rest (bind) local transform.
    pub transform: Transform,
    pub mesh: Option<usize>,
    pub skin: Option<usize>,
    /// Default morph target weights for this node's mesh.
    pub weights: Vec<f32>,
}

#[derive(Clone, Debug, Default)]
pub struct Mesh {
    pub name: String,
    pub primitives: Vec<Primitive>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Topology {
    Triangles,
    Lines,
    Points,
}

/// Purpose of a primitive, used for display toggles.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PrimitiveRole {
    #[default]
    Surface,
    /// CAD feature edges (STEP/IGES), shown with the "Edges" toggle.
    FeatureEdges,
}

#[derive(Clone, Debug)]
pub struct Primitive {
    pub topology: Topology,
    pub role: PrimitiveRole,
    pub positions: Vec<[f32; 3]>,
    /// Always present for triangles after loading; may be empty for lines/points.
    pub normals: Vec<[f32; 3]>,
    /// Empty when absent. Origin at the top-left of the image (glTF convention).
    pub uvs: Vec<[f32; 2]>,
    /// sRGB-encoded RGB + linear alpha. Empty when absent.
    pub colors: Vec<[u8; 4]>,
    pub joints: Vec<[u16; 4]>,
    pub weights: Vec<[f32; 4]>,
    pub targets: Vec<MorphTarget>,
    /// Empty = non-indexed (points). Triangles and lines are always indexed.
    pub indices: Vec<u32>,
    pub material: Option<usize>,
}

impl Primitive {
    pub fn new(topology: Topology) -> Self {
        Self {
            topology,
            role: PrimitiveRole::Surface,
            positions: Vec::new(),
            normals: Vec::new(),
            uvs: Vec::new(),
            colors: Vec::new(),
            joints: Vec::new(),
            weights: Vec::new(),
            targets: Vec::new(),
            indices: Vec::new(),
            material: None,
        }
    }

    pub fn is_skinned(&self) -> bool {
        !self.joints.is_empty() && self.joints.len() == self.positions.len() && self.weights.len() == self.positions.len()
    }

    pub fn element_count(&self) -> usize {
        match self.topology {
            Topology::Triangles => self.indices.len() / 3,
            Topology::Lines => self.indices.len() / 2,
            Topology::Points => self.positions.len(),
        }
    }
}

/// Morph target deltas; each vector is empty or has one entry per vertex.
#[derive(Clone, Debug, Default)]
pub struct MorphTarget {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AlphaMode {
    #[default]
    Opaque,
    Mask,
    Blend,
}

/// 2x3 affine UV transform, row-major: u' = a*u + b*v + c, v' = d*u + e*v + f.
pub type UvTransform = [f32; 6];

#[derive(Clone, Copy, Debug)]
pub struct TextureRef {
    pub texture: usize,
    pub transform: Option<UvTransform>,
}

impl TextureRef {
    pub fn new(texture: usize) -> Self {
        Self { texture, transform: None }
    }
}

#[derive(Clone, Debug)]
pub struct Material {
    pub name: String,
    /// Linear RGBA.
    pub base_color: [f32; 4],
    pub base_color_texture: Option<TextureRef>,
    pub metallic: f32,
    pub roughness: f32,
    pub metallic_texture: Option<TextureRef>,
    /// Channel of `metallic_texture` holding the value (glTF: 2 = blue).
    pub metallic_channel: u8,
    pub roughness_texture: Option<TextureRef>,
    /// Channel of `roughness_texture` holding the value (glTF: 1 = green).
    pub roughness_channel: u8,
    /// The roughness texture/value stores glossiness (1 - roughness).
    pub glossiness_workflow: bool,
    pub normal_texture: Option<TextureRef>,
    pub normal_scale: f32,
    pub occlusion_texture: Option<TextureRef>,
    pub occlusion_strength: f32,
    /// Linear RGB, already multiplied by any strength factor.
    pub emissive: [f32; 3],
    pub emissive_texture: Option<TextureRef>,
    pub alpha_mode: AlphaMode,
    pub alpha_cutoff: f32,
    pub double_sided: bool,
    pub unlit: bool,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            name: String::new(),
            base_color: [0.8, 0.8, 0.8, 1.0],
            base_color_texture: None,
            metallic: 0.0,
            roughness: 0.6,
            metallic_texture: None,
            metallic_channel: 2,
            roughness_texture: None,
            roughness_channel: 1,
            glossiness_workflow: false,
            normal_texture: None,
            normal_scale: 1.0,
            occlusion_texture: None,
            occlusion_strength: 1.0,
            emissive: [0.0; 3],
            emissive_texture: None,
            alpha_mode: AlphaMode::Opaque,
            alpha_cutoff: 0.5,
            double_sided: true,
            unlit: false,
        }
    }
}

/// Decoded RGBA8 image (shared by textures that differ only by sampler).
#[derive(Clone)]
pub struct Image {
    pub name: String,
    pub width: u32,
    pub height: u32,
    /// Tightly packed RGBA8 rows, top row first. Released after GPU upload.
    pub pixels: Arc<Vec<u8>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Wrap {
    #[default]
    Repeat,
    Mirror,
    Clamp,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Sampler {
    pub wrap_u: Wrap,
    pub wrap_v: Wrap,
    pub mag_nearest: bool,
    pub min_nearest: bool,
}

impl Default for Sampler {
    fn default() -> Self {
        Self { wrap_u: Wrap::Repeat, wrap_v: Wrap::Repeat, mag_nearest: false, min_nearest: false }
    }
}

#[derive(Clone, Debug)]
pub struct Texture {
    pub image: usize,
    pub sampler: Sampler,
}

#[derive(Clone, Debug, Default)]
pub struct Skin {
    pub joints: Vec<usize>,
    pub inverse_bind: Vec<Mat4>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Property {
    Translation,
    Rotation,
    Scale,
    /// Weight of one morph target of the node's mesh.
    MorphWeight(u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interpolation {
    Step,
    Linear,
    CubicSpline,
}

#[derive(Clone, Debug)]
pub struct Channel {
    pub node: usize,
    pub property: Property,
    pub interpolation: Interpolation,
    /// Key times in seconds, ascending.
    pub times: Vec<f32>,
    /// Flat values: per key `components` floats (×3 with in/out tangents for
    /// cubic splines, ordered in-tangent, value, out-tangent).
    pub values: Vec<f32>,
}

impl Property {
    pub fn components(self) -> usize {
        match self {
            Property::Translation | Property::Scale => 3,
            Property::Rotation => 4,
            Property::MorphWeight(_) => 1,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Animation {
    pub name: String,
    pub duration: f32,
    pub channels: Vec<Channel>,
}

/// Axis-aligned bounding box.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Default for Aabb {
    fn default() -> Self {
        Self::EMPTY
    }
}

impl Aabb {
    pub const EMPTY: Aabb = Aabb { min: Vec3::splat(f32::INFINITY), max: Vec3::splat(f32::NEG_INFINITY) };

    pub fn is_empty(&self) -> bool {
        !(self.min.x <= self.max.x && self.min.y <= self.max.y && self.min.z <= self.max.z)
    }

    pub fn extend_point(&mut self, p: Vec3) {
        if p.is_finite() {
            self.min = self.min.min(p);
            self.max = self.max.max(p);
        }
    }

    pub fn union(&self, other: &Aabb) -> Aabb {
        if other.is_empty() {
            return *self;
        }
        if self.is_empty() {
            return *other;
        }
        Aabb { min: self.min.min(other.min), max: self.max.max(other.max) }
    }

    pub fn from_points(points: &[[f32; 3]]) -> Aabb {
        use rayon::prelude::*;
        points
            .par_chunks(65536)
            .map(|chunk| {
                let mut b = Aabb::EMPTY;
                for p in chunk {
                    b.extend_point(Vec3::from_array(*p));
                }
                b
            })
            .reduce(|| Aabb::EMPTY, |a, b| a.union(&b))
    }

    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    pub fn size(&self) -> Vec3 {
        if self.is_empty() { Vec3::ZERO } else { self.max - self.min }
    }

    pub fn transformed(&self, m: &Mat4) -> Aabb {
        if self.is_empty() {
            return *self;
        }
        let mut out = Aabb::EMPTY;
        for i in 0..8 {
            let p = Vec3::new(
                if i & 1 == 0 { self.min.x } else { self.max.x },
                if i & 2 == 0 { self.min.y } else { self.max.y },
                if i & 4 == 0 { self.min.z } else { self.max.z },
            );
            out.extend_point(m.transform_point3(p));
        }
        out
    }
}

impl Scene {
    /// Parent-before-child order over all nodes reachable from the roots.
    pub fn traversal_order(&self) -> Vec<usize> {
        let mut order = Vec::with_capacity(self.nodes.len());
        let mut visited = vec![false; self.nodes.len()];
        let mut stack: Vec<usize> = self.roots.iter().rev().copied().collect();
        while let Some(n) = stack.pop() {
            if n >= self.nodes.len() || visited[n] {
                continue;
            }
            visited[n] = true;
            order.push(n);
            for &c in self.nodes[n].children.iter().rev() {
                stack.push(c);
            }
        }
        order
    }

    /// Global matrices of every node at rest pose.
    pub fn rest_globals(&self) -> Vec<Mat4> {
        let mut globals = vec![Mat4::IDENTITY; self.nodes.len()];
        for n in self.traversal_order() {
            let local = self.nodes[n].transform.matrix();
            globals[n] = match self.nodes[n].parent {
                Some(p) => globals[p] * local,
                None => local,
            };
        }
        globals
    }

    /// Adds a node and links it to its parent (or makes it a root).
    pub fn add_node(&mut self, mut node: Node, parent: Option<usize>) -> usize {
        let index = self.nodes.len();
        node.parent = parent;
        self.nodes.push(node);
        match parent {
            Some(p) => self.nodes[p].children.push(index),
            None => self.roots.push(index),
        }
        index
    }

    pub fn add_material(&mut self, material: Material) -> usize {
        self.materials.push(material);
        self.materials.len() - 1
    }
}

/// sRGB transfer function, for converting file colours to linear space.
pub fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

pub fn linear_to_srgb(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.003_130_8 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 }
}

/// Encodes a linear RGBA colour to the vertex colour format (sRGB RGB, linear alpha).
pub fn encode_vertex_color(rgba: [f32; 4]) -> [u8; 4] {
    let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    [q(linear_to_srgb(rgba[0])), q(linear_to_srgb(rgba[1])), q(linear_to_srgb(rgba[2])), q(rgba[3])]
}
