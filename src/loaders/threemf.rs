//! 3MF loader: core spec, materials extension (base materials, colour groups,
//! 2D textures) and production extension (objects split across several model
//! files, as written by Bambu Studio / PrusaSlicer).

use super::LoadCtx;
use super::util::{self, ImageSource};
use crate::scene::*;
use anyhow::{Context, Result, anyhow, bail};
use glam::{Mat4, Vec4};
use rayon::prelude::*;
use std::collections::HashMap;
use std::io::{Cursor, Read};

#[derive(Default)]
struct ModelFile {
    unit: Option<String>,
    objects: HashMap<u32, Object>,
    base_materials: HashMap<u32, Vec<[u8; 4]>>,
    color_groups: HashMap<u32, Vec<[u8; 4]>>,
    tex_groups: HashMap<u32, TexGroup>,
    textures: HashMap<u32, Tex2D>,
    build: Vec<Placement>,
}

struct Object {
    name: String,
    pid: Option<u32>,
    pindex: Option<u32>,
    content: ObjectContent,
}

enum ObjectContent {
    Mesh { vertices: Vec<[f32; 3]>, triangles: Vec<Triangle> },
    Components(Vec<Placement>),
}

struct Triangle {
    v: [u32; 3],
    pid: Option<u32>,
    p: [Option<u32>; 3],
}

/// Build item or component: object reference + transform (+ model file).
struct Placement {
    object: u32,
    transform: Mat4,
    path: Option<String>,
}

struct TexGroup {
    texture: u32,
    coords: Vec<[f32; 2]>,
}

struct Tex2D {
    path: String,
    sampler: Sampler,
}

fn attr<'a>(node: &roxmltree::Node<'a, '_>, local: &str) -> Option<&'a str> {
    node.attributes().find(|a| a.name() == local).map(|a| a.value())
}

fn attr_u32(node: &roxmltree::Node, local: &str) -> Option<u32> {
    attr(node, local).and_then(|v| v.trim().parse().ok())
}

fn attr_f32(node: &roxmltree::Node, local: &str) -> f32 {
    attr(node, local).and_then(|v| v.trim().parse().ok()).unwrap_or(0.0)
}

/// "#RRGGBB" or "#RRGGBBAA" (sRGB).
fn parse_color(s: &str) -> Option<[u8; 4]> {
    let h = s.trim().strip_prefix('#')?;
    let byte = |i: usize| u8::from_str_radix(h.get(i..i + 2)?, 16).ok();
    match h.len() {
        6 => Some([byte(0)?, byte(2)?, byte(4)?, 255]),
        8 => Some([byte(0)?, byte(2)?, byte(4)?, byte(6)?]),
        _ => None,
    }
}

/// 3MF matrices are row-vector 4x3: "m00 m01 m02 m10 m11 m12 m20 m21 m22 m30 m31 m32".
fn parse_transform(s: Option<&str>) -> Mat4 {
    let Some(s) = s else { return Mat4::IDENTITY };
    let v: Vec<f32> = s.split_ascii_whitespace().filter_map(|t| t.parse().ok()).collect();
    if v.len() != 12 {
        return Mat4::IDENTITY;
    }
    Mat4::from_cols(
        Vec4::new(v[0], v[1], v[2], 0.0),
        Vec4::new(v[3], v[4], v[5], 0.0),
        Vec4::new(v[6], v[7], v[8], 0.0),
        Vec4::new(v[9], v[10], v[11], 1.0),
    )
}

fn normalize_path(p: &str) -> String {
    p.trim().trim_start_matches('/').replace('\\', "/")
}

fn parse_model(xml: &str) -> Result<ModelFile> {
    let opts = roxmltree::ParsingOptions { allow_dtd: true, ..Default::default() };
    let doc = roxmltree::Document::parse_with_options(xml, opts).context("invalid 3MF model XML")?;
    let root = doc.root_element();
    if root.tag_name().name() != "model" {
        bail!("not a 3MF model file");
    }
    let mut m = ModelFile {
        unit: Some(
            match attr(&root, "unit").unwrap_or("millimeter") {
                "micron" => "µm",
                "millimeter" => "mm",
                "centimeter" => "cm",
                "inch" => "in",
                "foot" => "ft",
                "meter" => "m",
                other => other,
            }
            .to_owned(),
        ),
        ..Default::default()
    };

    for section in root.children().filter(|n| n.is_element()) {
        match section.tag_name().name() {
            "resources" => {
                for res in section.children().filter(|n| n.is_element()) {
                    let Some(id) = attr_u32(&res, "id") else { continue };
                    match res.tag_name().name() {
                        "basematerials" => {
                            let colors = res
                                .children()
                                .filter(|n| n.tag_name().name() == "base")
                                .map(|b| attr(&b, "displaycolor").and_then(parse_color).unwrap_or([204, 204, 204, 255]))
                                .collect();
                            m.base_materials.insert(id, colors);
                        }
                        "colorgroup" => {
                            let colors = res
                                .children()
                                .filter(|n| n.tag_name().name() == "color")
                                .map(|c| attr(&c, "color").and_then(parse_color).unwrap_or([204, 204, 204, 255]))
                                .collect();
                            m.color_groups.insert(id, colors);
                        }
                        "texture2d" => {
                            let tile = |name: &str| match attr(&res, name).unwrap_or("wrap") {
                                "mirror" => Wrap::Mirror,
                                "clamp" | "none" => Wrap::Clamp,
                                _ => Wrap::Repeat,
                            };
                            let nearest = attr(&res, "filter") == Some("nearest");
                            m.textures.insert(
                                id,
                                Tex2D {
                                    path: normalize_path(attr(&res, "path").unwrap_or_default()),
                                    sampler: Sampler {
                                        wrap_u: tile("tilestyleu"),
                                        wrap_v: tile("tilestylev"),
                                        mag_nearest: nearest,
                                        min_nearest: nearest,
                                    },
                                },
                            );
                        }
                        "texture2dgroup" => {
                            let coords = res
                                .children()
                                .filter(|n| n.tag_name().name() == "tex2coord")
                                .map(|c| [attr_f32(&c, "u"), 1.0 - attr_f32(&c, "v")])
                                .collect();
                            m.tex_groups.insert(id, TexGroup { texture: attr_u32(&res, "texid").unwrap_or(0), coords });
                        }
                        "object" => {
                            if let Some(obj) = parse_object(&res) {
                                m.objects.insert(id, obj);
                            }
                        }
                        _ => {}
                    }
                }
            }
            "build" => {
                for item in section.children().filter(|n| n.tag_name().name() == "item") {
                    let Some(object) = attr_u32(&item, "objectid") else { continue };
                    m.build.push(Placement {
                        object,
                        transform: parse_transform(attr(&item, "transform")),
                        path: attr(&item, "path").map(normalize_path),
                    });
                }
            }
            _ => {}
        }
    }
    Ok(m)
}

fn parse_object(node: &roxmltree::Node) -> Option<Object> {
    let name = attr(node, "name").unwrap_or_default().to_owned();
    let pid = attr_u32(node, "pid");
    let pindex = attr_u32(node, "pindex");
    for child in node.children().filter(|n| n.is_element()) {
        match child.tag_name().name() {
            "mesh" => {
                let mut vertices = Vec::new();
                let mut triangles = Vec::new();
                for part in child.children().filter(|n| n.is_element()) {
                    match part.tag_name().name() {
                        "vertices" => {
                            vertices = part
                                .children()
                                .filter(|n| n.tag_name().name() == "vertex")
                                .map(|v| [attr_f32(&v, "x"), attr_f32(&v, "y"), attr_f32(&v, "z")])
                                .collect();
                        }
                        "triangles" => {
                            triangles = part
                                .children()
                                .filter(|n| n.tag_name().name() == "triangle")
                                .filter_map(|t| {
                                    Some(Triangle {
                                        v: [attr_u32(&t, "v1")?, attr_u32(&t, "v2")?, attr_u32(&t, "v3")?],
                                        pid: attr_u32(&t, "pid"),
                                        p: [attr_u32(&t, "p1"), attr_u32(&t, "p2"), attr_u32(&t, "p3")],
                                    })
                                })
                                .collect();
                        }
                        _ => {}
                    }
                }
                return Some(Object { name, pid, pindex, content: ObjectContent::Mesh { vertices, triangles } });
            }
            "components" => {
                let comps = child
                    .children()
                    .filter(|n| n.tag_name().name() == "component")
                    .filter_map(|c| {
                        Some(Placement {
                            object: attr_u32(&c, "objectid")?,
                            transform: parse_transform(attr(&c, "transform")),
                            path: attr(&c, "path").map(normalize_path),
                        })
                    })
                    .collect();
                return Some(Object { name, pid, pindex, content: ObjectContent::Components(comps) });
            }
            _ => {}
        }
    }
    None
}

struct Package {
    zip: zip::ZipArchive<Cursor<Vec<u8>>>,
}

impl Package {
    fn read(&mut self, path: &str) -> Result<Vec<u8>> {
        let name = normalize_path(path);
        let actual = if self.zip.by_name(&name).is_ok() {
            name.clone()
        } else {
            self.entry_names()
                .into_iter()
                .find(|n| n.eq_ignore_ascii_case(&name))
                .ok_or_else(|| anyhow!("'{name}' is missing from the 3MF package"))?
        };
        let mut f = self.zip.by_name(&actual)?;
        let mut buf = Vec::with_capacity(f.size() as usize);
        f.read_to_end(&mut buf)?;
        Ok(buf)
    }

    fn root_model_path(&mut self) -> Result<String> {
        if let Ok(rels) = self.read("_rels/.rels") {
            let text = String::from_utf8_lossy(&rels);
            if let Ok(doc) = roxmltree::Document::parse(&text) {
                for r in doc.descendants().filter(|n| n.tag_name().name() == "Relationship") {
                    if attr(&r, "Type").is_some_and(|t| t.ends_with("/3dmodel"))
                        && let Some(target) = attr(&r, "Target") {
                            return Ok(normalize_path(target));
                        }
                }
            }
        }
        self.entry_names()
            .into_iter()
            .filter(|n| n.to_ascii_lowercase().ends_with(".model"))
            .min_by_key(|n| (!n.eq_ignore_ascii_case("3D/3dmodel.model"), n.len()))
            .ok_or_else(|| anyhow!("no 3D model found in the 3MF package"))
    }

    fn entry_names(&self) -> Vec<String> {
        self.zip.file_names().filter_map(|n| n.ok()).map(|n| n.into_owned()).collect()
    }
}

pub fn load(ctx: &LoadCtx) -> Result<Scene> {
    let data = std::fs::read(ctx.path).with_context(|| format!("cannot read '{}'", ctx.path.display()))?;
    let mut pkg = Package { zip: zip::ZipArchive::new(Cursor::new(data)).context("not a valid 3MF (zip) file")? };
    let root_path = pkg.root_model_path()?;
    ctx.progress.set(0.05);

    // Parse the root model, then any model files it references (in parallel).
    let mut files: HashMap<String, ModelFile> = HashMap::new();
    let mut pending = vec![root_path.clone()];
    while !pending.is_empty() {
        ctx.progress.check()?;
        let blobs: Vec<(String, Vec<u8>)> =
            pending.drain(..).map(|p| pkg.read(&p).map(|b| (p, b))).collect::<Result<_>>()?;
        let parsed: Vec<(String, ModelFile)> = blobs
            .into_par_iter()
            .map(|(p, b)| {
                let text = String::from_utf8_lossy(&b);
                parse_model(&text).map(|m| (p, m)).with_context(|| format!("in '{}'", normalize_path(&root_path)))
            })
            .collect::<Result<_>>()?;
        for (p, m) in parsed {
            for obj in m.objects.values() {
                if let ObjectContent::Components(comps) = &obj.content {
                    for c in comps {
                        if let Some(cp) = &c.path
                            && !files.contains_key(cp) && !pending.contains(cp) && *cp != p {
                                pending.push(cp.clone());
                            }
                    }
                }
            }
            files.insert(p, m);
        }
        pending.retain(|p| !files.contains_key(p));
    }
    ctx.progress.set(0.4);
    ctx.progress.set_stage("Building meshes");

    let root = files.get(&root_path).ok_or_else(|| anyhow!("root model missing"))?;
    let mut scene = Scene { up_axis: UpAxis::Z, unit: root.unit.clone(), ..Default::default() };

    // Textures (decoded in parallel).
    let mut tex_keys: Vec<(String, u32)> = Vec::new();
    let mut sources = Vec::new();
    for (file, m) in &files {
        for (id, t) in &m.textures {
            match pkg.read(&t.path) {
                Ok(bytes) => {
                    tex_keys.push((file.clone(), *id));
                    sources.push((t.path.clone(), ImageSource::Bytes(bytes)));
                }
                Err(e) => ctx.warn(format!("{e:#}")),
            }
        }
    }
    let images = util::decode_images(sources, ctx.warnings());
    let mut texture_index: HashMap<(String, u32), usize> = HashMap::new();
    for ((file, id), img) in tex_keys.into_iter().zip(images) {
        if let Some(img) = img {
            scene.images.push(img);
            let sampler = files[&file].textures[&id].sampler;
            scene.textures.push(Texture { image: scene.images.len() - 1, sampler });
            texture_index.insert((file, id), scene.textures.len() - 1);
        }
    }

    // Convert every mesh object in parallel.
    let mesh_objects: Vec<(&String, u32, &Object)> = files
        .iter()
        .flat_map(|(f, m)| m.objects.iter().map(move |(id, o)| (f, *id, o)))
        .filter(|(_, _, o)| matches!(o.content, ObjectContent::Mesh { .. }))
        .collect();
    let converted: Vec<((String, u32), Vec<TypedPrimitive>)> = mesh_objects
        .par_iter()
        .map(|(f, id, o)| (((*f).clone(), *id), convert_mesh(&files[*f], o)))
        .collect();

    let mut materials: HashMap<MaterialKindKey, usize> = HashMap::new();
    let mut mesh_index: HashMap<(String, u32), usize> = HashMap::new();
    for (key, prims) in converted {
        let name = files[&key.0].objects[&key.1].name.clone();
        let mut mesh = Mesh { name, primitives: Vec::new() };
        for (mut prim, kind) in prims {
            let mkey = match &kind {
                MaterialKind::Plain => MaterialKindKey::Plain,
                MaterialKind::Colors { transparent } => MaterialKindKey::Colors(*transparent),
                MaterialKind::Texture(t) => MaterialKindKey::Texture(key.0.clone(), *t),
            };
            let mi = match materials.get(&mkey) {
                Some(&mi) => mi,
                None => {
                    let mut mat = Material { name: "3MF".into(), ..Default::default() };
                    match &kind {
                        MaterialKind::Plain => {}
                        MaterialKind::Colors { transparent } => {
                            mat.base_color = [1.0; 4];
                            if *transparent {
                                mat.alpha_mode = AlphaMode::Blend;
                            }
                        }
                        MaterialKind::Texture(t) => {
                            mat.base_color = [1.0; 4];
                            mat.base_color_texture =
                                texture_index.get(&(key.0.clone(), *t)).map(|&i| TextureRef::new(i));
                        }
                    }
                    let mi = scene.add_material(mat);
                    materials.insert(mkey, mi);
                    mi
                }
            };
            prim.material = Some(mi);
            mesh.primitives.push(prim);
        }
        mesh_index.insert(key, scene.meshes.len());
        scene.meshes.push(mesh);
    }

    // Instantiate the build.
    for item in &root.build {
        let file = item.path.clone().unwrap_or_else(|| root_path.clone());
        instantiate(&mut scene, &files, &mesh_index, &file, item, None, 0);
    }
    if scene.roots.is_empty() {
        bail!("the 3MF file has no build items");
    }
    Ok(scene)
}

fn instantiate(
    scene: &mut Scene,
    files: &HashMap<String, ModelFile>,
    meshes: &HashMap<(String, u32), usize>,
    file: &str,
    placement: &Placement,
    parent: Option<usize>,
    depth: usize,
) {
    if depth > 32 {
        return;
    }
    let Some(obj) = files.get(file).and_then(|m| m.objects.get(&placement.object)) else { return };
    let node = scene.add_node(
        Node {
            name: obj.name.clone(),
            transform: Transform::from_matrix(&placement.transform),
            mesh: meshes.get(&(file.to_owned(), placement.object)).copied(),
            ..Default::default()
        },
        parent,
    );
    if let ObjectContent::Components(comps) = &obj.content {
        for c in comps {
            let child_file = c.path.clone().unwrap_or_else(|| file.to_owned());
            instantiate(scene, files, meshes, &child_file, c, Some(node), depth + 1);
        }
    }
}

enum MaterialKind {
    Plain,
    Colors { transparent: bool },
    Texture(u32),
}

#[derive(Hash, PartialEq, Eq)]
enum MaterialKindKey {
    Plain,
    Colors(bool),
    Texture(String, u32),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Slot {
    Plain,
    Colors,
    Texture(u32),
}

/// Splits an object's triangles by property type and builds one primitive each.
/// A primitive and the kind of material it needs.
type TypedPrimitive = (Primitive, MaterialKind);

/// Per slot: corner vertex indices, corner colours, corner UVs.
type SlotCorners = (Vec<u32>, Vec<[u8; 4]>, Vec<[f32; 2]>);

/// Slot of a triangle with its three corner colours or UVs.
type TriangleProps = (Slot, Option<[[u8; 4]; 3]>, Option<[[f32; 2]; 3]>);

fn convert_mesh(file: &ModelFile, obj: &Object) -> Vec<TypedPrimitive> {
    let ObjectContent::Mesh { vertices, triangles } = &obj.content else { return Vec::new() };
    let nv = vertices.len() as u32;
    let mut slots: HashMap<Slot, SlotCorners> = HashMap::new();
    let default_color = [204, 204, 204, 255];

    for t in triangles {
        if t.v.iter().any(|&v| v >= nv) {
            continue;
        }
        let pid = t.pid.or(obj.pid);
        let p1 = t.p[0].or(obj.pindex);
        let ps = [p1, t.p[1].or(p1), t.p[2].or(p1)];
        let (slot, colors, uvs): TriangleProps = match pid {
            Some(pid) if file.base_materials.contains_key(&pid) || file.color_groups.contains_key(&pid) => {
                let group = file.base_materials.get(&pid).or_else(|| file.color_groups.get(&pid)).unwrap();
                let c = |p: Option<u32>| p.and_then(|i| group.get(i as usize).copied()).unwrap_or(default_color);
                // Base materials apply one colour per triangle (p1).
                let cols = if file.base_materials.contains_key(&pid) {
                    let col = c(ps[0]);
                    [col, col, col]
                } else {
                    [c(ps[0]), c(ps[1]), c(ps[2])]
                };
                (Slot::Colors, Some(cols), None)
            }
            Some(pid) if file.tex_groups.contains_key(&pid) => {
                let g = &file.tex_groups[&pid];
                let uv = |p: Option<u32>| p.and_then(|i| g.coords.get(i as usize).copied()).unwrap_or([0.0; 2]);
                (Slot::Texture(g.texture), None, Some([uv(ps[0]), uv(ps[1]), uv(ps[2])]))
            }
            _ => (Slot::Plain, None, None),
        };
        let entry = slots.entry(slot).or_default();
        entry.0.extend_from_slice(&t.v);
        if let Some(c) = colors {
            entry.1.extend_from_slice(&c);
        }
        if let Some(u) = uvs {
            entry.2.extend_from_slice(&u);
        }
    }

    let mut out = Vec::new();
    let mut keys: Vec<Slot> = slots.keys().copied().collect();
    keys.sort_by_key(|s| match s {
        Slot::Plain => (0, 0),
        Slot::Colors => (1, 0),
        Slot::Texture(t) => (2, *t),
    });
    for slot in keys {
        let (indices, colors, uvs) = &slots[&slot];
        let corner_keys: Option<Vec<u32>> = match slot {
            Slot::Plain => None,
            Slot::Colors => Some(colors.iter().map(|c| u32::from_le_bytes(*c)).collect()),
            Slot::Texture(_) => {
                let mut table: HashMap<u64, u32> = HashMap::new();
                Some(
                    uvs.iter()
                        .map(|uv| {
                            let k = (uv[0].to_bits() as u64) << 32 | uv[1].to_bits() as u64;
                            let next = table.len() as u32;
                            *table.entry(k).or_insert(next)
                        })
                        .collect(),
                )
            }
        };
        let s = util::smooth_normals(vertices, indices, corner_keys.as_deref(), 40.0);
        let mut p = Primitive::new(Topology::Triangles);
        let kind = match slot {
            Slot::Plain => MaterialKind::Plain,
            Slot::Colors => {
                p.colors = s.source_corner.iter().map(|&c| colors[c as usize]).collect();
                MaterialKind::Colors { transparent: colors.iter().any(|c| c[3] < 255) }
            }
            Slot::Texture(t) => {
                p.uvs = s.source_corner.iter().map(|&c| uvs[c as usize]).collect();
                MaterialKind::Texture(t)
            }
        };
        p.positions = s.positions;
        p.normals = s.normals;
        p.indices = s.indices;
        out.push((p, kind));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colors_and_transforms() {
        assert_eq!(parse_color("#FF8000"), Some([255, 128, 0, 255]));
        assert_eq!(parse_color("#00000080"), Some([0, 0, 0, 128]));
        assert_eq!(parse_color("red"), None);
        let m = parse_transform(Some("1 0 0 0 1 0 0 0 1 10 20 30"));
        assert_eq!(m.transform_point3(glam::Vec3::ZERO), glam::Vec3::new(10.0, 20.0, 30.0));
    }

    #[test]
    fn parses_model_with_colors() {
        let xml = r##"<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">
 <resources>
  <basematerials id="1"><base name="red" displaycolor="#FF0000" /></basematerials>
  <object id="2" type="model" pid="1" pindex="0">
   <mesh>
    <vertices><vertex x="0" y="0" z="0"/><vertex x="1" y="0" z="0"/><vertex x="0" y="1" z="0"/></vertices>
    <triangles><triangle v1="0" v2="1" v3="2"/></triangles>
   </mesh>
  </object>
 </resources>
 <build><item objectid="2" transform="1 0 0 0 1 0 0 0 1 5 0 0"/></build>
</model>"##;
        let m = parse_model(xml).unwrap();
        assert_eq!(m.unit.as_deref(), Some("mm"));
        assert_eq!(m.build.len(), 1);
        let prims = convert_mesh(&m, &m.objects[&2]);
        assert_eq!(prims.len(), 1);
        assert_eq!(prims[0].0.colors[0], [255, 0, 0, 255]);
        assert_eq!(prims[0].0.indices.len(), 3);
    }
}
