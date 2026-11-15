//! glTF 2.0 (.gltf + .bin, .glb) loader.
//!
//! The `gltf` crate parses the JSON; buffers, images and accessors are read
//! here so that quantised data (KHR_mesh_quantization), sparse accessors and
//! missing textures are handled gracefully.

use super::LoadCtx;
use super::util::{self, ImageSource};
use crate::scene::*;
use anyhow::{Context, Result, anyhow, bail};
use base64::Engine;
use glam::{Mat4, Quat, Vec3};
use gltf::accessor::DataType;
use gltf::mesh::Mode;
use rayon::prelude::*;
use std::path::Path;

pub fn load(ctx: &LoadCtx) -> Result<Scene> {
    let bytes = std::fs::read(ctx.path).with_context(|| format!("cannot read '{}'", ctx.path.display()))?;
    ctx.progress.set(0.05);

    let gltf = match gltf::Gltf::from_slice(&bytes) {
        Ok(g) => g,
        Err(gltf::Error::Validation(errors)) => {
            // Many exporters write slightly invalid files (e.g. missing
            // accessor bounds); load them anyway.
            let only_extensions =
                errors.iter().all(|(_, e)| matches!(e, gltf::json::validation::Error::Unsupported));
            if !only_extensions {
                ctx.warn(format!("The file does not fully follow the glTF specification ({} issues)", errors.len()));
            }
            gltf::Gltf::from_slice_without_validation(&bytes).context("invalid glTF file")?
        }
        Err(e) => return Err(anyhow!(e)).context("invalid glTF file"),
    };
    drop(bytes);
    ctx.progress.check()?;

    for ext in gltf.document.extensions_required() {
        match ext {
            "KHR_draco_mesh_compression" => {
                bail!("This file uses Draco mesh compression, which is not supported yet.")
            }
            "EXT_meshopt_compression" => {
                bail!("This file uses meshoptimizer compression, which is not supported yet.")
            }
            "KHR_texture_basisu" => ctx.warn("KTX2 (Basis Universal) textures are not supported; textures are missing"),
            "KHR_mesh_quantization"
            | "KHR_texture_transform"
            | "KHR_materials_unlit"
            | "KHR_materials_emissive_strength"
            | "KHR_materials_pbrSpecularGlossiness"
            | "EXT_texture_webp" => {}
            other => ctx.warn(format!("Required extension '{other}' is not supported; the model may look wrong")),
        }
    }

    let doc = &gltf.document;
    let mut blob = gltf.blob.clone();
    let mut buffers: Vec<Vec<u8>> = Vec::new();
    for b in doc.buffers() {
        let data = match b.source() {
            gltf::buffer::Source::Bin => blob.take().ok_or_else(|| anyhow!("the GLB binary chunk is missing"))?,
            gltf::buffer::Source::Uri(uri) => load_uri(uri, ctx.dir)
                .with_context(|| format!("cannot load buffer '{}'", util::percent_decode(uri)))?,
        };
        if data.len() < b.length() {
            bail!("buffer {} is shorter than declared ({} < {} bytes)", b.index(), data.len(), b.length());
        }
        buffers.push(data);
    }
    ctx.progress.set(0.15);
    ctx.progress.set_stage("Decoding textures and meshes");

    // Images and meshes are independent: process both at once.
    let (images, meshes) = rayon::join(|| decode_images(doc, &buffers, ctx), || read_meshes(doc, &buffers, ctx));
    let meshes = meshes?;
    ctx.progress.check()?;
    ctx.progress.set(0.85);

    let mut scene = Scene { up_axis: UpAxis::Y, unit: Some("m".into()), ..Default::default() };

    // Images -> scene images (failed decodes are skipped).
    let mut image_map = vec![None; images.len()];
    for (i, img) in images.into_iter().enumerate() {
        if let Some(img) = img {
            image_map[i] = Some(scene.images.len());
            scene.images.push(img);
        }
    }

    // Textures.
    let mut texture_map = Vec::new();
    for tex in doc.textures() {
        let image = texture_image(&tex).and_then(|i| image_map.get(i).copied().flatten());
        texture_map.push(image.map(|image| {
            scene.textures.push(Texture { image, sampler: convert_sampler(&tex.sampler()) });
            scene.textures.len() - 1
        }));
    }

    for m in doc.materials() {
        let mat = convert_material(&m, &texture_map, ctx);
        scene.materials.push(mat);
    }
    scene.meshes = meshes;

    // Nodes.
    for node in doc.nodes() {
        let (t, r, s) = node.transform().decomposed();
        let rotation = Quat::from_array(r);
        let weights = node
            .weights()
            .or_else(|| node.mesh().and_then(|m| m.weights()))
            .map(|w| w.to_vec())
            .unwrap_or_default();
        scene.nodes.push(Node {
            name: node.name().unwrap_or_default().to_owned(),
            parent: None,
            children: node.children().map(|c| c.index()).collect(),
            transform: Transform {
                translation: Vec3::from_array(t),
                rotation: if rotation.is_finite() && rotation.length_squared() > 0.0 {
                    rotation.normalize()
                } else {
                    Quat::IDENTITY
                },
                scale: Vec3::from_array(s),
            },
            mesh: node.mesh().map(|m| m.index()),
            skin: node.skin().map(|s| s.index()),
            weights,
        });
    }
    let node_count = scene.nodes.len();
    for i in 0..node_count {
        let children = scene.nodes[i].children.clone();
        for c in children {
            if c < node_count && scene.nodes[c].parent.is_none() && c != i {
                scene.nodes[c].parent = Some(i);
            }
        }
    }
    for n in &mut scene.nodes {
        n.children.retain(|&c| c < node_count);
    }
    scene.roots = match doc.default_scene().or_else(|| doc.scenes().next()) {
        Some(s) => s.nodes().map(|n| n.index()).collect(),
        None => (0..node_count).filter(|&i| scene.nodes[i].parent.is_none()).collect(),
    };

    // Skins.
    for skin in doc.skins() {
        let joints: Vec<usize> = skin.joints().map(|j| j.index()).collect();
        let inverse_bind = match skin.inverse_bind_matrices() {
            Some(acc) => {
                let (vals, comps) = read_floats(&acc, &buffers)?;
                if comps != 16 {
                    bail!("invalid inverse bind matrices");
                }
                vals.chunks_exact(16).map(|c| Mat4::from_cols_slice(c)).collect()
            }
            None => vec![Mat4::IDENTITY; joints.len()],
        };
        scene.skins.push(Skin { joints, inverse_bind });
    }

    // Animations.
    for (ai, anim) in doc.animations().enumerate() {
        let mut out = Animation {
            name: anim.name().map(str::to_owned).unwrap_or_else(|| format!("Animation {}", ai + 1)),
            ..Default::default()
        };
        for ch in anim.channels() {
            if let Err(e) = convert_channel(&ch, &buffers, &mut out) {
                ctx.warn(format!("Animation '{}': {e:#}", out.name));
            }
        }
        scene.animations.push(out);
    }
    ctx.progress.set(0.95);
    Ok(scene)
}

fn load_uri(uri: &str, dir: &Path) -> Result<Vec<u8>> {
    if let Some(rest) = uri.strip_prefix("data:") {
        let (meta, data) = rest.split_once(',').ok_or_else(|| anyhow!("malformed data URI"))?;
        if meta.ends_with(";base64") {
            return base64::engine::general_purpose::STANDARD
                .decode(data.trim())
                .context("invalid base64 data URI");
        }
        return Ok(util::percent_decode(data).into_bytes());
    }
    let path = dir.join(util::percent_decode(uri));
    std::fs::read(&path).with_context(|| format!("cannot read '{}'", path.display()))
}

fn texture_image(tex: &gltf::Texture) -> Option<usize> {
    if let Some(src) = tex
        .extension_value("EXT_texture_webp")
        .and_then(|v| v.get("source"))
        .and_then(|s| s.as_u64())
    {
        return Some(src as usize);
    }
    tex.source().map(|i| i.index())
}

fn decode_images(doc: &gltf::Document, buffers: &[Vec<u8>], ctx: &LoadCtx) -> Vec<Option<Image>> {
    let sources: Vec<(String, ImageSource)> = doc
        .images()
        .map(|img| {
            let fallback = format!("image {}", img.index());
            match img.source() {
                gltf::image::Source::View { view, .. } => {
                    let data = buffers
                        .get(view.buffer().index())
                        .and_then(|b| b.get(view.offset()..view.offset() + view.length()))
                        .map(|s| s.to_vec())
                        .unwrap_or_default();
                    (img.name().map(str::to_owned).unwrap_or(fallback), ImageSource::Bytes(data))
                }
                gltf::image::Source::Uri { uri, .. } => {
                    if uri.starts_with("data:") {
                        let data = load_uri(uri, ctx.dir).unwrap_or_default();
                        (img.name().map(str::to_owned).unwrap_or(fallback), ImageSource::Bytes(data))
                    } else {
                        let rel = util::percent_decode(uri);
                        let path = ctx.dir.join(&rel);
                        let path = if path.is_file() {
                            path
                        } else {
                            util::resolve_texture_path(ctx.dir, &[&rel]).unwrap_or(path)
                        };
                        (rel, ImageSource::File(path))
                    }
                }
            }
        })
        .collect();
    util::decode_images(sources, ctx.warnings())
}

fn convert_sampler(s: &gltf::texture::Sampler) -> Sampler {
    use gltf::texture::{MagFilter, MinFilter, WrappingMode};
    let wrap = |w: WrappingMode| match w {
        WrappingMode::Repeat => Wrap::Repeat,
        WrappingMode::MirroredRepeat => Wrap::Mirror,
        WrappingMode::ClampToEdge => Wrap::Clamp,
    };
    Sampler {
        wrap_u: wrap(s.wrap_s()),
        wrap_v: wrap(s.wrap_t()),
        mag_nearest: matches!(s.mag_filter(), Some(MagFilter::Nearest)),
        min_nearest: matches!(
            s.min_filter(),
            Some(MinFilter::Nearest | MinFilter::NearestMipmapNearest | MinFilter::NearestMipmapLinear)
        ),
    }
}

fn info_ref(info: &gltf::texture::Info, texture_map: &[Option<usize>], ctx: &LoadCtx, what: &str) -> Option<TextureRef> {
    let transform = info.texture_transform();
    let tex_coord = transform.as_ref().and_then(|t| t.tex_coord()).unwrap_or(info.tex_coord());
    if tex_coord != 0 {
        ctx.warn(format!("The {what} texture uses a second UV set, which is not supported"));
        return None;
    }
    let texture = texture_map.get(info.texture().index()).copied().flatten()?;
    let transform = transform.map(|t| {
        let [ox, oy] = t.offset();
        let [sx, sy] = t.scale();
        let (sin, cos) = t.rotation().sin_cos();
        [cos * sx, sin * sy, ox, -sin * sx, cos * sy, oy]
    });
    Some(TextureRef { texture, transform })
}

fn convert_material(m: &gltf::Material, texture_map: &[Option<usize>], ctx: &LoadCtx) -> Material {
    let pbr = m.pbr_metallic_roughness();
    let mut mat = Material {
        name: m.name().unwrap_or_default().to_owned(),
        base_color: pbr.base_color_factor(),
        metallic: pbr.metallic_factor(),
        roughness: pbr.roughness_factor(),
        double_sided: m.double_sided(),
        unlit: m.unlit(),
        alpha_cutoff: m.alpha_cutoff().unwrap_or(0.5),
        alpha_mode: match m.alpha_mode() {
            gltf::material::AlphaMode::Opaque => AlphaMode::Opaque,
            gltf::material::AlphaMode::Mask => AlphaMode::Mask,
            gltf::material::AlphaMode::Blend => AlphaMode::Blend,
        },
        ..Default::default()
    };
    mat.base_color_texture = pbr.base_color_texture().and_then(|i| info_ref(&i, texture_map, ctx, "base colour"));
    if let Some(mr) = pbr.metallic_roughness_texture().and_then(|i| info_ref(&i, texture_map, ctx, "metallic-roughness")) {
        mat.metallic_texture = Some(mr);
        mat.metallic_channel = 2;
        mat.roughness_texture = Some(mr);
        mat.roughness_channel = 1;
    }
    if let Some(sg) = m.pbr_specular_glossiness() {
        mat.base_color = sg.diffuse_factor();
        mat.base_color_texture = sg.diffuse_texture().and_then(|i| info_ref(&i, texture_map, ctx, "diffuse"));
        mat.metallic = 0.0;
        mat.metallic_texture = None;
        mat.glossiness_workflow = true;
        mat.roughness = sg.glossiness_factor();
        mat.roughness_texture =
            sg.specular_glossiness_texture().and_then(|i| info_ref(&i, texture_map, ctx, "specular-glossiness"));
        mat.roughness_channel = 3;
    }
    if let Some(n) = m.normal_texture()
        && n.tex_coord() == 0 {
            mat.normal_texture = texture_map.get(n.texture().index()).copied().flatten().map(TextureRef::new);
            mat.normal_scale = n.scale();
        }
    if let Some(o) = m.occlusion_texture() {
        // Occlusion often uses a second UV set (lightmap UVs): skip silently then.
        if o.tex_coord() == 0 {
            mat.occlusion_texture = texture_map.get(o.texture().index()).copied().flatten().map(TextureRef::new);
            mat.occlusion_strength = o.strength();
        }
    }
    let strength = m.emissive_strength().unwrap_or(1.0);
    let e = m.emissive_factor();
    mat.emissive = [e[0] * strength, e[1] * strength, e[2] * strength];
    mat.emissive_texture = m.emissive_texture().and_then(|i| info_ref(&i, texture_map, ctx, "emissive"));
    mat
}

fn read_meshes(doc: &gltf::Document, buffers: &[Vec<u8>], ctx: &LoadCtx) -> Result<Vec<Mesh>> {
    let meshes: Vec<gltf::Mesh> = doc.meshes().collect();
    meshes
        .par_iter()
        .map(|mesh| {
            ctx.progress.check()?;
            let mut out = Mesh { name: mesh.name().unwrap_or_default().to_owned(), primitives: Vec::new() };
            for prim in mesh.primitives() {
                match read_primitive(&prim, buffers) {
                    Ok(Some(p)) => out.primitives.push(p),
                    Ok(None) => {}
                    Err(e) => ctx.warn(format!("Mesh '{}': {e:#}", out.name)),
                }
            }
            Ok(out)
        })
        .collect()
}

fn read_primitive(prim: &gltf::Primitive, buffers: &[Vec<u8>]) -> Result<Option<Primitive>> {
    use gltf::Semantic;
    let Some(pos_acc) = prim.get(&Semantic::Positions) else { return Ok(None) };
    let topology = match prim.mode() {
        Mode::Triangles | Mode::TriangleStrip | Mode::TriangleFan => Topology::Triangles,
        Mode::Lines | Mode::LineStrip | Mode::LineLoop => Topology::Lines,
        Mode::Points => Topology::Points,
    };
    let mut p = Primitive::new(topology);
    p.material = prim.material().index();
    p.positions = to_vec3(read_floats(&pos_acc, buffers)?);
    let n = p.positions.len();

    if let Some(acc) = prim.get(&Semantic::Normals) {
        p.normals = to_vec3(read_floats(&acc, buffers)?);
    }
    if let Some(acc) = prim.get(&Semantic::TexCoords(0)) {
        let (v, c) = read_floats(&acc, buffers)?;
        if c == 2 {
            p.uvs = v.chunks_exact(2).map(|c| [c[0], c[1]]).collect();
        }
    }
    if let Some(acc) = prim.get(&Semantic::Colors(0)) {
        let (v, c) = read_floats(&acc, buffers)?;
        if c == 3 || c == 4 {
            p.colors = v
                .chunks_exact(c)
                .map(|x| encode_vertex_color([x[0], x[1], x[2], if c == 4 { x[3] } else { 1.0 }]))
                .collect();
        }
    }
    if let (Some(j), Some(w)) = (prim.get(&Semantic::Joints(0)), prim.get(&Semantic::Weights(0))) {
        let (jv, jc) = read_floats(&j, buffers)?;
        let (wv, wc) = read_floats(&w, buffers)?;
        if jc == 4 && wc == 4 {
            p.joints = jv
                .chunks_exact(4)
                .map(|c| [c[0] as u16, c[1] as u16, c[2] as u16, c[3] as u16])
                .collect();
            p.weights = wv
                .chunks_exact(4)
                .map(|c| {
                    let sum: f32 = c.iter().sum();
                    if sum > 0.0 { [c[0] / sum, c[1] / sum, c[2] / sum, c[3] / sum] } else { [1.0, 0.0, 0.0, 0.0] }
                })
                .collect();
        }
    }
    for target in prim.morph_targets() {
        let positions = match target.positions() {
            Some(acc) => to_vec3(read_floats(&acc, buffers)?),
            None => Vec::new(),
        };
        let normals = match target.normals() {
            Some(acc) => to_vec3(read_floats(&acc, buffers)?),
            None => Vec::new(),
        };
        p.targets.push(MorphTarget { positions, normals });
    }

    let raw_indices: Vec<u32> = match prim.indices() {
        Some(acc) => read_indices(&acc, buffers)?,
        None => (0..n as u32).collect(),
    };
    p.indices = match prim.mode() {
        Mode::Triangles | Mode::Lines => raw_indices,
        Mode::TriangleStrip => {
            let mut out = Vec::with_capacity(raw_indices.len().saturating_sub(2) * 3);
            for i in 0..raw_indices.len().saturating_sub(2) {
                let (a, b, c) = (raw_indices[i], raw_indices[i + 1], raw_indices[i + 2]);
                if i % 2 == 0 { out.extend([a, b, c]) } else { out.extend([b, a, c]) }
            }
            out
        }
        Mode::TriangleFan => {
            let mut out = Vec::new();
            for i in 1..raw_indices.len().saturating_sub(1) {
                out.extend([raw_indices[0], raw_indices[i], raw_indices[i + 1]]);
            }
            out
        }
        Mode::LineStrip | Mode::LineLoop => {
            let mut out = Vec::new();
            for w in raw_indices.windows(2) {
                out.extend([w[0], w[1]]);
            }
            if prim.mode() == Mode::LineLoop && raw_indices.len() > 2 {
                out.extend([raw_indices[raw_indices.len() - 1], raw_indices[0]]);
            }
            out
        }
        Mode::Points => {
            if prim.indices().is_some() {
                unweld(&mut p, &raw_indices, 1);
            }
            Vec::new()
        }
    };

    // The spec asks for flat normals when none are given.
    if topology == Topology::Triangles && p.normals.len() != n {
        let indices = std::mem::take(&mut p.indices);
        unweld(&mut p, &indices, 3);
        p.normals = p
            .positions
            .chunks_exact(3)
            .flat_map(|t| {
                let (a, b, c) = (Vec3::from_array(t[0]), Vec3::from_array(t[1]), Vec3::from_array(t[2]));
                let nrm = (b - a).cross(c - a).normalize_or(Vec3::Y).to_array();
                [nrm, nrm, nrm]
            })
            .collect();
        p.indices = (0..p.positions.len() as u32).collect();
    }
    Ok(Some(p))
}

/// Expands every per-vertex attribute through `indices` (one vertex per index),
/// dropping whole elements (`arity` indices) that reference missing vertices.
fn unweld(p: &mut Primitive, indices: &[u32], arity: usize) {
    fn expand<T: Copy>(v: &mut Vec<T>, idx: &[u32], n: usize) {
        if v.len() == n {
            *v = idx.iter().map(|&i| v[i as usize]).collect();
        }
    }
    let n = p.positions.len();
    let valid: Vec<u32> = indices
        .chunks_exact(arity)
        .filter(|c| c.iter().all(|&i| (i as usize) < n))
        .flatten()
        .copied()
        .collect();
    let valid = &valid[..];
    expand(&mut p.normals, valid, n);
    expand(&mut p.uvs, valid, n);
    expand(&mut p.colors, valid, n);
    expand(&mut p.joints, valid, n);
    expand(&mut p.weights, valid, n);
    for t in &mut p.targets {
        expand(&mut t.positions, valid, n);
        expand(&mut t.normals, valid, n);
    }
    expand(&mut p.positions, valid, n);
}

fn to_vec3((v, c): (Vec<f32>, usize)) -> Vec<[f32; 3]> {
    if c < 3 {
        return Vec::new();
    }
    v.chunks_exact(c).map(|x| [x[0], x[1], x[2]]).collect()
}

fn component(buf: &[u8], off: usize, dt: DataType, normalized: bool) -> f32 {
    let b = &buf[off..];
    match dt {
        DataType::F32 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        DataType::U8 => {
            let v = b[0] as f32;
            if normalized { v / 255.0 } else { v }
        }
        DataType::I8 => {
            let v = b[0] as i8 as f32;
            if normalized { (v / 127.0).max(-1.0) } else { v }
        }
        DataType::U16 => {
            let v = u16::from_le_bytes([b[0], b[1]]) as f32;
            if normalized { v / 65535.0 } else { v }
        }
        DataType::I16 => {
            let v = i16::from_le_bytes([b[0], b[1]]) as f32;
            if normalized { (v / 32767.0).max(-1.0) } else { v }
        }
        DataType::U32 => u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32,
    }
}

/// Reads any accessor as floats (normalised integers mapped to [0,1] / [-1,1]).
/// Returns the values and the number of components per element.
fn read_floats(acc: &gltf::Accessor, buffers: &[Vec<u8>]) -> Result<(Vec<f32>, usize)> {
    let comps = acc.dimensions().multiplicity();
    let count = acc.count();
    let dt = acc.data_type();
    let csize = dt.size();
    let normalized = acc.normalized();
    let mut out = vec![0f32; count * comps];

    if let Some(view) = acc.view() {
        let buf = buffers.get(view.buffer().index()).ok_or_else(|| anyhow!("missing buffer"))?;
        let elem = csize * comps;
        let stride = view.stride().unwrap_or(elem).max(elem);
        let base = view.offset() + acc.offset();
        let view_end = (view.offset() + view.length()).min(buf.len());
        if count > 0 && base + stride * (count - 1) + elem > view_end {
            bail!("accessor {} reads past the end of its buffer", acc.index());
        }
        if dt == DataType::F32 && stride == elem {
            let bytes = &buf[base..base + elem * count];
            for (o, c) in out.iter_mut().zip(bytes.chunks_exact(4)) {
                *o = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        } else {
            for i in 0..count {
                for c in 0..comps {
                    out[i * comps + c] = component(buf, base + i * stride + c * csize, dt, normalized);
                }
            }
        }
    }

    if let Some(sparse) = acc.sparse() {
        let idx = sparse.indices();
        let vals = sparse.values();
        let ibuf = buffers.get(idx.view().buffer().index()).ok_or_else(|| anyhow!("missing buffer"))?;
        let vbuf = buffers.get(vals.view().buffer().index()).ok_or_else(|| anyhow!("missing buffer"))?;
        let ioff = idx.view().offset() + idx.offset();
        let voff = vals.view().offset() + vals.offset();
        let isize = match idx.index_type() {
            gltf::accessor::sparse::IndexType::U8 => 1,
            gltf::accessor::sparse::IndexType::U16 => 2,
            gltf::accessor::sparse::IndexType::U32 => 4,
        };
        let n = sparse.count();
        if ioff + n * isize > ibuf.len() || voff + n * comps * csize > vbuf.len() {
            bail!("sparse accessor {} is out of bounds", acc.index());
        }
        for k in 0..n {
            let o = ioff + k * isize;
            let target = match isize {
                1 => ibuf[o] as usize,
                2 => u16::from_le_bytes([ibuf[o], ibuf[o + 1]]) as usize,
                _ => u32::from_le_bytes([ibuf[o], ibuf[o + 1], ibuf[o + 2], ibuf[o + 3]]) as usize,
            };
            if target >= count {
                continue;
            }
            for c in 0..comps {
                out[target * comps + c] = component(vbuf, voff + (k * comps + c) * csize, dt, normalized);
            }
        }
    }
    Ok((out, comps))
}

fn read_indices(acc: &gltf::Accessor, buffers: &[Vec<u8>]) -> Result<Vec<u32>> {
    // u8/u16 indices are exact in f32; u32 indices are read directly.
    let direct = acc.data_type() == DataType::U32 && acc.sparse().is_none() && acc.view().is_some();
    if !direct {
        let (vals, _) = read_floats(acc, buffers)?;
        return Ok(vals.into_iter().map(|v| v as u32).collect());
    }
    let view = acc.view().ok_or_else(|| anyhow!("index accessor without buffer view"))?;
    let buf = buffers.get(view.buffer().index()).ok_or_else(|| anyhow!("missing buffer"))?;
    let stride = view.stride().unwrap_or(4).max(4);
    let base = view.offset() + acc.offset();
    let view_end = (view.offset() + view.length()).min(buf.len());
    if acc.count() > 0 && base + stride * (acc.count() - 1) + 4 > view_end {
        bail!("index accessor {} reads past the end of its buffer", acc.index());
    }
    Ok((0..acc.count())
        .map(|i| {
            let o = base + i * stride;
            u32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]])
        })
        .collect())
}

fn convert_channel(ch: &gltf::animation::Channel, buffers: &[Vec<u8>], anim: &mut Animation) -> Result<()> {
    use gltf::animation::{Interpolation as GI, Property as GP};
    let target = ch.target();
    let node = target.node().index();
    let sampler = ch.sampler();
    let (times, _) = read_floats(&sampler.input(), buffers)?;
    let (values, _) = read_floats(&sampler.output(), buffers)?;
    let interpolation = match sampler.interpolation() {
        GI::Linear => Interpolation::Linear,
        GI::Step => Interpolation::Step,
        GI::CubicSpline => Interpolation::CubicSpline,
    };
    let keys = times.len();
    if keys == 0 {
        return Ok(());
    }
    let per_key = if interpolation == Interpolation::CubicSpline { 3 } else { 1 };
    let property = match target.property() {
        GP::Translation => Property::Translation,
        GP::Rotation => Property::Rotation,
        GP::Scale => Property::Scale,
        GP::MorphTargetWeights => {
            // One channel per target with scalar values.
            let targets = values.len() / (keys * per_key);
            if targets == 0 {
                bail!("empty morph weight channel");
            }
            for t in 0..targets {
                let mut v = Vec::with_capacity(keys * per_key);
                for k in 0..keys * per_key {
                    v.push(values[k * targets + t]);
                }
                anim.channels.push(Channel {
                    node,
                    property: Property::MorphWeight(t as u32),
                    interpolation,
                    times: times.clone(),
                    values: v,
                });
            }
            return Ok(());
        }
    };
    if values.len() < keys * per_key * property.components() {
        bail!("animation sampler output is too short");
    }
    anim.channels.push(Channel { node, property, interpolation, times, values });
    Ok(())
}
