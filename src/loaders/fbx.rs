//! FBX and OBJ (+MTL) loader based on ufbx.
//!
//! ufbx normalises the scene (Y-up, geometry transforms baked, pivots
//! resolved); animations are baked to TRS keys, blend shapes become morph
//! targets and skin clusters become joints.

use super::util::{self, ImageSource};
use super::{Format, LoadCtx};
use crate::scene::*;
use anyhow::{Result, anyhow};
use glam::{Mat4, Quat, Vec3, Vec4};
use rayon::prelude::*;
use std::collections::HashMap;

fn fbx_error(e: &ufbx::Error) -> String {
    let info = e.info();
    if info.is_empty() { e.description.to_string() } else { format!("{} ({info})", e.description) }
}

fn mat4(m: &ufbx::Matrix) -> Mat4 {
    Mat4::from_cols(
        Vec4::new(m.m00 as f32, m.m10 as f32, m.m20 as f32, 0.0),
        Vec4::new(m.m01 as f32, m.m11 as f32, m.m21 as f32, 0.0),
        Vec4::new(m.m02 as f32, m.m12 as f32, m.m22 as f32, 0.0),
        Vec4::new(m.m03 as f32, m.m13 as f32, m.m23 as f32, 1.0),
    )
}

fn vec3(v: ufbx::Vec3) -> Vec3 {
    Vec3::new(v.x as f32, v.y as f32, v.z as f32)
}

fn quat(q: ufbx::Quat) -> Quat {
    let q = Quat::from_xyzw(q.x as f32, q.y as f32, q.z as f32, q.w as f32);
    if q.is_finite() && q.length_squared() > 0.0 { q.normalize() } else { Quat::IDENTITY }
}

fn transform(t: &ufbx::Transform) -> Transform {
    Transform { translation: vec3(t.translation), rotation: quat(t.rotation), scale: vec3(t.scale) }
}

pub fn load(ctx: &LoadCtx, format: Format) -> Result<Scene> {
    let path = ctx.path.to_str().ok_or_else(|| anyhow!("the file path is not valid Unicode"))?;
    let opts = ufbx::LoadOpts {
        load_external_files: true,
        ignore_missing_external_files: true,
        // Missing normals are rebuilt later with a crease angle (smooth
        // surfaces, sharp edges) instead of ufbx's flat normals.
        generate_missing_normals: false,
        clean_skin_weights: true,
        obj_search_mtl_by_filename: true,
        target_axes: ufbx::CoordinateAxes {
            right: ufbx::CoordinateAxis::PositiveX,
            up: ufbx::CoordinateAxis::PositiveY,
            front: ufbx::CoordinateAxis::PositiveZ,
        },
        space_conversion: ufbx::SpaceConversion::TransformRoot,
        geometry_transform_handling: ufbx::GeometryTransformHandling::ModifyGeometry,
        inherit_mode_handling: ufbx::InheritModeHandling::Compensate,
        ..Default::default()
    };
    let root = ufbx::load_file(path, opts).map_err(|e| anyhow!("{}", fbx_error(&e)))?;
    ctx.progress.check()?;
    ctx.progress.set(0.3);
    ctx.progress.set_stage("Converting meshes and textures");

    let fbx: &ufbx::Scene = &root;
    let mut scene = Scene { up_axis: UpAxis::Y, ..Default::default() };
    scene.unit = unit_name(fbx.settings.unit_meters);
    if format == Format::Obj {
        scene.unit = None;
    }

    // Textures and images (decoded in parallel with mesh conversion).
    let (texture_slots, image_sources) = collect_textures(fbx, ctx);
    let (images, meshes) = rayon::join(
        || util::decode_images(image_sources, ctx.warnings()),
        || convert_meshes(&root, ctx),
    );
    let meshes = meshes?;
    ctx.progress.set(0.75);

    let mut image_map = vec![None; images.len()];
    for (i, img) in images.into_iter().enumerate() {
        if let Some(img) = img {
            image_map[i] = Some(scene.images.len());
            scene.images.push(img);
        }
    }
    // ufbx texture typed_id -> scene texture.
    let mut texture_map: HashMap<u32, TextureRef> = HashMap::new();
    for slot in texture_slots {
        if let Some(image) = image_map.get(slot.image).copied().flatten() {
            scene.textures.push(Texture { image, sampler: slot.sampler });
            texture_map.insert(slot.typed_id, TextureRef { texture: scene.textures.len() - 1, transform: slot.transform });
        }
    }

    // Materials: ufbx material typed_id == index in fbx.materials.
    for m in fbx.materials.iter() {
        scene.materials.push(convert_material(m, &texture_map));
    }

    // Skins: one per skin deformer.
    let mut skin_of_deformer: HashMap<u32, usize> = HashMap::new();
    for deformer in fbx.skin_deformers.iter() {
        let mut skin = Skin::default();
        for cluster in deformer.clusters.iter() {
            let joint = cluster.bone_node.as_ref().map(|n| n.element.typed_id as usize).unwrap_or(0);
            skin.joints.push(joint);
            skin.inverse_bind.push(mat4(&cluster.geometry_to_bone));
        }
        skin_of_deformer.insert(deformer.element.typed_id, scene.skins.len());
        scene.skins.push(skin);
    }

    // Meshes (same order as fbx.meshes).
    let mut blend_targets: HashMap<u32, Vec<(usize, u32)>> = HashMap::new(); // blend channel element_id -> (mesh, target)
    let mut mesh_default_weights: Vec<Vec<f32>> = Vec::new();
    for (mi, converted) in meshes.into_iter().enumerate() {
        for (element_id, target) in &converted.channel_targets {
            blend_targets.entry(*element_id).or_default().push((mi, *target));
        }
        mesh_default_weights.push(converted.default_weights);
        scene.meshes.push(converted.mesh);
    }

    // Nodes: our index == ufbx typed_id.
    for node in fbx.nodes.iter() {
        let mesh = node.mesh.as_ref().filter(|_| node.visible).map(|m| m.element.typed_id as usize);
        let skin = node
            .mesh
            .as_ref()
            .and_then(|m| m.skin_deformers.iter().next())
            .and_then(|d| skin_of_deformer.get(&d.element.typed_id).copied());
        scene.nodes.push(Node {
            name: node.element.name.to_string(),
            parent: node.parent.as_ref().map(|p| p.element.typed_id as usize),
            children: node.children.iter().map(|c| c.element.typed_id as usize).collect(),
            transform: transform(&node.local_transform),
            mesh,
            skin,
            weights: mesh.map(|m| mesh_default_weights[m].clone()).unwrap_or_default(),
        });
    }
    scene.roots = vec![fbx.root_node.element.typed_id as usize];

    // Animations: bake every stack to linear TRS keys.
    for stack in fbx.anim_stacks.iter() {
        ctx.progress.check()?;
        let opts = ufbx::BakeOpts { trim_start_time: true, ..Default::default() };
        let baked = match ufbx::bake_anim(fbx, &stack.anim, opts) {
            Ok(b) => b,
            Err(e) => {
                ctx.warn(format!("Animation '{}' could not be read: {}", stack.element.name, fbx_error(&e)));
                continue;
            }
        };
        let mut anim = Animation {
            name: if stack.element.name.is_empty() {
                format!("Animation {}", scene.animations.len() + 1)
            } else {
                stack.element.name.to_string()
            },
            duration: baked.playback_duration as f32,
            channels: Vec::new(),
        };
        for bn in baked.nodes.iter() {
            let node = bn.typed_id as usize;
            let Some(rest) = scene.nodes.get(node).map(|n| n.transform) else { continue };
            push_vec3_channel(&mut anim, node, Property::Translation, &bn.translation_keys, rest.translation);
            push_vec3_channel(&mut anim, node, Property::Scale, &bn.scale_keys, rest.scale);
            if !(bn.rotation_keys.len() == 1 && quat(bn.rotation_keys[0].value).dot(rest.rotation).abs() > 0.99999) {
                let mut times = Vec::with_capacity(bn.rotation_keys.len());
                let mut values = Vec::with_capacity(bn.rotation_keys.len() * 4);
                for k in bn.rotation_keys.iter() {
                    times.push(k.time as f32);
                    values.extend(quat(k.value).to_array());
                }
                if !times.is_empty() {
                    anim.channels.push(Channel {
                        node,
                        property: Property::Rotation,
                        interpolation: Interpolation::Linear,
                        times,
                        values,
                    });
                }
            }
        }
        for be in baked.elements.iter() {
            let Some(targets) = blend_targets.get(&be.element_id) else { continue };
            let Some(prop) = be.props.iter().find(|p| p.name == "DeformPercent") else { continue };
            let times: Vec<f32> = prop.keys.iter().map(|k| k.time as f32).collect();
            let values: Vec<f32> = prop.keys.iter().map(|k| k.value.x as f32 / 100.0).collect();
            for &(mesh, target) in targets {
                for (ni, n) in scene.nodes.iter().enumerate() {
                    if n.mesh == Some(mesh) {
                        anim.channels.push(Channel {
                            node: ni,
                            property: Property::MorphWeight(target),
                            interpolation: Interpolation::Linear,
                            times: times.clone(),
                            values: values.clone(),
                        });
                    }
                }
            }
        }
        if !anim.channels.is_empty() {
            scene.animations.push(anim);
        }
    }
    ctx.progress.set(0.95);
    Ok(scene)
}

fn unit_name(meters: f64) -> Option<String> {
    let known = [(0.001, "mm"), (0.01, "cm"), (0.1, "dm"), (1.0, "m"), (0.0254, "in"), (0.3048, "ft")];
    known
        .iter()
        .find(|(m, _)| (meters - m).abs() < m * 1e-3)
        .map(|(_, n)| n.to_string())
}

fn push_vec3_channel(anim: &mut Animation, node: usize, property: Property, keys: &[ufbx::BakedVec3], rest: Vec3) {
    if keys.is_empty() || (keys.len() == 1 && vec3(keys[0].value).abs_diff_eq(rest, 1e-5 * rest.length().max(1.0))) {
        return;
    }
    let mut times = Vec::with_capacity(keys.len());
    let mut values = Vec::with_capacity(keys.len() * 3);
    for k in keys {
        times.push(k.time as f32);
        values.extend(vec3(k.value).to_array());
    }
    anim.channels.push(Channel { node, property, interpolation: Interpolation::Linear, times, values });
}

struct TextureSlot {
    typed_id: u32,
    image: usize,
    sampler: Sampler,
    transform: Option<UvTransform>,
}

/// File texture behind a (possibly layered) texture.
fn file_texture(tex: &ufbx::Texture) -> Option<&ufbx::Texture> {
    if tex.type_ == ufbx::TextureType::File {
        Some(tex)
    } else {
        tex.file_textures.iter().next().map(|t| &**t)
    }
}

fn collect_textures(fbx: &ufbx::Scene, ctx: &LoadCtx) -> (Vec<TextureSlot>, Vec<(String, ImageSource)>) {
    let mut slots = Vec::new();
    let mut sources: Vec<(String, ImageSource)> = Vec::new();
    let mut by_key: HashMap<String, usize> = HashMap::new();
    for tex in fbx.textures.iter() {
        let Some(file) = file_texture(tex) else { continue };
        let name = [&file.relative_filename, &file.filename, &file.absolute_filename]
            .iter()
            .find(|s| !s.is_empty())
            .map(|s| s.to_string())
            .unwrap_or_else(|| file.element.name.to_string());
        let image = if !file.content.is_empty() {
            let key = format!("embedded:{}:{}", file.file_index, file.content.len());
            *by_key.entry(key).or_insert_with(|| {
                sources.push((name.clone(), ImageSource::Bytes(file.content.to_vec())));
                sources.len() - 1
            })
        } else {
            let candidates = [&*file.absolute_filename, &*file.relative_filename, &*file.filename];
            match util::resolve_texture_path(ctx.dir, &candidates) {
                Some(path) => {
                    let key = path.to_string_lossy().into_owned();
                    *by_key.entry(key).or_insert_with(|| {
                        sources.push((name.clone(), ImageSource::File(path)));
                        sources.len() - 1
                    })
                }
                None => {
                    if !name.is_empty() {
                        ctx.warn(format!("Texture '{name}' was not found"));
                    }
                    continue;
                }
            }
        };
        let wrap = |w: ufbx::WrapMode| match w {
            ufbx::WrapMode::Repeat => Wrap::Repeat,
            ufbx::WrapMode::Clamp => Wrap::Clamp,
        };
        let transform = tex.has_uv_transform.then(|| {
            // Texture matrix in FBX UV space (V up), adapted to our flipped V.
            let m = &tex.uv_to_texture;
            [
                m.m00 as f32,
                -(m.m01 as f32),
                (m.m01 + m.m03) as f32,
                -(m.m10 as f32),
                m.m11 as f32,
                (1.0 - m.m11 - m.m13) as f32,
            ]
        });
        slots.push(TextureSlot {
            typed_id: tex.element.typed_id,
            image,
            sampler: Sampler { wrap_u: wrap(tex.wrap_u), wrap_v: wrap(tex.wrap_v), ..Default::default() },
            transform,
        });
    }
    (slots, sources)
}

fn map_texture(map: &ufbx::MaterialMap, textures: &HashMap<u32, TextureRef>) -> Option<TextureRef> {
    if !map.texture_enabled {
        return None;
    }
    map.texture.as_ref().and_then(|t| textures.get(&t.element.typed_id).copied())
}

fn map_value(map: &ufbx::MaterialMap) -> Option<f32> {
    map.has_value.then_some(map.value_vec4.x as f32)
}

fn convert_material(m: &ufbx::Material, textures: &HashMap<u32, TextureRef>) -> Material {
    let pbr = &m.pbr;
    let legacy = &m.fbx;
    let color_map = if pbr.base_color.has_value || pbr.base_color.texture.is_some() { &pbr.base_color } else { &legacy.diffuse_color };
    let c = color_map.value_vec4;
    let factor = map_value(&pbr.base_factor).unwrap_or(1.0);
    let opacity = map_value(&pbr.opacity).unwrap_or(1.0).clamp(0.0, 1.0);
    let lin = |v: f64| srgb_to_linear(v.clamp(0.0, 1.0) as f32);

    let mut mat = Material {
        name: m.element.name.to_string(),
        base_color: if color_map.has_value {
            [lin(c.x) * factor, lin(c.y) * factor, lin(c.z) * factor, opacity]
        } else {
            [0.8, 0.8, 0.8, opacity]
        },
        base_color_texture: map_texture(&pbr.base_color, textures).or_else(|| map_texture(&legacy.diffuse_color, textures)),
        metallic: map_value(&pbr.metalness).unwrap_or(0.0).clamp(0.0, 1.0),
        roughness: map_value(&pbr.roughness).unwrap_or(0.6).clamp(0.0, 1.0),
        metallic_texture: map_texture(&pbr.metalness, textures),
        metallic_channel: 0,
        roughness_texture: map_texture(&pbr.roughness, textures),
        roughness_channel: 0,
        glossiness_workflow: m.features.roughness_as_glossiness.enabled,
        normal_texture: map_texture(&pbr.normal_map, textures).or_else(|| map_texture(&legacy.normal_map, textures)),
        occlusion_texture: map_texture(&pbr.ambient_occlusion, textures),
        emissive_texture: map_texture(&pbr.emission_color, textures),
        unlit: m.features.unlit.enabled,
        double_sided: true,
        ..Default::default()
    };
    let e = pbr.emission_color.value_vec4;
    let ef = map_value(&pbr.emission_factor).unwrap_or(if pbr.emission_color.has_value { 1.0 } else { 0.0 });
    if pbr.emission_color.has_value || mat.emissive_texture.is_some() {
        let base = if mat.emissive_texture.is_some() && !pbr.emission_color.has_value { [1.0; 3] } else { [lin(e.x), lin(e.y), lin(e.z)] };
        mat.emissive = [base[0] * ef, base[1] * ef, base[2] * ef];
    }
    if opacity < 0.999 || pbr.opacity.texture.is_some() {
        mat.alpha_mode = AlphaMode::Blend;
    }
    mat
}

struct ConvertedMesh {
    mesh: Mesh,
    /// Blend channel element_id -> morph target index.
    channel_targets: Vec<(u32, u32)>,
    default_weights: Vec<f32>,
}

fn convert_meshes(root: &ufbx::SceneRoot, ctx: &LoadCtx) -> Result<Vec<ConvertedMesh>> {
    (0..root.meshes.len())
        .into_par_iter()
        .map(|i| {
            ctx.progress.check()?;
            Ok(convert_mesh(&root.meshes[i]))
        })
        .collect()
}

/// Dense blend shape offsets for one channel, indexed by mesh vertex.
struct DenseTarget {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
}

fn convert_mesh(mesh: &ufbx::Mesh) -> ConvertedMesh {
    let mut out = Mesh { name: mesh.element.name.to_string(), primitives: Vec::new() };
    let num_vertices = mesh.num_vertices;

    // Morph targets (one per blend channel).
    let mut channel_targets = Vec::new();
    let mut default_weights = Vec::new();
    let mut dense: Vec<DenseTarget> = Vec::new();
    for deformer in mesh.blend_deformers.iter() {
        for channel in deformer.channels.iter() {
            let shape = channel.target_shape.as_ref().or_else(|| channel.keyframes.iter().last().map(|k| &k.shape));
            let Some(shape) = shape else { continue };
            let mut t = DenseTarget { positions: vec![[0.0; 3]; num_vertices], normals: Vec::new() };
            let has_normals = !shape.normal_offsets.is_empty();
            if has_normals {
                t.normals = vec![[0.0; 3]; num_vertices];
            }
            for (k, &v) in shape.offset_vertices.iter().enumerate() {
                let v = v as usize;
                if v >= num_vertices {
                    continue;
                }
                if let Some(p) = shape.position_offsets.get(k) {
                    t.positions[v] = vec3(*p).to_array();
                }
                if has_normals
                    && let Some(n) = shape.normal_offsets.get(k) {
                        t.normals[v] = vec3(*n).to_array();
                    }
            }
            channel_targets.push((channel.element.element_id, dense.len() as u32));
            default_weights.push(channel.weight as f32);
            dense.push(t);
        }
    }

    let skin = mesh.skin_deformers.iter().next();
    let mut tri = vec![0u32; mesh.max_face_triangles.max(1) * 3];

    for part in mesh.material_parts.iter() {
        let mut prim = Primitive::new(Topology::Triangles);
        let mut lines = Primitive::new(Topology::Lines);
        prim.material = mesh.materials.get(part.index as usize).map(|m| m.element.typed_id as usize);
        lines.material = prim.material;
        let mut vertex_of: HashMap<VertexKey, u32> = HashMap::new();
        let mut source_vertex: Vec<u32> = Vec::new();
        let mut line_vertex_of: HashMap<u32, u32> = HashMap::new();

        let has_normals = mesh.vertex_normal.exists;
        let has_uvs = mesh.vertex_uv.exists;
        let has_colors = mesh.vertex_color.exists;

        for &fi in part.face_indices.iter() {
            let face = mesh.faces[fi as usize];
            if face.num_indices == 2 {
                for k in 0..2 {
                    let corner = face.index_begin + k;
                    let v = mesh.vertex_indices[corner as usize];
                    let next = lines.positions.len() as u32;
                    let idx = *line_vertex_of.entry(v).or_insert_with(|| {
                        lines.positions.push(vec3(mesh.vertex_position[corner as usize]).to_array());
                        next
                    });
                    lines.indices.push(idx);
                }
                continue;
            }
            if face.num_indices < 3 {
                continue;
            }
            let n = ufbx::triangulate_face(&mut tri, mesh, face) as usize;
            for &corner in &tri[..n * 3] {
                let c = corner as usize;
                let v = mesh.vertex_indices[c];
                let pos = vec3(mesh.vertex_position[c]);
                let normal = if has_normals { vec3(mesh.vertex_normal[c]).normalize_or(Vec3::Y) } else { Vec3::ZERO };
                let uv = if has_uvs {
                    let t = mesh.vertex_uv[c];
                    [t.x as f32, 1.0 - t.y as f32]
                } else {
                    [0.0; 2]
                };
                let color = if has_colors {
                    let q = |x: f64| (x.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                    let cc = mesh.vertex_color[c];
                    [q(cc.x), q(cc.y), q(cc.z), q(cc.w)]
                } else {
                    [255; 4]
                };
                let key = VertexKey {
                    vertex: v,
                    normal: normal.to_array().map(f32::to_bits),
                    uv: uv.map(f32::to_bits),
                    color,
                };
                let next = prim.positions.len() as u32;
                let idx = *vertex_of.entry(key).or_insert_with(|| {
                    prim.positions.push(pos.to_array());
                    if has_normals {
                        prim.normals.push(normal.to_array());
                    }
                    if has_uvs {
                        prim.uvs.push(uv);
                    }
                    if has_colors {
                        prim.colors.push(color);
                    }
                    source_vertex.push(v);
                    next
                });
                prim.indices.push(idx);
            }
        }

        if let Some(skin) = skin {
            prim.joints.reserve(source_vertex.len());
            prim.weights.reserve(source_vertex.len());
            for &v in &source_vertex {
                let (j, w) = skin_weights(skin, v as usize);
                prim.joints.push(j);
                prim.weights.push(w);
            }
        }
        for t in &dense {
            prim.targets.push(MorphTarget {
                positions: source_vertex.iter().map(|&v| t.positions[v as usize]).collect(),
                normals: if t.normals.is_empty() {
                    Vec::new()
                } else {
                    source_vertex.iter().map(|&v| t.normals[v as usize]).collect()
                },
            });
        }
        if !prim.indices.is_empty() {
            out.primitives.push(prim);
        }
        if !lines.indices.is_empty() {
            out.primitives.push(lines);
        }
    }
    ConvertedMesh { mesh: out, channel_targets, default_weights }
}

#[derive(Hash, PartialEq, Eq)]
struct VertexKey {
    vertex: u32,
    normal: [u32; 3],
    uv: [u32; 2],
    color: [u8; 4],
}

/// Four strongest influences of a vertex, normalised.
fn skin_weights(skin: &ufbx::SkinDeformer, vertex: usize) -> ([u16; 4], [f32; 4]) {
    let mut joints = [0u16; 4];
    let mut weights = [0f32; 4];
    let Some(sv) = skin.vertices.get(vertex) else { return (joints, [1.0, 0.0, 0.0, 0.0]) };
    let begin = sv.weight_begin as usize;
    let influences: &[ufbx::SkinWeight] = &skin.weights;
    let end = (begin + sv.num_weights as usize).min(influences.len());
    let begin = begin.min(end);
    let mut all: Vec<(u32, f32)> = influences[begin..end].iter().map(|w| (w.cluster_index, w.weight as f32)).collect();
    all.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut total = 0.0;
    for (k, (cluster, weight)) in all.into_iter().take(4).enumerate() {
        joints[k] = cluster.min(u16::MAX as u32) as u16;
        weights[k] = weight.max(0.0);
        total += weights[k];
    }
    if total > 0.0 {
        for w in &mut weights {
            *w /= total;
        }
    } else {
        weights = [1.0, 0.0, 0.0, 0.0];
    }
    (joints, weights)
}
