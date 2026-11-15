//! A loaded model on the GPU: buffers, textures, materials, draw list and
//! the compute work needed for animated meshes.

use super::context::{DRAW_UNIFORM_SIZE, GpuContext};
use crate::anim::Pose;
use crate::scene::{Aabb, AlphaMode, Material, PrimitiveRole, Scene, TextureRef, Topology};
use bytemuck::{Pod, Zeroable};
use eframe::wgpu;
use eframe::wgpu::util::DeviceExt;
use glam::{Mat3, Mat4};
use rayon::prelude::*;
use std::sync::Arc;
use std::sync::OnceLock;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MaterialUniform {
    base_color: [f32; 4],
    emissive: [f32; 4],
    params: [f32; 4],
    params2: [f32; 4],
    channels: [u32; 4],
    uv: [[f32; 4]; 12],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DrawUniform {
    model: [[f32; 4]; 4],
    normal: [[f32; 4]; 3],
    flags: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DeformParams {
    vertex_count: u32,
    target_count: u32,
    skinned: u32,
    row_width: u32,
}

pub struct GpuMaterial {
    pub bind_group: wgpu::BindGroup,
    pub blend: bool,
    pub double_sided: bool,
}

pub struct GpuPrimitive {
    pub topology: Topology,
    pub role: PrimitiveRole,
    /// Positions + normals (24 bytes per vertex).
    pub stream0: wgpu::Buffer,
    /// UVs + colours (12 bytes per vertex), only when the primitive has them.
    pub stream1: Option<wgpu::Buffer>,
    pub index: Option<wgpu::Buffer>,
    pub index_count: u32,
    pub vertex_count: u32,
    pub material: usize,
    pub bounds: Aabb,
    morph: Option<wgpu::Buffer>,
    target_count: u32,
    skin: Option<wgpu::Buffer>,
    /// Line-list indices for the wireframe overlay, built on first use.
    wire: OnceLock<Option<wgpu::Buffer>>,
}

pub struct Draw {
    pub node: usize,
    pub prim: usize,
    /// Index into `deforms` when the vertices are animated on the GPU.
    pub deform: Option<usize>,
    pub slot: u32,
    pub world_bounds: Aabb,
    pub mirrored: bool,
    /// Skinned draws are already in world space and are never culled.
    pub skinned: bool,
}

struct Deform {
    output: wgpu::Buffer,
    weights: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    node: usize,
    vertex_count: u32,
    target_count: u32,
}

struct GpuSkin {
    joints: Vec<usize>,
    inverse_bind: Vec<Mat4>,
    buffer: wgpu::Buffer,
}

pub struct GpuScene {
    pub prims: Vec<GpuPrimitive>,
    pub materials: Vec<GpuMaterial>,
    pub draws: Vec<Draw>,
    deforms: Vec<Deform>,
    skins: Vec<GpuSkin>,
    draw_buffer: wgpu::Buffer,
    pub draw_bind_group: wgpu::BindGroup,
    draw_data: Vec<u8>,
    deform_dirty: bool,
    _textures: Vec<wgpu::Texture>,
    pub has_feature_edges: bool,
}

impl GpuScene {
    /// Uploads `scene` (vertex data, textures). Heavy CPU arrays are released
    /// from `scene` afterwards; node/animation data stays for posing.
    pub fn upload(ctx: &Arc<GpuContext>, scene: &mut Scene, root: Mat4) -> GpuScene {
        let device = &ctx.device;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("upload") });

        // --- Textures: sRGB for colour usage, linear for data usage.
        let mut srgb_used = vec![false; scene.images.len()];
        let mut linear_used = vec![false; scene.images.len()];
        let image_of = |t: &TextureRef| scene.textures.get(t.texture).map(|tex| tex.image);
        for m in &scene.materials {
            for t in [m.base_color_texture, m.emissive_texture].iter().flatten() {
                if let Some(i) = image_of(t) {
                    srgb_used[i] = true;
                }
            }
            for t in [m.metallic_texture, m.roughness_texture, m.normal_texture, m.occlusion_texture].iter().flatten() {
                if let Some(i) = image_of(t) {
                    linear_used[i] = true;
                }
            }
        }
        let mut textures = Vec::new();
        let mut srgb_views: Vec<Option<wgpu::TextureView>> = vec![None; scene.images.len()];
        let mut linear_views: Vec<Option<wgpu::TextureView>> = vec![None; scene.images.len()];
        for (i, img) in scene.images.iter().enumerate() {
            for (used, srgb) in [(srgb_used[i], true), (linear_used[i], false)] {
                if !used {
                    continue;
                }
                let tex = create_texture(ctx, &mut encoder, img, srgb);
                let view = tex.create_view(&Default::default());
                if srgb {
                    srgb_views[i] = Some(view);
                } else {
                    linear_views[i] = Some(view);
                }
                textures.push(tex);
            }
        }

        // --- Materials (+ a default one at the end).
        let default_material = Material::default();
        let materials: Vec<GpuMaterial> = scene
            .materials
            .iter()
            .chain(std::iter::once(&default_material))
            .map(|m| create_material(ctx, scene, m, &srgb_views, &linear_views))
            .collect();
        let default_index = materials.len() - 1;

        // --- Primitives (vertex data packed in parallel).
        let mut prims = Vec::new();
        let mut mesh_prims: Vec<std::ops::Range<usize>> = Vec::new();
        let compute = ctx.compute.is_some();
        for mesh in &scene.meshes {
            let start = prims.len();
            for p in &mesh.primitives {
                let animated_input = compute && (p.is_skinned() || !p.targets.is_empty());
                prims.push(create_primitive(ctx, p, animated_input, default_index));
            }
            mesh_prims.push(start..prims.len());
        }

        // --- Skins.
        let skins: Vec<GpuSkin> = scene
            .skins
            .iter()
            .map(|s| GpuSkin {
                joints: s.joints.clone(),
                inverse_bind: s.inverse_bind.clone(),
                buffer: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("joints"),
                    size: (s.joints.len().max(1) * 64) as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
            })
            .collect();

        // --- Draws and GPU deformation instances.
        let empty = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("empty storage"),
            size: 64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let mut draws = Vec::new();
        let mut deforms = Vec::new();
        for n in scene.traversal_order() {
            let node = &scene.nodes[n];
            let Some(range) = node.mesh.and_then(|m| mesh_prims.get(m)).cloned() else { continue };
            for (k, pi) in range.enumerate() {
                let src = &scene.meshes[node.mesh.unwrap()].primitives[k];
                let skin = node.skin.filter(|_| src.is_skinned());
                let gp = &prims[pi];
                let mut deform = None;
                if let Some(kernels) = &ctx.compute
                    && (skin.is_some() || gp.target_count > 0) {
                        let d = create_deform(ctx, kernels, gp, &empty, skin.map(|s| &skins[s].buffer), n, skin);
                        deform = Some(deforms.len());
                        deforms.push(d);
                    }
                draws.push(Draw {
                    node: n,
                    prim: pi,
                    deform,
                    slot: draws.len() as u32,
                    world_bounds: Aabb::EMPTY,
                    mirrored: false,
                    skinned: skin.is_some() && deform.is_some(),
                });
            }
        }

        let draw_stride = ctx.draw_stride;
        let draw_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("draw uniforms"),
            size: (draws.len().max(1) as u64) * draw_stride,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let draw_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("draws"),
            layout: &ctx.draw_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &draw_buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(DRAW_UNIFORM_SIZE),
                }),
            }],
        });

        // Submit textures/mips first; buffers were created with data already.
        ctx.queue.submit([encoder.finish()]);

        // Release CPU copies: the GPU owns the data now (keeps memory low).
        for mesh in &mut scene.meshes {
            for p in &mut mesh.primitives {
                let targets = p.targets.len();
                *p = crate::scene::Primitive {
                    targets: vec![Default::default(); targets],
                    ..crate::scene::Primitive::new(p.topology)
                };
            }
        }
        for img in &mut scene.images {
            img.pixels = Arc::new(Vec::new());
        }

        let has_feature_edges = prims.iter().any(|p| p.role == PrimitiveRole::FeatureEdges);
        let mut gpu = GpuScene {
            prims,
            materials,
            draws,
            deforms,
            skins,
            draw_buffer,
            draw_bind_group,
            draw_data: vec![0; 0],
            deform_dirty: true,
            _textures: textures,
            has_feature_edges,
        };
        let globals = scene.rest_globals();
        let weights: Vec<Vec<f32>> = scene.nodes.iter().map(|n| n.weights.clone()).collect();
        gpu.update(ctx, &globals, &weights, root);
        gpu
    }

    /// Uploads transforms, joint matrices and morph weights for a pose.
    pub fn apply_pose(&mut self, ctx: &GpuContext, pose: &Pose, root: Mat4) {
        self.update(ctx, &pose.globals, &pose.weights, root);
    }

    fn update(&mut self, ctx: &GpuContext, globals: &[Mat4], weights: &[Vec<f32>], root: Mat4) {
        let stride = ctx.draw_stride as usize;
        self.draw_data.resize(self.draws.len().max(1) * stride, 0);
        for d in &mut self.draws {
            let model = if d.skinned { root } else { root * globals[d.node] };
            let normal = Mat3::from_mat4(model).inverse().transpose();
            let normal = if normal.is_finite() { normal } else { Mat3::from_mat4(model) };
            d.mirrored = Mat3::from_mat4(model).determinant() < 0.0;
            d.world_bounds = if d.skinned { Aabb::EMPTY } else { self.prims[d.prim].bounds.transformed(&model) };
            let u = DrawUniform {
                model: model.to_cols_array_2d(),
                normal: [
                    normal.x_axis.extend(0.0).to_array(),
                    normal.y_axis.extend(0.0).to_array(),
                    normal.z_axis.extend(0.0).to_array(),
                ],
                flags: [d.mirrored as u32, 0, 0, 0],
            };
            let off = d.slot as usize * stride;
            self.draw_data[off..off + std::mem::size_of::<DrawUniform>()].copy_from_slice(bytemuck::bytes_of(&u));
        }
        ctx.queue.write_buffer(&self.draw_buffer, 0, &self.draw_data);

        for skin in &self.skins {
            let mats: Vec<[f32; 16]> = skin
                .joints
                .iter()
                .zip(&skin.inverse_bind)
                .map(|(&j, ibm)| (globals.get(j).copied().unwrap_or(Mat4::IDENTITY) * *ibm).to_cols_array())
                .collect();
            if !mats.is_empty() {
                ctx.queue.write_buffer(&skin.buffer, 0, bytemuck::cast_slice(&mats));
            }
        }
        for d in &self.deforms {
            if d.target_count > 0 {
                let mut w = weights.get(d.node).cloned().unwrap_or_default();
                w.resize(d.target_count as usize, 0.0);
                ctx.queue.write_buffer(&d.weights, 0, bytemuck::cast_slice(&w));
            }
        }
        self.deform_dirty = true;
    }

    /// Records the compute passes that deform animated meshes (only when the
    /// pose changed since the last call).
    pub fn encode_deform(&mut self, ctx: &GpuContext, encoder: &mut wgpu::CommandEncoder) {
        if !self.deform_dirty || self.deforms.is_empty() {
            return;
        }
        let Some(kernels) = &ctx.compute else { return };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("deform"), timestamp_writes: None });
        pass.set_pipeline(&kernels.deform);
        for d in &self.deforms {
            let (x, y, _) = kernels.dispatch_size(d.vertex_count);
            pass.set_bind_group(0, &d.bind_group, &[]);
            pass.dispatch_workgroups(x, y, 1);
        }
        self.deform_dirty = false;
    }

    /// Vertex buffer holding the current (possibly deformed) positions.
    pub fn stream0(&self, draw: &Draw) -> &wgpu::Buffer {
        match draw.deform {
            Some(i) => &self.deforms[i].output,
            None => &self.prims[draw.prim].stream0,
        }
    }

    /// Wireframe index buffer of a triangle primitive (built on the GPU the
    /// first time it is needed).
    pub fn wire_indices(&self, ctx: &GpuContext, prim: usize) -> Option<&wgpu::Buffer> {
        let p = &self.prims[prim];
        p.wire
            .get_or_init(|| {
                let kernels = ctx.compute.as_ref()?;
                let index = p.index.as_ref()?;
                if p.topology != Topology::Triangles || p.index_count < 3 {
                    return None;
                }
                let tris = p.index_count / 3;
                let out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("wire indices"),
                    size: tris as u64 * 6 * 4,
                    usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                });
                let (x, y, row) = kernels.dispatch_size(tris);
                let params = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("wire params"),
                    contents: bytemuck::cast_slice(&[tris, row, 0, 0]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("wire"),
                    layout: &kernels.wire_layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: index.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: out.as_entire_binding() },
                    ],
                });
                let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("wire") });
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("wire"), timestamp_writes: None });
                    pass.set_pipeline(&kernels.wire);
                    pass.set_bind_group(0, &bind, &[]);
                    pass.dispatch_workgroups(x, y, 1);
                }
                ctx.queue.submit([enc.finish()]);
                Some(out)
            })
            .as_ref()
    }
}

fn create_texture(ctx: &GpuContext, encoder: &mut wgpu::CommandEncoder, img: &crate::scene::Image, srgb: bool) -> wgpu::Texture {
    // Downscale images larger than the GPU allows (rare, e.g. 16k maps).
    let max = ctx.max_texture_size.max(1);
    let (w, h, pixels): (u32, u32, std::borrow::Cow<[u8]>) = if img.width > max || img.height > max {
        let scale = max as f32 / img.width.max(img.height) as f32;
        let nw = ((img.width as f32 * scale) as u32).max(1);
        let nh = ((img.height as f32 * scale) as u32).max(1);
        let src = image::RgbaImage::from_raw(img.width, img.height, img.pixels.to_vec()).unwrap_or_default();
        let resized = image::imageops::resize(&src, nw, nh, image::imageops::FilterType::Triangle);
        (nw, nh, std::borrow::Cow::Owned(resized.into_raw()))
    } else {
        (img.width.max(1), img.height.max(1), std::borrow::Cow::Borrowed(&img.pixels[..]))
    };
    let format = if srgb { wgpu::TextureFormat::Rgba8UnormSrgb } else { wgpu::TextureFormat::Rgba8Unorm };
    let size = wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 };
    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some(&img.name),
        size,
        mip_level_count: super::context::mip_count(w, h),
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    if pixels.len() as u64 >= 4 * w as u64 * h as u64 {
        // Upload in strips of at most ~4 MB: each write gets its own staging
        // buffer, and small ones keep the upload heap (system RAM) small.
        let row_bytes = 4 * w as usize;
        let rows_per_strip = ((4 << 20) / row_bytes).clamp(1, h as usize) as u32;
        let mut y = 0;
        while y < h {
            let rows = rows_per_strip.min(h - y);
            let start = y as usize * row_bytes;
            ctx.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: 0, y, z: 0 },
                    aspect: wgpu::TextureAspect::All,
                },
                &pixels[start..start + rows as usize * row_bytes],
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4 * w), rows_per_image: Some(rows) },
                wgpu::Extent3d { width: w, height: rows, depth_or_array_layers: 1 },
            );
            y += rows;
        }
        ctx.mips.generate(&ctx.device, encoder, &texture);
    }
    texture
}

fn create_material(
    ctx: &GpuContext,
    scene: &Scene,
    m: &Material,
    srgb_views: &[Option<wgpu::TextureView>],
    linear_views: &[Option<wgpu::TextureView>],
) -> GpuMaterial {
    let slots = [
        (m.base_color_texture, true),
        (m.metallic_texture, false),
        (m.roughness_texture, false),
        (m.normal_texture, false),
        (m.occlusion_texture, false),
        (m.emissive_texture, true),
    ];
    let mut mask = 0u32;
    let mut uv = [[0f32; 4]; 12];
    let mut views: Vec<&wgpu::TextureView> = Vec::with_capacity(6);
    let mut samplers: Vec<Arc<wgpu::Sampler>> = Vec::with_capacity(6);
    for (i, (tref, srgb)) in slots.iter().enumerate() {
        let tr = tref.and_then(|t| scene.textures.get(t.texture).map(|tex| (t, tex)));
        let view = tr.and_then(|(_, tex)| if *srgb { srgb_views[tex.image].as_ref() } else { linear_views[tex.image].as_ref() });
        let transform = tr.and_then(|(t, _)| t.transform).unwrap_or([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        uv[i * 2] = [transform[0], transform[1], transform[2], 0.0];
        uv[i * 2 + 1] = [transform[3], transform[4], transform[5], 0.0];
        match (view, tr) {
            (Some(v), Some((_, tex))) => {
                mask |= 1 << i;
                views.push(v);
                samplers.push(ctx.sampler(&tex.sampler));
            }
            _ => {
                views.push(&ctx.white);
                samplers.push(ctx.default_sampler.clone());
            }
        }
    }
    let u = MaterialUniform {
        base_color: m.base_color,
        emissive: [m.emissive[0], m.emissive[1], m.emissive[2], 0.0],
        params: [m.metallic, m.roughness, m.normal_scale, m.occlusion_strength],
        params2: [
            m.alpha_cutoff,
            match m.alpha_mode {
                AlphaMode::Opaque => 0.0,
                AlphaMode::Mask => 1.0,
                AlphaMode::Blend => 2.0,
            },
            m.unlit as u32 as f32,
            m.glossiness_workflow as u32 as f32,
        ],
        channels: [m.metallic_channel.min(3) as u32, m.roughness_channel.min(3) as u32, mask, 0],
        uv,
    };
    let buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("material"),
        contents: bytemuck::bytes_of(&u),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let mut entries = vec![wgpu::BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() }];
    for (i, v) in views.iter().enumerate() {
        entries.push(wgpu::BindGroupEntry { binding: 1 + i as u32, resource: wgpu::BindingResource::TextureView(v) });
    }
    for (i, s) in samplers.iter().enumerate() {
        entries.push(wgpu::BindGroupEntry { binding: 7 + i as u32, resource: wgpu::BindingResource::Sampler(s) });
    }
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(&m.name),
        layout: &ctx.material_layout,
        entries: &entries,
    });
    GpuMaterial { bind_group, blend: m.alpha_mode == AlphaMode::Blend, double_sided: m.double_sided }
}

fn create_primitive(ctx: &GpuContext, p: &crate::scene::Primitive, animated_input: bool, default_material: usize) -> GpuPrimitive {
    let device = &ctx.device;
    let n = p.positions.len();
    let has_normals = p.normals.len() == n;
    let stream0: Vec<f32> = p
        .positions
        .par_iter()
        .enumerate()
        .flat_map_iter(|(i, pos)| {
            let nrm = if has_normals { p.normals[i] } else { [0.0, 1.0, 0.0] };
            [pos[0], pos[1], pos[2], nrm[0], nrm[1], nrm[2]]
        })
        .collect();
    let mut usage0 = wgpu::BufferUsages::VERTEX;
    if animated_input {
        usage0 |= wgpu::BufferUsages::STORAGE;
    }
    let stream0_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("stream0"),
        contents: bytemuck::cast_slice(&stream0),
        usage: usage0,
    });
    drop(stream0);

    let has_uv = p.uvs.len() == n;
    let has_color = p.colors.len() == n;
    let stream1 = (has_uv || has_color).then(|| {
        let data: Vec<u32> = (0..n)
            .into_par_iter()
            .flat_map_iter(|i| {
                let uv = if has_uv { p.uvs[i] } else { [0.0, 0.0] };
                let c = if has_color { p.colors[i] } else { [255; 4] };
                [uv[0].to_bits(), uv[1].to_bits(), u32::from_le_bytes(c)]
            })
            .collect();
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("stream1"),
            contents: bytemuck::cast_slice(&data),
            usage: wgpu::BufferUsages::VERTEX,
        })
    });

    let index = (!p.indices.is_empty()).then(|| {
        let mut usage = wgpu::BufferUsages::INDEX;
        if ctx.compute.is_some() && p.topology == Topology::Triangles {
            usage |= wgpu::BufferUsages::STORAGE;
        }
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("indices"),
            contents: bytemuck::cast_slice(&p.indices),
            usage,
        })
    });

    let target_count = if animated_input { p.targets.len() as u32 } else { 0 };
    let morph = (target_count > 0).then(|| {
        let data: Vec<f32> = p
            .targets
            .par_iter()
            .flat_map_iter(|t| {
                (0..n).flat_map(move |i| {
                    let dp = t.positions.get(i).copied().unwrap_or([0.0; 3]);
                    let dn = t.normals.get(i).copied().unwrap_or([0.0; 3]);
                    [dp[0], dp[1], dp[2], dn[0], dn[1], dn[2]]
                })
            })
            .collect();
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("morph targets"),
            contents: bytemuck::cast_slice(&data),
            usage: wgpu::BufferUsages::STORAGE,
        })
    });
    let skin = (animated_input && p.is_skinned()).then(|| {
        let data: Vec<u32> = (0..n)
            .into_par_iter()
            .flat_map_iter(|i| {
                let j = p.joints[i];
                let w = p.weights[i];
                [
                    j[0] as u32 | (j[1] as u32) << 16,
                    j[2] as u32 | (j[3] as u32) << 16,
                    w[0].to_bits(),
                    w[1].to_bits(),
                    w[2].to_bits(),
                    w[3].to_bits(),
                ]
            })
            .collect();
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("skin"),
            contents: bytemuck::cast_slice(&data),
            usage: wgpu::BufferUsages::STORAGE,
        })
    });

    GpuPrimitive {
        topology: p.topology,
        role: p.role,
        stream0: stream0_buf,
        stream1,
        index,
        index_count: p.indices.len() as u32,
        vertex_count: n as u32,
        material: p.material.unwrap_or(default_material),
        bounds: Aabb::from_points(&p.positions),
        morph,
        target_count,
        skin,
        wire: OnceLock::new(),
    }
}

fn create_deform(
    ctx: &GpuContext,
    kernels: &super::context::ComputeKernels,
    p: &GpuPrimitive,
    empty: &wgpu::Buffer,
    joints: Option<&wgpu::Buffer>,
    node: usize,
    skin: Option<usize>,
) -> Deform {
    let device = &ctx.device;
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("deformed vertices"),
        size: (p.vertex_count.max(1) as u64) * 24,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let weights = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("morph weights"),
        size: (p.target_count.max(1) as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let skinned = skin.is_some() && p.skin.is_some() && joints.is_some();
    let (_, _, row) = kernels.dispatch_size(p.vertex_count);
    let params = DeformParams {
        vertex_count: p.vertex_count,
        target_count: if p.morph.is_some() { p.target_count } else { 0 },
        skinned: skinned as u32,
        row_width: row,
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("deform params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("deform"),
        layout: &kernels.deform_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: p.stream0.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: p.morph.as_ref().unwrap_or(empty).as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: weights.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: p.skin.as_ref().filter(|_| skinned).unwrap_or(empty).as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: joints.filter(|_| skinned).unwrap_or(empty).as_entire_binding(),
            },
            wgpu::BindGroupEntry { binding: 6, resource: output.as_entire_binding() },
        ],
    });
    Deform { output, weights, bind_group, node, vertex_count: p.vertex_count, target_count: p.target_count }
}
