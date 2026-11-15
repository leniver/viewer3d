//! wgpu renderer: draws a [`GpuScene`] into an offscreen MSAA target that the
//! UI shows as an image (also used for screenshots and headless rendering).

pub mod context;
pub mod gpu_scene;

pub use context::GpuContext;
pub use gpu_scene::GpuScene;

use crate::scene::{Aabb, PrimitiveRole, Topology};
use bytemuck::{Pod, Zeroable};
use context::{COLOR_FORMAT, DEPTH_FORMAT, SAMPLES};
use eframe::wgpu;
use glam::{Mat4, Vec3, Vec4};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum DisplayMode {
    Shaded,
    ShadedWireframe,
    Wireframe,
}

#[derive(Clone, Copy, Debug)]
pub struct Lighting {
    /// Directions towards the lights, world space.
    pub key_dir: Vec3,
    pub key: [f32; 3],
    pub fill_dir: Vec3,
    pub fill: [f32; 3],
    pub sky: [f32; 3],
    pub ground: [f32; 3],
    pub ambient: f32,
    pub exposure: f32,
    pub flat: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct Grid {
    pub height: f32,
    pub cell: f32,
    pub center: [f32; 2],
    pub extent: f32,
    pub color: [f32; 4],
}

/// Everything that describes one rendered frame.
#[derive(Clone, Copy, Debug)]
pub struct ViewParams {
    pub view_proj: Mat4,
    pub eye: Vec3,
    pub width: u32,
    pub height: u32,
    pub lighting: Lighting,
    /// Linear colours.
    pub bg_top: [f32; 3],
    pub bg_bottom: [f32; 3],
    pub grid: Option<Grid>,
    pub display: DisplayMode,
    pub show_edges: bool,
    pub wire_color: [f32; 4],
    pub point_size: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FrameUniform {
    view_proj: [[f32; 4]; 4],
    inv_view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 4],
    key_dir: [f32; 4],
    key_color: [f32; 4],
    fill_dir: [f32; 4],
    fill_color: [f32; 4],
    sky_color: [f32; 4],
    ground_color: [f32; 4],
    bg_top: [f32; 4],
    bg_bottom: [f32; 4],
    grid_params: [f32; 4],
    grid_center: [f32; 4],
    grid_color: [f32; 4],
    wire_color: [f32; 4],
    viewport: [f32; 4],
    params: [f32; 4],
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Kind {
    Surface,
    Lines,
    Wire,
    Points,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct PipelineKey {
    kind: Kind,
    stream1: bool,
    blend: bool,
    cull: bool,
    mirrored: bool,
}

struct Target {
    width: u32,
    height: u32,
    /// Rgba8Unorm storage holding sRGB-encoded bytes (what egui expects).
    color: wgpu::Texture,
    /// Raw (Unorm) view, sampled by egui.
    color_view: wgpu::TextureView,
    /// sRGB view of the same texture: MSAA resolve target (linear -> sRGB encode).
    resolve_view: wgpu::TextureView,
    msaa_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
}

pub struct Renderer {
    pub ctx: Arc<GpuContext>,
    layout: wgpu::PipelineLayout,
    pipelines: HashMap<PipelineKey, wgpu::RenderPipeline>,
    background: wgpu::RenderPipeline,
    grid: wgpu::RenderPipeline,
    frame_buffer: wgpu::Buffer,
    frame_bind_group: wgpu::BindGroup,
    target: Option<Target>,
    /// Incremented whenever the render target is recreated.
    target_generation: u64,
}

const STREAM0_ATTRS: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];
const STREAM1_ATTRS: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![2 => Float32x2, 3 => Unorm8x4];

impl Renderer {
    pub fn new(ctx: Arc<GpuContext>) -> Self {
        let device = &ctx.device;
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scene"),
            bind_group_layouts: &[Some(&ctx.frame_layout), Some(&ctx.material_layout), Some(&ctx.draw_layout)],
            immediate_size: 0,
        });
        let frame_only = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("frame only"),
            bind_group_layouts: &[Some(&ctx.frame_layout)],
            immediate_size: 0,
        });
        let frame_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame uniform"),
            size: std::mem::size_of::<FrameUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let frame_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frame"),
            layout: &ctx.frame_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: frame_buffer.as_entire_binding() }],
        });
        let simple = |label: &str, vs: &str, fs: &str, topology, depth: Option<wgpu::DepthStencilState>, blend| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&frame_only),
                vertex: wgpu::VertexState {
                    module: &ctx.scene_shader,
                    entry_point: Some(vs),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState { topology, ..Default::default() },
                depth_stencil: depth,
                multisample: wgpu::MultisampleState { count: SAMPLES, ..Default::default() },
                fragment: Some(wgpu::FragmentState {
                    module: &ctx.scene_shader,
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState { format: COLOR_FORMAT, blend, write_mask: wgpu::ColorWrites::ALL })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let background = simple(
            "background",
            "vs_background",
            "fs_background",
            wgpu::PrimitiveTopology::TriangleList,
            Some(depth_state(false, wgpu::CompareFunction::Always)),
            None,
        );
        let grid = simple(
            "grid",
            "vs_grid",
            "fs_grid",
            wgpu::PrimitiveTopology::TriangleStrip,
            Some(depth_state(false, wgpu::CompareFunction::GreaterEqual)),
            Some(wgpu::BlendState::ALPHA_BLENDING),
        );
        Self {
            ctx,
            layout,
            pipelines: HashMap::new(),
            background,
            grid,
            frame_buffer,
            frame_bind_group,
            target: None,
            target_generation: 0,
        }
    }

    /// Texture with the last rendered frame (single-sampled, sRGB bytes).
    pub fn color_view(&self) -> Option<&wgpu::TextureView> {
        self.target.as_ref().map(|t| &t.color_view)
    }

    /// Changes whenever [`Self::color_view`] refers to a new texture.
    pub fn target_generation(&self) -> u64 {
        self.target_generation
    }

    /// Recreates the render targets if the size changed; returns true then.
    pub fn resize(&mut self, width: u32, height: u32) -> bool {
        let max = self.ctx.max_texture_size;
        let (width, height) = (width.clamp(1, max), height.clamp(1, max));
        if self.target.as_ref().is_some_and(|t| t.width == width && t.height == height) {
            return false;
        }
        let device = &self.ctx.device;
        let size = wgpu::Extent3d { width, height, depth_or_array_layers: 1 };
        let make = |label, format, samples, usage, view_formats: &[wgpu::TextureFormat]| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size,
                mip_level_count: 1,
                sample_count: samples,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats,
            })
        };
        // egui samples native textures as Rgba8Unorm with gamma-encoded values,
        // so store sRGB bytes in a Unorm texture and resolve through an sRGB view.
        let color = make(
            "view color",
            wgpu::TextureFormat::Rgba8Unorm,
            1,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            &[COLOR_FORMAT],
        );
        let msaa = make("view msaa", COLOR_FORMAT, SAMPLES, wgpu::TextureUsages::RENDER_ATTACHMENT, &[]);
        let depth = make("view depth", DEPTH_FORMAT, SAMPLES, wgpu::TextureUsages::RENDER_ATTACHMENT, &[]);
        let resolve_view = color.create_view(&wgpu::TextureViewDescriptor { format: Some(COLOR_FORMAT), ..Default::default() });
        self.target = Some(Target {
            width,
            height,
            color_view: color.create_view(&Default::default()),
            resolve_view,
            color,
            msaa_view: msaa.create_view(&Default::default()),
            depth_view: depth.create_view(&Default::default()),
        });
        self.target_generation += 1;
        true
    }

    fn ensure_pipeline(&mut self, key: PipelineKey) {
        if self.pipelines.contains_key(&key) {
            return;
        }
        let step = if key.kind == Kind::Points { wgpu::VertexStepMode::Instance } else { wgpu::VertexStepMode::Vertex };
        let mut buffers = vec![Some(wgpu::VertexBufferLayout { array_stride: 24, step_mode: step, attributes: &STREAM0_ATTRS })];
        if key.stream1 {
            buffers.push(Some(wgpu::VertexBufferLayout { array_stride: 12, step_mode: step, attributes: &STREAM1_ATTRS }));
        }
        let (vs, fs, topology) = match (key.kind, key.stream1) {
            (Kind::Surface, true) => ("vs_full", "fs_main", wgpu::PrimitiveTopology::TriangleList),
            (Kind::Surface, false) => ("vs_basic", "fs_main", wgpu::PrimitiveTopology::TriangleList),
            (Kind::Lines, true) => ("vs_lines_full", "fs_lines", wgpu::PrimitiveTopology::LineList),
            (Kind::Lines, false) => ("vs_lines_basic", "fs_lines", wgpu::PrimitiveTopology::LineList),
            (Kind::Wire, _) => ("vs_lines_basic", "fs_wire", wgpu::PrimitiveTopology::LineList),
            (Kind::Points, true) => ("vs_points_full", "fs_points", wgpu::PrimitiveTopology::TriangleStrip),
            (Kind::Points, false) => ("vs_points_basic", "fs_points", wgpu::PrimitiveTopology::TriangleStrip),
        };
        let depth_write = !key.blend && key.kind != Kind::Wire;
        let blend = (key.blend || key.kind == Kind::Wire || key.kind == Kind::Lines).then_some(wgpu::BlendState::ALPHA_BLENDING);
        let pipeline = self.ctx.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("scene pipeline"),
            layout: Some(&self.layout),
            vertex: wgpu::VertexState {
                module: &self.ctx.scene_shader,
                entry_point: Some(vs),
                compilation_options: Default::default(),
                buffers: &buffers,
            },
            primitive: wgpu::PrimitiveState {
                topology,
                front_face: if key.mirrored { wgpu::FrontFace::Cw } else { wgpu::FrontFace::Ccw },
                cull_mode: key.cull.then_some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(depth_state(depth_write, wgpu::CompareFunction::GreaterEqual)),
            multisample: wgpu::MultisampleState { count: SAMPLES, ..Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: &self.ctx.scene_shader,
                entry_point: Some(fs),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format: COLOR_FORMAT, blend, write_mask: wgpu::ColorWrites::ALL })],
            }),
            multiview_mask: None,
            cache: None,
        });
        self.pipelines.insert(key, pipeline);
    }

    /// Renders one frame into the offscreen target and submits it.
    pub fn render(&mut self, mut scene: Option<&mut GpuScene>, p: &ViewParams) {
        self.resize(p.width, p.height);
        let ctx = self.ctx.clone();
        self.write_frame_uniform(p);

        // Decide what to draw and make sure the pipelines exist.
        let mut opaque: Vec<(PipelineKey, usize, usize)> = Vec::new(); // key, material, draw
        let mut lines: Vec<(PipelineKey, usize, usize)> = Vec::new();
        let mut points: Vec<(PipelineKey, usize, usize)> = Vec::new();
        let mut blended: Vec<(f32, PipelineKey, usize, usize)> = Vec::new();
        let mut wires: Vec<usize> = Vec::new();
        if let Some(scene) = scene.as_deref() {
            let planes = frustum_planes(&p.view_proj);
            let surfaces = p.display != DisplayMode::Wireframe;
            for (di, d) in scene.draws.iter().enumerate() {
                if !d.skinned && !aabb_visible(&planes, &d.world_bounds) {
                    continue;
                }
                let prim = &scene.prims[d.prim];
                let mat = &scene.materials[prim.material];
                let stream1 = prim.stream1.is_some();
                match prim.topology {
                    Topology::Triangles => {
                        if surfaces {
                            let key = PipelineKey {
                                kind: Kind::Surface,
                                stream1,
                                blend: mat.blend,
                                cull: !mat.double_sided,
                                mirrored: d.mirrored,
                            };
                            if mat.blend {
                                let depth = (d.world_bounds.center() - p.eye).length_squared();
                                blended.push((depth, key, prim.material, di));
                            } else {
                                opaque.push((key, prim.material, di));
                            }
                        }
                        if p.display != DisplayMode::Shaded {
                            wires.push(di);
                        }
                    }
                    Topology::Lines => {
                        if prim.role == PrimitiveRole::FeatureEdges && !p.show_edges {
                            continue;
                        }
                        let key = PipelineKey { kind: Kind::Lines, stream1, blend: false, cull: false, mirrored: false };
                        lines.push((key, prim.material, di));
                    }
                    Topology::Points => {
                        let key = PipelineKey { kind: Kind::Points, stream1, blend: false, cull: false, mirrored: false };
                        points.push((key, prim.material, di));
                    }
                }
            }
            opaque.sort_by_key(|(k, m, _)| (k.stream1, k.cull, k.mirrored, *m));
            blended.sort_by(|a, b| b.0.total_cmp(&a.0));
        }
        let wire_key = PipelineKey { kind: Kind::Wire, stream1: false, blend: false, cull: false, mirrored: false };
        let keys: Vec<PipelineKey> = opaque
            .iter()
            .map(|e| e.0)
            .chain(lines.iter().map(|e| e.0))
            .chain(points.iter().map(|e| e.0))
            .chain(blended.iter().map(|e| e.1))
            .chain((!wires.is_empty()).then_some(wire_key))
            .collect();
        for k in keys {
            self.ensure_pipeline(k);
        }

        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });
        if let Some(s) = scene.as_deref_mut() {
            s.encode_deform(&ctx, &mut encoder);
        }
        let scene = scene.as_deref();
        // Wireframe index buffers are generated (on the GPU) before the pass.
        let wire_buffers: Vec<(usize, &wgpu::Buffer)> = match scene {
            Some(s) => wires.iter().filter_map(|&di| s.wire_indices(&ctx, s.draws[di].prim).map(|b| (di, b))).collect(),
            None => Vec::new(),
        };
        let target = self.target.as_ref().expect("render target");
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target.msaa_view,
                    depth_slice: None,
                    resolve_target: Some(&target.resolve_view),
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Discard },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &target.depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(0.0), store: wgpu::StoreOp::Discard }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, &self.frame_bind_group, &[]);
            pass.set_pipeline(&self.background);
            pass.draw(0..3, 0..1);

            if let Some(scene) = scene {
                let binder = Binder { pipelines: &self.pipelines, scene, stride: ctx.draw_stride };
                let mut current: Option<PipelineKey> = None;
                for &(key, m, di) in &opaque {
                    binder.bind(&mut pass, &mut current, key, m, di);
                    binder.draw(&mut pass, di);
                }
                if p.grid.is_some() {
                    pass.set_pipeline(&self.grid);
                    current = None;
                    pass.draw(0..4, 0..1);
                }
                for &(key, m, di) in &lines {
                    binder.bind(&mut pass, &mut current, key, m, di);
                    binder.draw(&mut pass, di);
                }
                for &(key, m, di) in &points {
                    binder.bind(&mut pass, &mut current, key, m, di);
                    let prim = &scene.prims[scene.draws[di].prim];
                    pass.draw(0..4, 0..prim.vertex_count);
                }
                for &(_, key, m, di) in &blended {
                    binder.bind(&mut pass, &mut current, key, m, di);
                    binder.draw(&mut pass, di);
                }
                for &(di, wire) in &wire_buffers {
                    let prim = &scene.prims[scene.draws[di].prim];
                    binder.bind(&mut pass, &mut current, wire_key, prim.material, di);
                    pass.set_index_buffer(wire.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..prim.index_count / 3 * 6, 0, 0..1);
                }
            } else if p.grid.is_some() {
                pass.set_pipeline(&self.grid);
                pass.draw(0..4, 0..1);
            }
        }
        ctx.queue.submit([encoder.finish()]);
    }

    fn write_frame_uniform(&self, p: &ViewParams) {
        let l = &p.lighting;
        let grid = p.grid.unwrap_or(Grid { height: 0.0, cell: 1.0, center: [0.0; 2], extent: 0.0, color: [0.0; 4] });
        let u = FrameUniform {
            view_proj: p.view_proj.to_cols_array_2d(),
            inv_view_proj: p.view_proj.inverse().to_cols_array_2d(),
            camera_pos: p.eye.extend(1.0).to_array(),
            key_dir: l.key_dir.normalize_or(Vec3::Y).extend(1.0).to_array(),
            key_color: [l.key[0], l.key[1], l.key[2], 1.0],
            fill_dir: l.fill_dir.normalize_or(Vec3::Y).extend(1.0).to_array(),
            fill_color: [l.fill[0], l.fill[1], l.fill[2], 1.0],
            sky_color: [l.sky[0], l.sky[1], l.sky[2], l.ambient],
            ground_color: [l.ground[0], l.ground[1], l.ground[2], 1.0],
            bg_top: [p.bg_top[0], p.bg_top[1], p.bg_top[2], 1.0],
            bg_bottom: [p.bg_bottom[0], p.bg_bottom[1], p.bg_bottom[2], 1.0],
            grid_params: [grid.height, grid.cell.max(1e-9), grid.extent, 0.0],
            grid_center: [grid.center[0], 0.0, grid.center[1], grid.extent],
            grid_color: grid.color,
            wire_color: p.wire_color,
            viewport: [p.width as f32, p.height as f32, p.point_size, l.flat as u32 as f32],
            params: [l.exposure, 4e-4, 0.0, 0.0],
        };
        self.ctx.queue.write_buffer(&self.frame_buffer, 0, bytemuck::bytes_of(&u));
    }

    /// Reads the last rendered frame back as tightly packed RGBA8 (sRGB).
    pub fn read_pixels(&self) -> Option<(u32, u32, Vec<u8>)> {
        let t = self.target.as_ref()?;
        let device = &self.ctx.device;
        let row = (t.width * 4).div_ceil(256) * 256;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: row as u64 * t.height as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("readback") });
        enc.copy_texture_to_buffer(
            t.color.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(t.height) },
            },
            wgpu::Extent3d { width: t.width, height: t.height, depth_or_array_layers: 1 },
        );
        self.ctx.queue.submit([enc.finish()]);
        let slice = buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok()?;
        rx.recv().ok()?.ok()?;
        let data = slice.get_mapped_range().ok()?;
        let mut out = Vec::with_capacity((t.width * t.height * 4) as usize);
        for y in 0..t.height {
            let start = (y * row) as usize;
            out.extend_from_slice(&data[start..start + (t.width * 4) as usize]);
        }
        drop(data);
        buffer.unmap();
        Some((t.width, t.height, out))
    }
}

/// Sets pipeline, bind groups and vertex buffers for one draw.
struct Binder<'a> {
    pipelines: &'a HashMap<PipelineKey, wgpu::RenderPipeline>,
    scene: &'a GpuScene,
    stride: u64,
}

impl Binder<'_> {
    fn bind(&self, pass: &mut wgpu::RenderPass<'_>, current: &mut Option<PipelineKey>, key: PipelineKey, material: usize, di: usize) {
        if *current != Some(key) {
            pass.set_pipeline(&self.pipelines[&key]);
            *current = Some(key);
        }
        let d = &self.scene.draws[di];
        pass.set_bind_group(1, &self.scene.materials[material].bind_group, &[]);
        pass.set_bind_group(2, &self.scene.draw_bind_group, &[(d.slot as u64 * self.stride) as u32]);
        pass.set_vertex_buffer(0, self.scene.stream0(d).slice(..));
        if key.stream1
            && let Some(s1) = &self.scene.prims[d.prim].stream1 {
                pass.set_vertex_buffer(1, s1.slice(..));
            }
    }

    fn draw(&self, pass: &mut wgpu::RenderPass<'_>, di: usize) {
        let prim = &self.scene.prims[self.scene.draws[di].prim];
        match &prim.index {
            Some(ib) => {
                pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..prim.index_count, 0, 0..1);
            }
            None => pass.draw(0..prim.vertex_count, 0..1),
        }
    }
}

fn depth_state(write: bool, compare: wgpu::CompareFunction) -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: DEPTH_FORMAT,
        depth_write_enabled: Some(write),
        depth_compare: Some(compare),
        stencil: Default::default(),
        bias: Default::default(),
    }
}

/// Clip planes (left, right, bottom, top, near) of a reverse-Z, infinite
/// projection; the far plane is at infinity.
fn frustum_planes(m: &Mat4) -> [Vec4; 5] {
    let r0 = m.row(0);
    let r1 = m.row(1);
    let r2 = m.row(2);
    let r3 = m.row(3);
    [r3 + r0, r3 - r0, r3 + r1, r3 - r1, r3 - r2]
}

fn aabb_visible(planes: &[Vec4; 5], b: &Aabb) -> bool {
    if b.is_empty() {
        return true;
    }
    planes.iter().all(|pl| {
        let p = Vec3::new(
            if pl.x >= 0.0 { b.max.x } else { b.min.x },
            if pl.y >= 0.0 { b.max.y } else { b.min.y },
            if pl.z >= 0.0 { b.max.z } else { b.min.z },
        );
        pl.truncate().dot(p) + pl.w >= 0.0
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frustum_culling() {
        let view = glam::camera::rh::view::look_at_mat4(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y);
        let proj = glam::camera::rh::proj::directx::perspective_infinite_reverse(1.0, 1.0, 0.1);
        let planes = frustum_planes(&(proj * view));
        let inside = Aabb { min: Vec3::splat(-0.5), max: Vec3::splat(0.5) };
        let behind = Aabb { min: Vec3::new(-0.5, -0.5, 6.0), max: Vec3::new(0.5, 0.5, 7.0) };
        let left = Aabb { min: Vec3::new(-100.0, -0.5, -0.5), max: Vec3::new(-90.0, 0.5, 0.5) };
        assert!(aabb_visible(&planes, &inside));
        assert!(!aabb_visible(&planes, &behind));
        assert!(!aabb_visible(&planes, &left));
    }
}
