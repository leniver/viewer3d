//! GPU resources shared by the UI thread and the loader thread (wgpu objects
//! are Send + Sync, so models are uploaded off the UI thread).

use crate::scene::{Sampler as SceneSampler, Wrap};
use eframe::wgpu;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
pub const SAMPLES: u32 = 4;

pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub frame_layout: wgpu::BindGroupLayout,
    pub material_layout: wgpu::BindGroupLayout,
    pub draw_layout: wgpu::BindGroupLayout,
    pub scene_shader: wgpu::ShaderModule,
    pub white: wgpu::TextureView,
    pub default_sampler: Arc<wgpu::Sampler>,
    samplers: Mutex<HashMap<SceneSampler, Arc<wgpu::Sampler>>>,
    pub mips: MipGenerator,
    /// None when the adapter cannot run compute shaders with storage buffers.
    pub compute: Option<ComputeKernels>,
    /// Stride between per-draw uniforms (dynamic offsets).
    pub draw_stride: u64,
    pub max_texture_size: u32,
    pub adapter_name: String,
}

pub struct ComputeKernels {
    pub deform_layout: wgpu::BindGroupLayout,
    pub deform: wgpu::ComputePipeline,
    pub wire_layout: wgpu::BindGroupLayout,
    pub wire: wgpu::ComputePipeline,
    pub max_workgroups: u32,
}

/// Size in bytes of the per-draw uniform block (see `DrawU` in scene.wgsl).
pub const DRAW_UNIFORM_SIZE: u64 = 128;

fn uniform_entry(binding: u32, visibility: wgpu::ShaderStages, dynamic: bool, size: u64) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: dynamic,
            min_binding_size: wgpu::BufferSize::new(size),
        },
        count: None,
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

impl GpuContext {
    pub fn new(adapter: &wgpu::Adapter, device: wgpu::Device, queue: wgpu::Queue) -> Arc<Self> {
        // Log GPU validation errors instead of the default panic: a bad model
        // must never take the whole application down.
        device.on_uncaptured_error(Arc::new(|e: wgpu::Error| log::error!("GPU error: {e}")));
        let limits = device.limits();
        let frame_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("frame"),
            entries: &[uniform_entry(0, wgpu::ShaderStages::VERTEX_FRAGMENT, false, 368)],
        });
        let mut material_entries = vec![uniform_entry(0, wgpu::ShaderStages::FRAGMENT, false, 272)];
        material_entries.extend((1..=6).map(texture_entry));
        material_entries.extend((7..=12).map(sampler_entry));
        let material_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("material"),
            entries: &material_entries,
        });
        let draw_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("draw"),
            entries: &[uniform_entry(0, wgpu::ShaderStages::VERTEX_FRAGMENT, true, DRAW_UNIFORM_SIZE)],
        });
        let scene_shader = device.create_shader_module(wgpu::include_wgsl!("shaders/scene.wgsl"));

        let white_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("white"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            white_tex.as_image_copy(),
            &[255, 255, 255, 255],
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4), rows_per_image: Some(1) },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        let white = white_tex.create_view(&Default::default());

        let downlevel = adapter.get_downlevel_capabilities();
        let compute_ok = downlevel.flags.contains(wgpu::DownlevelFlags::COMPUTE_SHADERS)
            && limits.max_storage_buffers_per_shader_stage >= 6
            && limits.max_compute_workgroup_size_x >= 64;
        let compute = compute_ok.then(|| ComputeKernels::new(&device, limits.max_compute_workgroups_per_dimension));

        let align = limits.min_uniform_buffer_offset_alignment as u64;
        let draw_stride = DRAW_UNIFORM_SIZE.div_ceil(align) * align;
        let mips = MipGenerator::new(&device);
        let default_sampler = Arc::new(create_sampler(&device, &SceneSampler::default()));

        Arc::new(Self {
            device,
            queue,
            frame_layout,
            material_layout,
            draw_layout,
            scene_shader,
            white,
            default_sampler,
            samplers: Mutex::new(HashMap::new()),
            mips,
            compute,
            draw_stride,
            max_texture_size: limits.max_texture_dimension_2d,
            adapter_name: {
                let info = adapter.get_info();
                format!("{} ({:?})", info.name, info.backend)
            },
        })
    }

    pub fn sampler(&self, s: &SceneSampler) -> Arc<wgpu::Sampler> {
        let mut cache = self.samplers.lock().unwrap_or_else(|e| e.into_inner());
        cache.entry(*s).or_insert_with(|| Arc::new(create_sampler(&self.device, s))).clone()
    }
}

fn create_sampler(device: &wgpu::Device, s: &SceneSampler) -> wgpu::Sampler {
    let address = |w: Wrap| match w {
        Wrap::Repeat => wgpu::AddressMode::Repeat,
        Wrap::Mirror => wgpu::AddressMode::MirrorRepeat,
        Wrap::Clamp => wgpu::AddressMode::ClampToEdge,
    };
    let linear = !s.mag_nearest && !s.min_nearest;
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("texture sampler"),
        address_mode_u: address(s.wrap_u),
        address_mode_v: address(s.wrap_v),
        address_mode_w: wgpu::AddressMode::Repeat,
        mag_filter: if s.mag_nearest { wgpu::FilterMode::Nearest } else { wgpu::FilterMode::Linear },
        min_filter: if s.min_nearest { wgpu::FilterMode::Nearest } else { wgpu::FilterMode::Linear },
        mipmap_filter: if linear { wgpu::MipmapFilterMode::Linear } else { wgpu::MipmapFilterMode::Nearest },
        anisotropy_clamp: if linear { 16 } else { 1 },
        ..Default::default()
    })
}

impl ComputeKernels {
    fn new(device: &wgpu::Device, max_workgroups: u32) -> Self {
        let deform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("deform"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(16),
                    },
                    count: None,
                },
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, true),
                storage_entry(5, true),
                storage_entry(6, false),
            ],
        });
        let wire_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("wire"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(16),
                    },
                    count: None,
                },
                storage_entry(1, true),
                storage_entry(2, false),
            ],
        });
        let pipeline = |layout: &wgpu::BindGroupLayout, shader: wgpu::ShaderModuleDescriptor, entry: &str| {
            let module = device.create_shader_module(shader);
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(entry),
                bind_group_layouts: &[Some(layout)],
                immediate_size: 0,
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pl),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let deform = pipeline(&deform_layout, wgpu::include_wgsl!("shaders/deform.wgsl"), "deform");
        let wire = pipeline(&wire_layout, wgpu::include_wgsl!("shaders/wire.wgsl"), "wire");
        Self { deform_layout, deform, wire_layout, wire, max_workgroups: max_workgroups.max(1) }
    }

    /// 2D dispatch size for `count` invocations of 64 threads, and the number
    /// of invocations per row.
    pub fn dispatch_size(&self, count: u32) -> (u32, u32, u32) {
        let groups = count.div_ceil(64).max(1);
        let x = groups.min(self.max_workgroups);
        let y = groups.div_ceil(x);
        (x, y, x * 64)
    }
}

/// Generates mip chains on the GPU with a chain of downsampling passes.
pub struct MipGenerator {
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    pipelines: HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,
}

impl MipGenerator {
    fn new(device: &wgpu::Device) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mipmap"),
            entries: &[texture_entry(0), sampler_entry(1)],
        });
        let module = device.create_shader_module(wgpu::include_wgsl!("shaders/mipmap.wgsl"));
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mipmap"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let mut pipelines = HashMap::new();
        for format in [wgpu::TextureFormat::Rgba8Unorm, wgpu::TextureFormat::Rgba8UnormSrgb] {
            let p = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("mipmap"),
                layout: Some(&pl),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(format.into())],
                }),
                multiview_mask: None,
                cache: None,
            });
            pipelines.insert(format, p);
        }
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("mipmap"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self { layout, sampler, pipelines }
    }

    pub fn generate(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, texture: &wgpu::Texture) {
        let Some(pipeline) = self.pipelines.get(&texture.format()) else { return };
        let levels = texture.mip_level_count();
        for level in 1..levels {
            let view = |l: u32| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    base_mip_level: l,
                    mip_level_count: Some(1),
                    ..Default::default()
                })
            };
            let src = view(level - 1);
            let dst = view(level);
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("mipmap"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                ],
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("mipmap"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &dst,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}

pub fn mip_count(width: u32, height: u32) -> u32 {
    32 - width.max(height).max(1).leading_zeros()
}
