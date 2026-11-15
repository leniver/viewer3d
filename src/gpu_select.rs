//! GPU adapter selection: prefer the discrete (NVIDIA first) GPU over an
//! integrated one, and request everything the adapter supports.

use eframe::wgpu;

const VENDOR_NVIDIA: u32 = 0x10DE;
const VENDOR_AMD: u32 = 0x1002;

fn score(info: &wgpu::AdapterInfo) -> i32 {
    let mut s = match info.device_type {
        wgpu::DeviceType::DiscreteGpu => 100,
        wgpu::DeviceType::IntegratedGpu => 50,
        wgpu::DeviceType::VirtualGpu => 30,
        wgpu::DeviceType::Cpu => 10,
        wgpu::DeviceType::Other => 0,
    };
    s += match info.vendor {
        VENDOR_NVIDIA => 20,
        VENDOR_AMD => 10,
        _ => 0,
    };
    s += match info.backend {
        wgpu::Backend::Dx12 if cfg!(windows) => 6,
        wgpu::Backend::Vulkan => 5,
        wgpu::Backend::Metal => 6,
        wgpu::Backend::Gl => -40,
        _ => 0,
    };
    s
}

/// Picks the best adapter, optionally restricted to those that can present to `surface`.
pub fn pick(adapters: &[wgpu::Adapter], surface: Option<&wgpu::Surface<'_>>) -> Result<wgpu::Adapter, String> {
    adapters
        .iter()
        .filter(|a| surface.is_none_or(|s| a.is_surface_supported(s)))
        .max_by_key(|a| score(&a.get_info()))
        .cloned()
        .ok_or_else(|| "no compatible GPU adapter found".to_owned())
}

pub fn device_descriptor(adapter: &wgpu::Adapter) -> wgpu::DeviceDescriptor<'static> {
    let max = adapter.limits();
    let base = if adapter.get_info().backend == wgpu::Backend::Gl {
        wgpu::Limits::downlevel_webgl2_defaults()
    } else {
        wgpu::Limits::default()
    };
    let full = std::env::var_os("VIEWER3D_FULL_LIMITS").is_some();
    wgpu::DeviceDescriptor {
        label: Some("viewer3d device"),
        required_features: wgpu::Features::empty(),
        // Defaults, raised only where big models need more (huge textures,
        // vertex buffers and point clouds, compute over large meshes).
        required_limits: if full {
            max
        } else {
            wgpu::Limits {
                max_texture_dimension_2d: max.max_texture_dimension_2d.min(16384),
                max_buffer_size: max.max_buffer_size,
                max_storage_buffer_binding_size: max.max_storage_buffer_binding_size,
                max_storage_buffers_per_shader_stage: max.max_storage_buffers_per_shader_stage.min(8),
                max_compute_workgroups_per_dimension: max.max_compute_workgroups_per_dimension,
                ..base
            }
        },
        ..Default::default()
    }
}

/// Backends to initialise. Each backend loads its driver into the process
/// (tens of MB), so only the native one is used: DirectX 12 on Windows (with
/// the WARP software fallback), Vulkan with an OpenGL fallback elsewhere.
pub fn backends() -> wgpu::Backends {
    wgpu::Backends::from_env().unwrap_or(if cfg!(windows) {
        wgpu::Backends::DX12
    } else if cfg!(target_os = "macos") {
        wgpu::Backends::METAL
    } else {
        wgpu::Backends::VULKAN | wgpu::Backends::GL
    })
}

pub fn instance_descriptor() -> wgpu::InstanceDescriptor {
    wgpu::InstanceDescriptor {
        backends: backends(),
        flags: wgpu::InstanceFlags::from_build_config().with_env(),
        backend_options: wgpu::BackendOptions::from_env_or_default(),
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        display: None,
    }
}
