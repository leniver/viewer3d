// Release builds are GUI applications (no console window on Windows).
#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

mod anim;
mod app;
mod camera;
mod gpu_select;
mod headless;
mod loaders;
mod model;
mod platform;
mod render;
mod scene;
mod settings;

use eframe::{egui, egui_wgpu, wgpu};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

// Ask hybrid-graphics drivers to run this program on the discrete GPU
// (NVIDIA Optimus / AMD PowerXpress). Exported from the executable by build.rs.
#[cfg(windows)]
#[allow(non_upper_case_globals)]
#[unsafe(no_mangle)]
#[used]
pub static NvOptimusEnablement: u32 = 1;

#[cfg(windows)]
#[allow(non_upper_case_globals)]
#[unsafe(no_mangle)]
#[used]
pub static AmdPowerXpressRequestHighPerformance: u32 = 1;

fn main() -> ExitCode {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn,viewer3d=info")).init();
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.first().map(String::as_str) == Some("--render") {
        attach_console();
        return match headless::run(&args[1..]) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("error: {e:#}");
                ExitCode::FAILURE
            }
        };
    }
    if args.iter().any(|a| a == "--help" || a == "-h") {
        attach_console();
        println!(
            "3D Viewer {}\n\nusage: viewer3d [FILE]\n       {}\n       viewer3d --register | --unregister   (Windows \"Open with\" entry)",
            env!("CARGO_PKG_VERSION"),
            headless::USAGE
        );
        return ExitCode::SUCCESS;
    }
    if let Some(flag) = args.first().filter(|a| *a == "--register" || *a == "--unregister") {
        attach_console();
        let result = if flag == "--register" { platform::register() } else { platform::unregister() };
        return match result {
            Ok(()) => {
                println!("done");
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        };
    }
    let initial = args.iter().find(|a| !a.starts_with("--")).map(PathBuf::from);

    match run_gui(initial) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            log::error!("{e}");
            ExitCode::FAILURE
        }
    }
}

fn run_gui(initial: Option<PathBuf>) -> eframe::Result {
    let mut setup = egui_wgpu::WgpuSetupCreateNew::without_display_handle();
    setup.instance_descriptor.backends = gpu_select::backends();
    setup.power_preference = wgpu::PowerPreference::HighPerformance;
    setup.native_adapter_selector = Some(Arc::new(|adapters: &[wgpu::Adapter], surface: Option<&wgpu::Surface<'_>>| {
        gpu_select::pick(adapters, surface)
    }));
    setup.device_descriptor = Arc::new(|adapter: &wgpu::Adapter| gpu_select::device_descriptor(adapter));
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("3D Viewer")
            .with_app_id("viewer3d")
            .with_icon(window_icon())
            .with_inner_size([1280.0, 820.0])
            .with_min_inner_size([520.0, 360.0])
            .with_drag_and_drop(true),
        renderer: eframe::Renderer::Wgpu,
        wgpu_options: egui_wgpu::WgpuConfiguration {
            wgpu_setup: egui_wgpu::WgpuSetup::CreateNew(setup),
            ..Default::default()
        },
        persist_window: true,
        ..Default::default()
    };
    eframe::run_native("viewer3d", options, Box::new(move |cc| Ok(Box::new(app::ViewerApp::new(cc, initial)))))
}

fn window_icon() -> egui::IconData {
    let rgba = image::load_from_memory(include_bytes!("../assets/icon.png"))
        .map(|i| i.into_rgba8())
        .unwrap_or_else(|_| image::RgbaImage::new(1, 1));
    let (width, height) = rgba.dimensions();
    egui::IconData { rgba: rgba.into_raw(), width, height }
}

/// Release builds have no console; reattach to the parent's one for CLI output.
fn attach_console() {
    #[cfg(all(windows, not(debug_assertions)))]
    {
        unsafe extern "system" {
            fn AttachConsole(process_id: u32) -> i32;
        }
        const ATTACH_PARENT_PROCESS: u32 = u32::MAX;
        // SAFETY: plain Win32 call without pointers.
        unsafe {
            AttachConsole(ATTACH_PARENT_PROCESS);
        }
    }
}
