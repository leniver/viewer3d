//! `--render <model> <image.png>`: renders a model to a PNG without a window
//! (thumbnails, batch checks, automated tests).

use crate::camera::Camera;
use crate::loaders::{LoadOptions, Progress};
use crate::model::load_model;
use crate::render::{DisplayMode, GpuContext, Renderer, ViewParams};
use crate::scene::UpAxis;
use crate::settings::{Background, LightingPreset};
use crate::{app, gpu_select};
use anyhow::{Context, Result, anyhow, bail};
use eframe::wgpu;
use std::path::PathBuf;

pub const USAGE: &str = "viewer3d --render <model> <output.png> [--size WxH] [--yaw DEG] [--pitch DEG]
                 [--wire | --shaded-wire] [--light] [--flat] [--no-grid] [--no-edges]
                 [--time SECONDS] [--anim NAME|INDEX] [--up y|z] [--blender PATH] [--verbose]";

struct Options {
    input: PathBuf,
    output: PathBuf,
    width: u32,
    height: u32,
    yaw: Option<f32>,
    pitch: Option<f32>,
    display: DisplayMode,
    background: Background,
    lighting: LightingPreset,
    grid: bool,
    edges: bool,
    time: Option<f32>,
    anim: Option<String>,
    up: Option<UpAxis>,
    blender: Option<PathBuf>,
    verbose: bool,
}

fn parse(args: &[String]) -> Result<Options> {
    let mut positional = Vec::new();
    let mut o = Options {
        input: PathBuf::new(),
        output: PathBuf::new(),
        width: 1280,
        height: 800,
        yaw: None,
        pitch: None,
        display: DisplayMode::Shaded,
        background: Background::Dark,
        lighting: LightingPreset::Studio,
        grid: true,
        edges: true,
        time: None,
        anim: None,
        up: None,
        blender: None,
        verbose: false,
    };
    let mut it = args.iter();
    let num = |v: Option<&String>, name: &str| -> Result<f32> {
        v.ok_or_else(|| anyhow!("{name} needs a value"))?.parse().map_err(|_| anyhow!("invalid value for {name}"))
    };
    while let Some(a) = it.next() {
        match a.as_str() {
            "--size" => {
                let v = it.next().ok_or_else(|| anyhow!("--size needs WxH"))?;
                let (w, h) = v.split_once(['x', 'X']).ok_or_else(|| anyhow!("--size needs WxH"))?;
                o.width = w.parse().context("invalid width")?;
                o.height = h.parse().context("invalid height")?;
            }
            "--yaw" => o.yaw = Some(num(it.next(), "--yaw")?.to_radians()),
            "--pitch" => o.pitch = Some(num(it.next(), "--pitch")?.to_radians()),
            "--time" => o.time = Some(num(it.next(), "--time")?),
            "--anim" => o.anim = Some(it.next().ok_or_else(|| anyhow!("--anim needs a name or an index"))?.clone()),
            "--blender" => o.blender = Some(PathBuf::from(it.next().ok_or_else(|| anyhow!("--blender needs a path"))?)),
            "--wire" => o.display = DisplayMode::Wireframe,
            "--shaded-wire" => o.display = DisplayMode::ShadedWireframe,
            "--light" => o.background = Background::Light,
            "--flat" => o.lighting = LightingPreset::Flat,
            "--no-grid" => o.grid = false,
            "--no-edges" => o.edges = false,
            "--verbose" | "-v" => o.verbose = true,
            "--up" => {
                o.up = Some(match it.next().map(|s| s.to_ascii_lowercase()).as_deref() {
                    Some("y") => UpAxis::Y,
                    Some("z") => UpAxis::Z,
                    _ => bail!("--up needs y or z"),
                })
            }
            other if other.starts_with("--") => bail!("unknown option {other}"),
            other => positional.push(PathBuf::from(other)),
        }
    }
    if positional.len() != 2 {
        bail!("usage: {USAGE}");
    }
    o.output = positional.pop().unwrap();
    o.input = positional.pop().unwrap();
    Ok(o)
}

pub fn run(args: &[String]) -> Result<()> {
    let o = parse(args)?;
    let instance = wgpu::Instance::new(gpu_select::instance_descriptor());
    let adapters = pollster::block_on(instance.enumerate_adapters(gpu_select::backends()));
    let adapter = gpu_select::pick(&adapters, None).map_err(|e| anyhow!(e))?;
    let (device, queue) = pollster::block_on(adapter.request_device(&gpu_select::device_descriptor(&adapter)))
        .context("cannot create the GPU device")?;
    let ctx = GpuContext::new(&adapter, device, queue);
    println!("GPU: {}", ctx.adapter_name);

    let progress = Progress::default();
    let options = LoadOptions { blender: o.blender.clone() };
    let mut model = load_model(&ctx, &o.input, &progress, &options, 1)?;
    if let Some(up) = o.up {
        model.set_up(&ctx, up);
    }
    if let Some(want) = &o.anim {
        let names: Vec<&str> = model.scene.animations.iter().map(|a| a.name.as_str()).collect();
        model.anim.current = names
            .iter()
            .position(|n| n == want)
            .or_else(|| want.parse::<usize>().ok().filter(|&i| i < names.len()))
            .ok_or_else(|| anyhow!("no animation '{want}' (the model has: {names:?})"))?;
        model.apply_pose(&ctx);
    }
    if let Some(t) = o.time {
        model.anim.time = t.clamp(0.0, model.duration());
        model.apply_pose(&ctx);
    }

    let aspect = o.width as f32 / o.height as f32;
    let mut camera = Camera::default();
    if let Some(y) = o.yaw {
        camera.yaw = y;
    }
    if let Some(p) = o.pitch {
        camera.pitch = p.clamp(-1.55, 1.55);
    }
    camera.frame(&model.world_bounds(), aspect);
    let (right, up, forward) = camera.basis();
    let (bg_top, bg_bottom) = o.background.colors();
    let params = ViewParams {
        view_proj: camera.proj(aspect) * camera.view(),
        eye: camera.eye(),
        width: o.width,
        height: o.height,
        lighting: o.lighting.lighting(right, up, forward),
        bg_top,
        bg_bottom,
        grid: o.grid.then(|| app::grid_for(&model.world_bounds(), o.background.grid_color())),
        display: o.display,
        show_edges: o.edges,
        wire_color: o.background.wire_color(o.display == DisplayMode::Wireframe),
        point_size: 3.0,
    };
    let mut renderer = Renderer::new(ctx.clone());
    renderer.render(Some(&mut model.gpu), &params);
    let (w, h, pixels) = renderer.read_pixels().ok_or_else(|| anyhow!("cannot read the rendered image"))?;
    image::save_buffer(&o.output, &pixels, w, h, image::ColorType::Rgba8)
        .with_context(|| format!("cannot write '{}'", o.output.display()))?;

    let s = &model.stats;
    let format = match &model.scene.converted_with {
        Some(by) => format!("{} (converted with {by})", model.format.name()),
        None => model.format.name().to_owned(),
    };
    println!(
        "{}: {} | {} triangles, {} vertices, {} points, {} materials, {} textures, {} animations | loaded in {:.2}s",
        model.name(),
        format,
        s.triangles,
        s.vertices,
        s.points,
        s.materials,
        s.textures.len(),
        s.animations.len(),
        model.load_time.as_secs_f32()
    );
    for w in &model.warnings {
        println!("warning: {w}");
    }
    if o.verbose {
        print_details(&model);
    }
    Ok(())
}

/// Animation and morph target summary (`--verbose`), for diagnosing files.
fn print_details(model: &crate::model::Model) {
    use crate::scene::Property;
    let targets: usize = model
        .scene
        .meshes
        .iter()
        .map(|m| m.primitives.iter().map(|p| p.targets.len()).max().unwrap_or(0))
        .sum();
    println!("morph targets (sum over meshes): {targets}, skins: {}", model.scene.skins.len());
    let roots: Vec<&str> = model.scene.roots.iter().take(8).map(|&r| model.scene.nodes[r].name.as_str()).collect();
    println!("nodes: {}, top level: {roots:?}", model.scene.nodes.len());
    for a in &model.scene.animations {
        let count = |f: &dyn Fn(Property) -> bool| a.channels.iter().filter(|c| f(c.property)).count();
        println!(
            "animation '{}': {:.2}s, {} channels (T {}, R {}, S {}, morph {})",
            a.name,
            a.duration,
            a.channels.len(),
            count(&|p| p == Property::Translation),
            count(&|p| p == Property::Rotation),
            count(&|p| p == Property::Scale),
            count(&|p| matches!(p, Property::MorphWeight(_))),
        );
    }
}
