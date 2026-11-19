//! A model ready to display (CPU pose data + GPU resources) and the
//! background loading job that produces it.

use crate::anim::{Animator, Pose};
use crate::loaders::{self, BlenderNotFound, Cancelled, Format, LoadOptions, Progress, Stats};
use crate::render::{GpuContext, GpuScene};
use crate::scene::{Aabb, Scene, UpAxis};
use glam::Mat4;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, TryRecvError, channel};
use std::time::{Duration, Instant};

pub struct AnimState {
    pub current: usize,
    pub time: f32,
    pub playing: bool,
    pub speed: f32,
}

pub struct Model {
    pub path: PathBuf,
    pub scene: Scene,
    pub gpu: GpuScene,
    pub animator: Animator,
    pub pose: Pose,
    pub format: Format,
    pub stats: Stats,
    pub warnings: Vec<String>,
    pub file_size: u64,
    pub load_time: Duration,
    /// Rest-pose bounds in file coordinates.
    pub bounds: Aabb,
    pub up: UpAxis,
    pub anim: AnimState,
    pub generation: u64,
}

/// Rotation that brings the file's up axis to +Y.
pub fn root_matrix(up: UpAxis) -> Mat4 {
    match up {
        UpAxis::Y => Mat4::IDENTITY,
        UpAxis::Z => Mat4::from_rotation_x(-std::f32::consts::FRAC_PI_2),
    }
}

impl Model {
    pub fn root(&self) -> Mat4 {
        root_matrix(self.up)
    }

    pub fn world_bounds(&self) -> Aabb {
        self.bounds.transformed(&self.root())
    }

    pub fn name(&self) -> String {
        self.path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
    }

    pub fn has_animation(&self) -> bool {
        !self.scene.animations.is_empty()
    }

    pub fn set_up(&mut self, ctx: &GpuContext, up: UpAxis) {
        self.up = up;
        self.apply_pose(ctx);
    }

    /// Evaluates the current animation time and uploads the pose.
    pub fn apply_pose(&mut self, ctx: &GpuContext) {
        let anim = self.scene.animations.get(self.anim.current);
        self.animator.evaluate(anim, self.anim.time, &mut self.pose);
        let root = self.root();
        self.gpu.apply_pose(ctx, &self.pose, root);
    }

    pub fn duration(&self) -> f32 {
        self.scene.animations.get(self.anim.current).map_or(0.0, |a| a.duration)
    }

    /// Advances playback by `dt` seconds.
    pub fn advance(&mut self, ctx: &GpuContext, dt: f32, looped: bool) {
        let duration = self.duration();
        if duration <= 0.0 {
            self.anim.playing = false;
            return;
        }
        self.anim.time += dt * self.anim.speed;
        if looped {
            self.anim.time = self.anim.time.rem_euclid(duration);
        } else if self.anim.time >= duration {
            self.anim.time = duration;
            self.anim.playing = false;
        }
        self.apply_pose(ctx);
    }
}

/// Why a model could not be loaded.
#[derive(Debug)]
pub enum LoadError {
    Cancelled,
    /// A .blend file and no usable Blender: the user can install or choose one.
    NeedsBlender(String),
    Failed(String),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Cancelled => f.write_str("loading cancelled"),
            LoadError::NeedsBlender(msg) | LoadError::Failed(msg) => f.write_str(msg),
        }
    }
}

impl std::error::Error for LoadError {}

/// Parses a file and uploads it to the GPU (runs on a worker thread).
pub fn load_model(
    ctx: &Arc<GpuContext>,
    path: &std::path::Path,
    progress: &Progress,
    options: &LoadOptions,
    generation: u64,
) -> Result<Model, LoadError> {
    crate::platform::memlog("start");
    let loaded = loaders::load_file(path, progress, options).map_err(|e| {
        if e.downcast_ref::<Cancelled>().is_some() {
            LoadError::Cancelled
        } else if let Some(missing) = e.downcast_ref::<BlenderNotFound>() {
            LoadError::NeedsBlender(missing.to_string())
        } else {
            LoadError::Failed(format!("{e:#}"))
        }
    })?;
    crate::platform::memlog("parsing and decoding");
    progress.set_stage("Uploading to GPU");
    progress.set(0.97);
    let mut scene = loaded.scene;
    let up = scene.up_axis;
    let animator = Animator::new(&scene);
    let upload = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| GpuScene::upload(ctx, &mut scene, root_matrix(up))));
    let gpu = upload.map_err(|_| LoadError::Failed("The model could not be uploaded to the GPU".to_owned()))?;
    crate::platform::memlog("GPU upload");
    // Wait (on this worker thread) for the upload to finish so wgpu releases
    // its staging memory now; an idle UI would otherwise keep it allocated.
    let _ = ctx.device.poll(eframe::wgpu::PollType::Wait { submission_index: None, timeout: Some(Duration::from_secs(10)) });
    crate::platform::memlog("GPU wait");
    let pose = animator.rest_pose();
    Ok(Model {
        path: path.to_path_buf(),
        scene,
        gpu,
        animator,
        pose,
        format: loaded.format,
        stats: loaded.stats,
        warnings: loaded.warnings,
        file_size: loaded.file_size,
        load_time: loaded.load_time,
        bounds: loaded.bounds,
        up,
        // Animations wait for the user (Space or the play button).
        anim: AnimState { current: 0, time: 0.0, playing: false, speed: 1.0 },
        generation,
    })
}

pub enum LoadOutcome {
    Pending,
    Done(Box<Model>),
    Failed(LoadError),
}

pub struct LoadJob {
    pub path: PathBuf,
    pub progress: Arc<Progress>,
    pub started: Instant,
    rx: Receiver<Result<Model, LoadError>>,
}

impl LoadJob {
    pub fn spawn(
        ctx: Arc<GpuContext>,
        path: PathBuf,
        options: LoadOptions,
        generation: u64,
        repaint: impl Fn() + Send + 'static,
    ) -> LoadJob {
        let progress = Arc::new(Progress::default());
        let (tx, rx) = channel();
        let job_path = path.clone();
        let job_progress = progress.clone();
        let spawned = std::thread::Builder::new()
            .name("loader".into())
            // OpenCASCADE and some parsers recurse deeply on large assemblies.
            .stack_size(64 * 1024 * 1024)
            .spawn(move || {
                let result = load_model(&ctx, &job_path, &job_progress, &options, generation);
                let _ = tx.send(result);
                repaint();
            });
        if let Err(e) = spawned {
            log::error!("cannot start loader thread: {e}");
        }
        LoadJob { path, progress, started: Instant::now(), rx }
    }

    pub fn poll(&self) -> LoadOutcome {
        match self.rx.try_recv() {
            Ok(Ok(model)) => LoadOutcome::Done(Box::new(model)),
            Ok(Err(e)) => LoadOutcome::Failed(e),
            Err(TryRecvError::Empty) => LoadOutcome::Pending,
            Err(TryRecvError::Disconnected) => LoadOutcome::Failed(LoadError::Failed("The loader stopped unexpectedly".into())),
        }
    }
}
