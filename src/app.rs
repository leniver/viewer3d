//! The application window: toolbar, viewport, animation bar and info panel.

use crate::camera::Camera;
use crate::loaders::{self, LoadOptions, SUPPORTED_EXTENSIONS};
use crate::model::{LoadError, LoadJob, LoadOutcome, Model};
use crate::render::{DisplayMode, GpuContext, Grid, Renderer, ViewParams};
use crate::scene::{Aabb, UpAxis};
use crate::settings::{Background, LightingPreset, Settings};
use eframe::egui::{self, Align2, Color32, FontId, Key, Modifiers, PointerButton, Pos2, Rect, Sense, Stroke, Vec2};
use eframe::wgpu;
use glam::Vec3;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, channel};
use std::time::{Duration, Instant};

/// Frame interval while an animation plays (60 frames per second).
const ANIMATION_FRAME: Duration = Duration::from_micros(16_667);

enum DialogResult {
    Open(Option<PathBuf>),
    Screenshot(Option<PathBuf>),
    Blender(Option<PathBuf>),
}

/// Inputs of the last rendered frame; the 3D view is only redrawn when they change.
#[derive(Clone, Copy, PartialEq)]
struct RenderKey {
    width: u32,
    height: u32,
    camera: [f32; 7],
    display: DisplayMode,
    edges: bool,
    grid: bool,
    background: Background,
    lighting: LightingPreset,
    point_size: u32,
    generation: u64,
    anim_time: u32,
    up: Option<UpAxis>,
}

pub struct ViewerApp {
    ctx: Arc<GpuContext>,
    renderer: Renderer,
    texture: Option<egui::TextureId>,
    texture_generation: u64,
    settings: Settings,
    model: Option<Model>,
    loading: Option<LoadJob>,
    camera: Camera,
    error: Option<String>,
    /// The error is a missing Blender: the banner offers to choose one.
    error_needs_blender: bool,
    /// File whose loading failed, opened again once Blender has been chosen.
    failed_path: Option<PathBuf>,
    /// Blender used for .blend files, as shown in the `?` menu (refreshed
    /// every few seconds while the menu is open: Blender may be installed meanwhile).
    blender_status: Option<(Instant, String)>,
    last_render: Option<RenderKey>,
    generation: u64,
    dialog: Option<Receiver<DialogResult>>,
    title: String,
    reset_view: bool,
    /// Frame-rate logging, enabled with the VIEWER3D_FPS environment variable.
    fps: Option<FpsLog>,
    /// Scheduled time of the next animation frame.
    next_anim_frame: Option<std::time::Instant>,
}

struct FpsLog {
    frames: u32,
    since: std::time::Instant,
}

impl FpsLog {
    fn tick(&mut self) {
        self.frames += 1;
        let elapsed = self.since.elapsed();
        if elapsed >= Duration::from_secs(1) {
            log::info!("{:.1} frames/s", self.frames as f64 / elapsed.as_secs_f64());
            self.frames = 0;
            self.since = std::time::Instant::now();
        }
    }
}

impl ViewerApp {
    pub fn new(cc: &eframe::CreationContext<'_>, initial: Option<PathBuf>) -> Self {
        let rs = cc.wgpu_render_state.as_ref().expect("the wgpu renderer is required");
        let ctx = GpuContext::new(&rs.adapter, rs.device.clone(), rs.queue.clone());
        log::info!("GPU: {}", ctx.adapter_name);
        let settings: Settings = cc.storage.and_then(|s| eframe::get_value(s, eframe::APP_KEY)).unwrap_or_default();
        let mut app = Self {
            renderer: Renderer::new(ctx.clone()),
            ctx,
            texture: None,
            texture_generation: 0,
            settings,
            model: None,
            loading: None,
            camera: Camera::default(),
            error: None,
            error_needs_blender: false,
            failed_path: None,
            blender_status: None,
            last_render: None,
            generation: 0,
            dialog: None,
            title: String::new(),
            reset_view: false,
            fps: std::env::var_os("VIEWER3D_FPS").map(|_| FpsLog { frames: 0, since: std::time::Instant::now() }),
            next_anim_frame: None,
        };
        if let Some(path) = initial {
            app.open(path, &cc.egui_ctx);
        }
        app
    }

    fn open(&mut self, path: PathBuf, ectx: &egui::Context) {
        if let Some(job) = self.loading.take() {
            job.progress.cancel();
        }
        self.error = None;
        self.error_needs_blender = false;
        self.generation += 1;
        let repaint_ctx = ectx.clone();
        let options = LoadOptions { blender: self.settings.blender_path.clone() };
        self.loading =
            Some(LoadJob::spawn(self.ctx.clone(), path, options, self.generation, move || repaint_ctx.request_repaint()));
    }

    fn poll_loading(&mut self) {
        let Some(job) = &self.loading else { return };
        match job.poll() {
            LoadOutcome::Pending => {}
            LoadOutcome::Done(model) => {
                log::info!(
                    "loaded {} in {:.2}s ({} triangles)",
                    model.path.display(),
                    model.load_time.as_secs_f32(),
                    model.stats.triangles
                );
                self.settings.add_recent(model.path.clone());
                self.model = Some(*model);
                self.loading = None;
                self.failed_path = None;
                self.reset_view = true;
            }
            LoadOutcome::Failed(LoadError::Cancelled) => self.loading = None,
            LoadOutcome::Failed(e) => {
                let name = job.path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                self.error = Some(format!("Cannot open {name}: {e}"));
                self.error_needs_blender = matches!(e, LoadError::NeedsBlender(_));
                self.failed_path = Some(job.path.clone());
                self.loading = None;
            }
        }
    }

    fn open_dialog(&mut self) {
        if self.dialog.is_some() {
            return;
        }
        let start = self.current_dir();
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            let mut d = rfd::FileDialog::new()
                .set_title("Open a 3D model")
                .add_filter("3D models", SUPPORTED_EXTENSIONS)
                .add_filter("All files", &["*"]);
            if let Some(dir) = start {
                d = d.set_directory(dir);
            }
            let _ = tx.send(DialogResult::Open(d.pick_file()));
        });
        self.dialog = Some(rx);
    }

    fn screenshot_dialog(&mut self) {
        if self.dialog.is_some() || self.model.is_none() {
            return;
        }
        let name = self
            .model
            .as_ref()
            .and_then(|m| m.path.file_stem())
            .map(|s| format!("{}.png", s.to_string_lossy()))
            .unwrap_or_else(|| "screenshot.png".into());
        let start = self.current_dir();
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            let mut d = rfd::FileDialog::new().set_title("Save image").add_filter("PNG image", &["png"]).set_file_name(name);
            if let Some(dir) = start {
                d = d.set_directory(dir);
            }
            let _ = tx.send(DialogResult::Screenshot(d.save_file()));
        });
        self.dialog = Some(rx);
    }

    fn blender_dialog(&mut self) {
        if self.dialog.is_some() {
            return;
        }
        let start = self
            .settings
            .blender_path
            .as_ref()
            .and_then(|p| p.parent().map(Path::to_path_buf))
            .or_else(|| loaders::find_blender(None).ok().and_then(|b| b.path.parent().map(Path::to_path_buf)));
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            let mut d = rfd::FileDialog::new().set_title("Choose the Blender program (blender.exe)");
            if cfg!(windows) {
                d = d.add_filter("Blender", &["exe"]);
            }
            if let Some(dir) = start {
                d = d.set_directory(dir);
            }
            let _ = tx.send(DialogResult::Blender(d.pick_file()));
        });
        self.dialog = Some(rx);
    }

    /// Sets the Blender for .blend files (None: found automatically) and opens
    /// again a .blend file that failed for want of Blender.
    fn set_blender(&mut self, path: Option<PathBuf>, ectx: &egui::Context) {
        if let Some(p) = &path
            && !loaders::is_blender_program(p)
        {
            self.error = Some(format!("{} is not the Blender program (blender.exe)", p.display()));
            self.error_needs_blender = true;
            return;
        }
        self.settings.blender_path = path.map(|p| loaders::describe_blender(&p).map_or(p, |b| b.path));
        self.blender_status = None;
        let is_blend = |p: &PathBuf| {
            p.extension().and_then(|e| e.to_str()).and_then(loaders::Format::from_extension) == Some(loaders::Format::Blend)
        };
        if let Some(p) = self.failed_path.take().filter(is_blend) {
            self.open(p, ectx);
        } else if self.error_needs_blender {
            self.error = None;
            self.error_needs_blender = false;
        }
    }

    /// One-line description of the Blender used for .blend files.
    fn blender_status(&mut self) -> String {
        let fresh = self.blender_status.as_ref().is_some_and(|(at, _)| at.elapsed() < Duration::from_secs(3));
        if !fresh {
            let text = match loaders::find_blender(self.settings.blender_path.as_deref()) {
                Ok(b) => format!("{} — {}", b.label(), b.path.display()),
                Err(e) => e.to_string(),
            };
            self.blender_status = Some((Instant::now(), text));
        }
        self.blender_status.as_ref().map(|(_, t)| t.clone()).unwrap_or_default()
    }

    fn current_dir(&self) -> Option<PathBuf> {
        self.model
            .as_ref()
            .map(|m| m.path.clone())
            .or_else(|| self.settings.recent.first().cloned())
            .and_then(|p| p.parent().map(Path::to_path_buf))
    }

    fn poll_dialog(&mut self, ectx: &egui::Context) {
        let Some(rx) = &self.dialog else { return };
        match rx.try_recv() {
            Ok(DialogResult::Open(Some(path))) => {
                self.dialog = None;
                self.open(path, ectx);
            }
            Ok(DialogResult::Screenshot(Some(path))) => {
                self.dialog = None;
                self.save_screenshot(&path);
            }
            Ok(DialogResult::Blender(Some(path))) => {
                self.dialog = None;
                self.set_blender(Some(path), ectx);
            }
            Ok(_) => self.dialog = None,
            Err(std::sync::mpsc::TryRecvError::Empty) => ectx.request_repaint_after(Duration::from_millis(100)),
            Err(std::sync::mpsc::TryRecvError::Disconnected) => self.dialog = None,
        }
    }

    fn save_screenshot(&mut self, path: &Path) {
        let path = if path.extension().is_none() { path.with_extension("png") } else { path.to_path_buf() };
        let result = self
            .renderer
            .read_pixels()
            .ok_or_else(|| "nothing rendered yet".to_owned())
            .and_then(|(w, h, px)| image::save_buffer(&path, &px, w, h, image::ColorType::Rgba8).map_err(|e| e.to_string()));
        if let Err(e) = result {
            self.error = Some(format!("Cannot save the image: {e}"));
        }
    }

    /// Next/previous supported file in the same folder.
    fn open_sibling(&mut self, delta: isize, ectx: &egui::Context) {
        let Some(current) = self.model.as_ref().map(|m| m.path.clone()).or_else(|| self.loading.as_ref().map(|l| l.path.clone())) else {
            return;
        };
        let Some(dir) = current.parent() else { return };
        let Ok(entries) = std::fs::read_dir(dir) else { return };
        let mut files: Vec<PathBuf> = entries.flatten().map(|e| e.path()).filter(|p| p.is_file() && loaders::is_supported(p)).collect();
        if files.len() < 2 {
            return;
        }
        files.sort_by_key(|p| p.file_name().map(|n| n.to_string_lossy().to_lowercase()));
        let idx = files.iter().position(|p| *p == current).unwrap_or(0) as isize;
        let next = (idx + delta).rem_euclid(files.len() as isize) as usize;
        self.open(files[next].clone(), ectx);
    }

    fn handle_input(&mut self, ectx: &egui::Context) {
        // Files dropped on the window.
        let dropped: Vec<PathBuf> = ectx.input(|i| i.raw.dropped_files.iter().map(|f| f.path().to_path_buf()).collect());
        if let Some(path) = dropped.into_iter().next() {
            self.open(path, ectx);
        }
        if ectx.memory(|m| m.focused().is_some()) {
            return;
        }
        let pressed = |key: Key| ectx.input_mut(|i| i.consume_key(Modifiers::NONE, key));
        if ectx.input_mut(|i| i.consume_key(Modifiers::COMMAND, Key::O)) {
            self.open_dialog();
        }
        if pressed(Key::Escape)
            && let Some(job) = &self.loading {
                job.progress.cancel();
            }
        if pressed(Key::PageDown) {
            self.open_sibling(1, ectx);
        }
        if pressed(Key::PageUp) {
            self.open_sibling(-1, ectx);
        }
        if pressed(Key::F) || pressed(Key::Home) {
            self.reset_view = true;
        }
        if pressed(Key::W) {
            self.settings.display = match self.settings.display {
                DisplayMode::Shaded => DisplayMode::ShadedWireframe,
                DisplayMode::ShadedWireframe => DisplayMode::Wireframe,
                DisplayMode::Wireframe => DisplayMode::Shaded,
            };
        }
        if pressed(Key::G) {
            self.settings.show_grid ^= true;
        }
        if pressed(Key::E) {
            self.settings.show_edges ^= true;
        }
        if pressed(Key::I) {
            self.settings.show_info ^= true;
        }
        if pressed(Key::Space)
            && let Some(m) = &mut self.model
                && m.has_animation() {
                    m.anim.playing ^= true;
                }
    }

    fn update_title(&mut self, ectx: &egui::Context) {
        let title = match (&self.loading, &self.model) {
            (Some(job), _) => format!(
                "Loading {} — 3D Viewer",
                job.path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
            ),
            (None, Some(m)) => format!("{} — 3D Viewer", m.name()),
            (None, None) => "3D Viewer".to_owned(),
        };
        if title != self.title {
            ectx.send_viewport_cmd(egui::ViewportCommand::Title(title.clone()));
            self.title = title;
        }
    }

    fn toolbar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing.x = 6.0;
            if ui.button("📂 Open…").on_hover_text("Open a 3D file (Ctrl+O)").clicked() {
                self.open_dialog();
            }
            let mut to_open = None;
            ui.menu_button("Recent", |ui| {
                if self.settings.recent.is_empty() {
                    ui.label("No recent files");
                }
                for p in &self.settings.recent {
                    let name = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                    if ui.button(name).on_hover_text(p.display().to_string()).clicked() {
                        to_open = Some(p.clone());
                        ui.close();
                    }
                }
                if !self.settings.recent.is_empty() {
                    ui.separator();
                    if ui.button("Clear list").clicked() {
                        self.settings.recent.clear();
                        ui.close();
                    }
                }
            });
            if let Some(p) = to_open {
                let ectx = ui.ctx().clone();
                self.open(p, &ectx);
            }
            ui.separator();

            let has_model = self.model.is_some();
            ui.add_enabled_ui(has_model, |ui| {
                if ui.button("Reset view").on_hover_text("Frame the model (F, Home or double-click)").clicked() {
                    self.reset_view = true;
                }
                egui::ComboBox::from_id_salt("display")
                    .selected_text(display_label(self.settings.display))
                    .show_ui(ui, |ui| {
                        for mode in [DisplayMode::Shaded, DisplayMode::ShadedWireframe, DisplayMode::Wireframe] {
                            ui.selectable_value(&mut self.settings.display, mode, display_label(mode));
                        }
                    })
                    .response
                    .on_hover_text("Display mode (W)");
                if self.model.as_ref().is_some_and(|m| m.gpu.has_feature_edges) {
                    ui.toggle_value(&mut self.settings.show_edges, "Edges").on_hover_text("CAD edges (E)");
                }
                ui.toggle_value(&mut self.settings.show_grid, "Grid").on_hover_text("Ground grid (G)");
                ui.label("Up:");
                let current_up = self.model.as_ref().map(|m| m.up);
                for (axis, label) in [(UpAxis::Y, "Y"), (UpAxis::Z, "Z")] {
                    if ui.selectable_label(current_up == Some(axis), label).on_hover_text("Vertical axis of the model").clicked()
                        && let Some(m) = &mut self.model
                            && m.up != axis {
                                m.set_up(&self.ctx, axis);
                                self.reset_view = true;
                            }
                }
                egui::ComboBox::from_id_salt("lighting")
                    .selected_text(self.settings.lighting.label())
                    .show_ui(ui, |ui| {
                        for l in LightingPreset::ALL {
                            ui.selectable_value(&mut self.settings.lighting, l, l.label());
                        }
                    });
            });
            egui::ComboBox::from_id_salt("background").selected_text(self.settings.background.label()).show_ui(ui, |ui| {
                for b in Background::ALL {
                    ui.selectable_value(&mut self.settings.background, b, b.label());
                }
            });
            if self.model.as_ref().is_some_and(|m| m.stats.points > 0) {
                ui.add(egui::Slider::new(&mut self.settings.point_size, 1.0..=12.0).text("Point size"));
            }

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.menu_button("?", |ui| {
                    ui.label(egui::RichText::new("Mouse").strong());
                    ui.label("Left drag: rotate\nRight / middle drag or Shift + left drag: pan\nWheel: zoom\nDouble-click: frame the model");
                    ui.separator();
                    ui.label(egui::RichText::new("Keyboard").strong());
                    ui.label(
                        "Ctrl+O: open\nPage Up / Page Down: previous / next file in the folder\nF or Home: frame\nW: display mode\nE: CAD edges\nG: grid\nI: information\nSpace: play / pause\nEsc: cancel loading",
                    );
                    if cfg!(windows) {
                        ui.separator();
                        ui.label(egui::RichText::new("Windows").strong());
                        if crate::platform::is_registered() {
                            ui.label("Listed in “Open with” for 3D files.");
                            if ui.button("Choose as default app…").on_hover_text("Opens Settings › Default apps").clicked() {
                                crate::platform::open_default_apps_settings();
                                ui.close();
                            }
                            if ui.button("Remove from “Open with”").clicked() {
                                if let Err(e) = crate::platform::unregister() {
                                    self.error = Some(format!("Cannot update the registry: {e}"));
                                }
                                ui.close();
                            }
                        } else if ui
                            .button("Add to “Open with” for 3D files")
                            .on_hover_text("Per-user registration, no administrator rights needed")
                            .clicked()
                        {
                            if let Err(e) = crate::platform::register() {
                                self.error = Some(format!("Cannot update the registry: {e}"));
                            }
                            ui.close();
                        }
                    }
                    ui.separator();
                    ui.label(egui::RichText::new("Blender (for .blend files)").strong());
                    let status = self.blender_status();
                    ui.label(status);
                    ui.horizontal(|ui| {
                        if ui.button("Choose Blender…").clicked() {
                            self.blender_dialog();
                            ui.close();
                        }
                        if self.settings.blender_path.is_some()
                            && ui.button("Find automatically").on_hover_text("Use the newest installed Blender").clicked()
                        {
                            let ectx = ui.ctx().clone();
                            self.set_blender(None, &ectx);
                            ui.close();
                        }
                    });
                    ui.separator();
                    ui.label(format!("GPU: {}", self.ctx.adapter_name));
                });
                ui.toggle_value(&mut self.settings.show_info, "ℹ Info").on_hover_text("Model information (I)");
                if ui.add_enabled(has_model, egui::Button::new("📷")).on_hover_text("Save the view as a PNG image").clicked() {
                    self.screenshot_dialog();
                }
            });
        });
    }

    fn animation_bar(&mut self, ui: &mut egui::Ui) {
        let Some(m) = &mut self.model else { return };
        let mut changed = false;
        ui.horizontal(|ui| {
            let icon = if m.anim.playing { "⏸" } else { "▶" };
            if ui.button(icon).on_hover_text("Play / pause (Space)").clicked() {
                if !m.anim.playing && !self.settings.loop_animation && m.anim.time >= m.duration() {
                    m.anim.time = 0.0;
                }
                m.anim.playing ^= true;
            }
            let names: Vec<String> = m.scene.animations.iter().map(|a| a.name.clone()).collect();
            if names.len() > 1 {
                let mut current = m.anim.current;
                egui::ComboBox::from_id_salt("animation")
                    .selected_text(names[current].clone())
                    .width(160.0)
                    .show_ui(ui, |ui| {
                        for (i, n) in names.iter().enumerate() {
                            ui.selectable_value(&mut current, i, n);
                        }
                    });
                if current != m.anim.current {
                    m.anim.current = current;
                    m.anim.time = 0.0;
                    changed = true;
                }
            } else {
                ui.label(&names[0]);
            }
            let duration = m.duration().max(1e-3);
            let label_width = 200.0;
            ui.spacing_mut().slider_width = (ui.available_width() - label_width - 150.0).max(80.0);
            let mut time = m.anim.time;
            if ui.add(egui::Slider::new(&mut time, 0.0..=duration).show_value(false)).changed() {
                m.anim.time = time;
                changed = true;
            }
            ui.label(format!("{:.2} / {:.2} s", m.anim.time, duration));
            egui::ComboBox::from_id_salt("speed").selected_text(format!("{}×", m.anim.speed)).width(60.0).show_ui(ui, |ui| {
                for s in [0.1, 0.25, 0.5, 1.0, 1.5, 2.0] {
                    ui.selectable_value(&mut m.anim.speed, s, format!("{s}×"));
                }
            });
            ui.toggle_value(&mut self.settings.loop_animation, "🔁").on_hover_text("Loop");
        });
        if changed {
            m.apply_pose(&self.ctx);
        }
    }

    fn info_panel(&self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            let Some(m) = &self.model else {
                ui.label("No model loaded.");
                return;
            };
            ui.heading(m.name());
            ui.label(egui::RichText::new(m.path.display().to_string()).small().weak());
            ui.add_space(6.0);
            let unit = m.scene.unit.clone().unwrap_or_default();
            let size = m.bounds.size();
            egui::Grid::new("info").num_columns(2).striped(true).show(ui, |ui| {
                let mut row = |k: &str, v: String| {
                    ui.label(k);
                    ui.label(v);
                    ui.end_row();
                };
                row("Format", m.format.name().to_owned());
                if let Some(by) = &m.scene.converted_with {
                    row("Converted with", by.clone());
                }
                row("File size", human_bytes(m.file_size));
                row("Load time", format!("{:.2} s", m.load_time.as_secs_f32()));
                row("Dimensions", format!("{} × {} × {} {unit}", fmt_num(size.x), fmt_num(size.y), fmt_num(size.z)));
                row("Triangles", thousands(m.stats.triangles));
                row("Vertices", thousands(m.stats.vertices));
                if m.stats.points > 0 {
                    row("Points", thousands(m.stats.points));
                }
                if m.stats.lines > 0 {
                    row("Lines", thousands(m.stats.lines));
                }
                row("Meshes", format!("{} ({} instances)", m.stats.meshes, m.stats.instances));
                row("Nodes", m.stats.nodes.to_string());
                row("Materials", m.stats.materials.to_string());
                row("Textures", m.stats.textures.len().to_string());
                row("Animations", m.stats.animations.len().to_string());
                if m.stats.skinned {
                    row("Skinning", "yes".into());
                }
                if m.stats.morph_targets {
                    row("Morph targets", "yes".into());
                }
            });
            if !m.stats.textures.is_empty() {
                ui.add_space(6.0);
                egui::CollapsingHeader::new(format!("Textures ({})", m.stats.textures.len())).show(ui, |ui| {
                    for (name, w, h) in &m.stats.textures {
                        ui.label(format!("{name} — {w}×{h}"));
                    }
                });
            }
            if !m.stats.animations.is_empty() {
                egui::CollapsingHeader::new(format!("Animations ({})", m.stats.animations.len())).show(ui, |ui| {
                    for (name, d) in &m.stats.animations {
                        ui.label(format!("{name} — {d:.2} s"));
                    }
                });
            }
            if !m.warnings.is_empty() {
                egui::CollapsingHeader::new(format!("⚠ Warnings ({})", m.warnings.len())).default_open(true).show(ui, |ui| {
                    for w in &m.warnings {
                        ui.label(egui::RichText::new(w).color(ui.visuals().warn_fg_color));
                    }
                });
            }
        });
    }

    fn viewport(&mut self, ui: &mut egui::Ui, frame: &eframe::Frame) {
        let rect = ui.max_rect();
        let response = ui.allocate_rect(rect, Sense::click_and_drag());
        let ppp = ui.ctx().pixels_per_point();
        let width = (rect.width() * ppp).round().max(1.0) as u32;
        let height = (rect.height() * ppp).round().max(1.0) as u32;
        let aspect = width as f32 / height as f32;

        if self.reset_view {
            if let Some(m) = &self.model {
                self.camera.reset(&m.world_bounds(), aspect);
            }
            self.reset_view = false;
        }

        if self.model.is_some() {
            let shift = ui.input(|i| i.modifiers.shift);
            let delta = response.drag_delta();
            if response.dragged_by(PointerButton::Primary) && !shift {
                self.camera.orbit(delta.x, delta.y);
            } else if response.dragged_by(PointerButton::Secondary)
                || response.dragged_by(PointerButton::Middle)
                || (response.dragged_by(PointerButton::Primary) && shift)
            {
                self.camera.pan(delta.x * ppp, delta.y * ppp, height as f32);
            }
            if response.hovered() {
                let (scroll, zoom) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
                if scroll != 0.0 {
                    self.camera.zoom((-scroll * 0.0018).exp());
                }
                if zoom != 1.0 {
                    self.camera.zoom(1.0 / zoom);
                }
            }
            if response.double_clicked() {
                self.reset_view = true;
                ui.ctx().request_repaint();
            }
        }

        // Animation playback.
        if let Some(m) = &mut self.model
            && m.anim.playing && m.has_animation() {
                let dt = ui.input(|i| i.unstable_dt).clamp(0.0, 0.1);
                m.advance(&self.ctx, dt, self.settings.loop_animation);
                // Pace playback ourselves on a fixed 60 Hz clock: vsync does not
                // always throttle (e.g. rendering on the NVIDIA GPU while the
                // display is driven by the Intel one).
                let now = std::time::Instant::now();
                let next = match self.next_anim_frame {
                    Some(t) if t > now => t,
                    Some(t) if now - t < ANIMATION_FRAME => t + ANIMATION_FRAME,
                    _ => now + ANIMATION_FRAME,
                };
                self.next_anim_frame = Some(next);
                ui.ctx().request_repaint_after(next.saturating_duration_since(now));
            }

        // Render the 3D view only when something changed.
        let key = RenderKey {
            width,
            height,
            camera: [
                self.camera.target.x,
                self.camera.target.y,
                self.camera.target.z,
                self.camera.distance,
                self.camera.yaw,
                self.camera.pitch,
                self.camera.fov_y,
            ],
            display: self.settings.display,
            edges: self.settings.show_edges,
            grid: self.settings.show_grid,
            background: self.settings.background,
            lighting: self.settings.lighting,
            point_size: (self.settings.point_size * ppp * 10.0) as u32,
            generation: self.model.as_ref().map_or(0, |m| m.generation),
            anim_time: self.model.as_ref().map_or(0, |m| m.anim.time.to_bits() ^ m.anim.current as u32),
            up: self.model.as_ref().map(|m| m.up),
        };
        if self.last_render != Some(key) {
            let params = self.view_params(width, height, ppp);
            self.renderer.render(self.model.as_mut().map(|m| &mut m.gpu), &params);
            self.register_texture(frame);
            self.last_render = Some(key);
        }
        if let Some(tex) = self.texture {
            ui.painter().image(tex, rect, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
        }

        self.overlays(ui, rect);
    }

    /// Makes the render target visible to egui (only when it was recreated).
    fn register_texture(&mut self, frame: &eframe::Frame) {
        let generation = self.renderer.target_generation();
        if self.texture.is_some() && self.texture_generation == generation {
            return;
        }
        let (Some(rs), Some(view)) = (frame.wgpu_render_state(), self.renderer.color_view()) else { return };
        let mut renderer = rs.renderer.write();
        match self.texture {
            None => self.texture = Some(renderer.register_native_texture(&rs.device, view, wgpu::FilterMode::Nearest)),
            Some(id) => renderer.update_egui_texture_from_wgpu_texture(&rs.device, view, wgpu::FilterMode::Nearest, id),
        }
        self.texture_generation = generation;
    }

    fn view_params(&self, width: u32, height: u32, ppp: f32) -> ViewParams {
        let aspect = width as f32 / height as f32;
        let (right, up, forward) = self.camera.basis();
        let (bg_top, bg_bottom) = self.settings.background.colors();
        let wire_only = self.settings.display == DisplayMode::Wireframe;
        let grid = self
            .model
            .as_ref()
            .filter(|_| self.settings.show_grid)
            .map(|m| grid_for(&m.world_bounds(), self.settings.background.grid_color()));
        ViewParams {
            view_proj: self.camera.proj(aspect) * self.camera.view(),
            eye: self.camera.eye(),
            width,
            height,
            lighting: self.settings.lighting.lighting(right, up, forward),
            bg_top,
            bg_bottom,
            grid,
            display: self.settings.display,
            show_edges: self.settings.show_edges,
            wire_color: self.settings.background.wire_color(wire_only),
            point_size: self.settings.point_size * ppp,
        }
    }

    fn overlays(&mut self, ui: &mut egui::Ui, rect: Rect) {
        let painter = ui.painter_at(rect);
        let dark = self.settings.background.is_dark();
        let text_color = if dark { Color32::from_gray(220) } else { Color32::from_gray(40) };

        if let Some(m) = &self.model {
            self.axis_gizmo(&painter, rect, m.up);
        } else if self.loading.is_none() {
            let c = rect.center();
            painter.text(c - Vec2::new(0.0, 28.0), Align2::CENTER_CENTER, "Open a 3D model", FontId::proportional(26.0), text_color);
            painter.text(
                c + Vec2::new(0.0, 8.0),
                Align2::CENTER_CENTER,
                "Drop a file here or press Ctrl+O",
                FontId::proportional(15.0),
                text_color.gamma_multiply(0.8),
            );
            painter.text(
                c + Vec2::new(0.0, 34.0),
                Align2::CENTER_CENTER,
                "glTF · GLB · FBX · OBJ · STL · PLY · 3MF · STEP · IGES · BLEND",
                FontId::proportional(13.0),
                text_color.gamma_multiply(0.6),
            );
        }

        // Drag & drop hint.
        if ui.input(|i| !i.raw.hovered_files.is_empty()) {
            painter.rect_filled(rect.shrink(8.0), 12.0, Color32::from_black_alpha(90));
            painter.rect_stroke(rect.shrink(8.0), 12.0, Stroke::new(2.0, Color32::from_rgb(90, 160, 255)), egui::StrokeKind::Inside);
            painter.text(rect.center(), Align2::CENTER_CENTER, "Drop to open", FontId::proportional(24.0), Color32::WHITE);
        }

        // Loading card.
        if let Some(job) = &self.loading {
            let mut cancel = false;
            let card = Rect::from_center_size(rect.center(), Vec2::new(360.0, 120.0));
            egui::Area::new(egui::Id::unique("loading")).fixed_pos(card.min).show(ui.ctx(), |ui| {
                egui::Frame::popup(ui.style()).show(ui, |ui| {
                    ui.set_width(card.width() - 20.0);
                    ui.horizontal(|ui| {
                        ui.add(egui::Spinner::new());
                        let name = job.path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                        ui.label(egui::RichText::new(name).strong());
                    });
                    let stage = job.progress.stage();
                    ui.label(if stage.is_empty() { "Loading…".to_owned() } else { stage });
                    ui.add(egui::ProgressBar::new(job.progress.fraction()).show_percentage());
                    ui.horizontal(|ui| {
                        ui.label(format!("{:.1} s", job.started.elapsed().as_secs_f32()));
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.button("Cancel").clicked() {
                                cancel = true;
                            }
                        });
                    });
                });
            });
            if cancel {
                job.progress.cancel();
            }
            ui.ctx().request_repaint_after(Duration::from_millis(50));
        }

        // Error banner.
        if let Some(err) = self.error.clone() {
            let mut close = false;
            let mut choose_blender = false;
            egui::Area::new(egui::Id::unique("error")).fixed_pos(rect.left_top() + Vec2::new(12.0, 12.0)).show(ui.ctx(), |ui| {
                egui::Frame::popup(ui.style()).fill(Color32::from_rgb(120, 30, 30)).show(ui, |ui| {
                    ui.set_max_width(rect.width() - 48.0);
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new(err).color(Color32::WHITE));
                        if self.error_needs_blender && ui.button("Choose Blender…").clicked() {
                            choose_blender = true;
                        }
                        if ui.button("✕").clicked() {
                            close = true;
                        }
                    });
                });
            });
            if close {
                self.error = None;
                self.error_needs_blender = false;
            }
            if choose_blender {
                self.blender_dialog();
            }
        }
    }

    /// Orientation gizmo: the file's X/Y/Z axes as seen by the camera.
    fn axis_gizmo(&self, painter: &egui::Painter, rect: Rect, up: UpAxis) {
        let root = crate::model::root_matrix(up);
        let (right, cam_up, forward) = self.camera.basis();
        let center = rect.left_bottom() + Vec2::new(46.0, -46.0);
        let len = 30.0;
        let mut axes: Vec<(f32, Vec2, Color32, &str)> = [
            (Vec3::X, Color32::from_rgb(230, 70, 70), "X"),
            (Vec3::Y, Color32::from_rgb(90, 200, 90), "Y"),
            (Vec3::Z, Color32::from_rgb(80, 140, 240), "Z"),
        ]
        .into_iter()
        .map(|(axis, color, label)| {
            let d = root.transform_vector3(axis);
            (d.dot(forward), Vec2::new(d.dot(right), -d.dot(cam_up)) * len, color, label)
        })
        .collect();
        axes.sort_by(|a, b| b.0.total_cmp(&a.0));
        painter.circle_filled(center, len + 12.0, Color32::from_black_alpha(40));
        for (_, v, color, label) in axes {
            painter.line_segment([center, center + v], Stroke::new(2.5, color));
            painter.circle_filled(center + v, 7.0, color);
            painter.text(center + v, Align2::CENTER_CENTER, label, FontId::proportional(10.0), Color32::WHITE);
        }
    }
}

impl eframe::App for ViewerApp {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let ectx = ui.ctx().clone();
        if let Some(fps) = &mut self.fps {
            fps.tick();
        }
        self.poll_loading();
        self.poll_dialog(&ectx);
        self.handle_input(&ectx);
        self.update_title(&ectx);

        egui::Panel::top("toolbar").show(ui, |ui| {
            ui.add_space(3.0);
            self.toolbar(ui);
            ui.add_space(2.0);
        });
        if self.model.as_ref().is_some_and(|m| m.has_animation()) {
            egui::Panel::bottom("animation").show(ui, |ui| {
                ui.add_space(3.0);
                self.animation_bar(ui);
                ui.add_space(2.0);
            });
        }
        if self.settings.show_info {
            egui::Panel::right("info").resizable(true).default_size(300.0).show(ui, |ui| self.info_panel(ui));
        }
        egui::CentralPanel::no_frame().show(ui, |ui| self.viewport(ui, frame));
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, &self.settings);
    }
}

fn display_label(mode: DisplayMode) -> &'static str {
    match mode {
        DisplayMode::Shaded => "Shaded",
        DisplayMode::ShadedWireframe => "Shaded + wireframe",
        DisplayMode::Wireframe => "Wireframe",
    }
}

/// Ground grid under the model with a power-of-ten cell size.
pub fn grid_for(bounds: &Aabb, color: [f32; 4]) -> Grid {
    let size = bounds.size();
    let span = size.x.max(size.z).max(size.y * 0.5).max(1e-6);
    let cell = 10f32.powf((span / 8.0).log10().floor());
    let c = bounds.center();
    Grid { height: bounds.min.y, cell, center: [c.x, c.z], extent: (span * 3.0).max(cell * 20.0), color }
}

fn human_bytes(b: u64) -> String {
    let b = b as f64;
    if b >= 1024.0 * 1024.0 * 1024.0 {
        format!("{:.2} GB", b / (1024.0 * 1024.0 * 1024.0))
    } else if b >= 1024.0 * 1024.0 {
        format!("{:.1} MB", b / (1024.0 * 1024.0))
    } else if b >= 1024.0 {
        format!("{:.1} KB", b / 1024.0)
    } else {
        format!("{b} B")
    }
}

fn thousands(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, ch) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

fn fmt_num(v: f32) -> String {
    let a = v.abs();
    if a >= 1000.0 {
        format!("{v:.0}")
    } else if a >= 10.0 {
        format!("{v:.1}")
    } else {
        format!("{v:.3}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formatting() {
        assert_eq!(thousands(1234567), "1,234,567");
        assert_eq!(thousands(12), "12");
        assert_eq!(human_bytes(2048), "2.0 KB");
    }

    #[test]
    fn grid_cell_is_power_of_ten() {
        let b = Aabb { min: Vec3::ZERO, max: Vec3::new(250.0, 40.0, 120.0) };
        let g = grid_for(&b, [1.0; 4]);
        assert_eq!(g.cell, 10.0);
        assert_eq!(g.height, 0.0);
    }
}
