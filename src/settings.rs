//! User preferences (persisted between sessions) and the visual presets.

use crate::render::{DisplayMode, Lighting};
use crate::scene::srgb_to_linear;
use glam::Vec3;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub const MAX_RECENT: usize = 10;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Background {
    Dark,
    Light,
    Black,
    White,
}

impl Background {
    pub const ALL: [Background; 4] = [Background::Dark, Background::Light, Background::Black, Background::White];

    pub fn label(self) -> &'static str {
        match self {
            Background::Dark => "Dark gradient",
            Background::Light => "Light gradient",
            Background::Black => "Black",
            Background::White => "White",
        }
    }

    /// (top, bottom) in linear RGB.
    pub fn colors(self) -> ([f32; 3], [f32; 3]) {
        let c = |hex: u32| {
            [
                srgb_to_linear(((hex >> 16) & 255) as f32 / 255.0),
                srgb_to_linear(((hex >> 8) & 255) as f32 / 255.0),
                srgb_to_linear((hex & 255) as f32 / 255.0),
            ]
        };
        match self {
            Background::Dark => (c(0x45484f), c(0x1b1c20)),
            Background::Light => (c(0xf6f7f9), c(0xc4c8d0)),
            Background::Black => (c(0x000000), c(0x000000)),
            Background::White => (c(0xffffff), c(0xffffff)),
        }
    }

    pub fn is_dark(self) -> bool {
        matches!(self, Background::Dark | Background::Black)
    }

    pub fn grid_color(self) -> [f32; 4] {
        if self.is_dark() { [0.70, 0.74, 0.82, 0.20] } else { [0.12, 0.13, 0.16, 0.20] }
    }

    pub fn wire_color(self, wire_only: bool) -> [f32; 4] {
        let a = if wire_only { 0.85 } else { 0.35 };
        if self.is_dark() { [0.85, 0.88, 0.95, a] } else { [0.05, 0.06, 0.08, a] }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LightingPreset {
    Studio,
    Soft,
    Flat,
}

impl LightingPreset {
    pub const ALL: [LightingPreset; 3] = [LightingPreset::Studio, LightingPreset::Soft, LightingPreset::Flat];

    pub fn label(self) -> &'static str {
        match self {
            LightingPreset::Studio => "Studio",
            LightingPreset::Soft => "Soft",
            LightingPreset::Flat => "Flat (no shading)",
        }
    }

    /// Lights follow the camera so the visible side is always lit.
    pub fn lighting(self, right: Vec3, up: Vec3, forward: Vec3) -> Lighting {
        let key_dir = (-right * 0.55 + up * 0.75 - forward * 0.45).normalize();
        let fill_dir = (right * 0.8 - up * 0.1 - forward * 0.35).normalize();
        let (key, fill, ambient) = match self {
            LightingPreset::Studio => (2.6, 0.7, 0.9),
            LightingPreset::Soft => (1.3, 0.6, 1.5),
            LightingPreset::Flat => (0.0, 0.0, 1.0),
        };
        let scale = |c: [f32; 3], s: f32| [c[0] * s, c[1] * s, c[2] * s];
        Lighting {
            key_dir,
            key: scale([1.0, 0.97, 0.92], key),
            fill_dir,
            fill: scale([0.82, 0.88, 1.0], fill),
            sky: [0.62, 0.66, 0.74],
            ground: [0.30, 0.28, 0.26],
            ambient,
            exposure: 1.0,
            flat: self == LightingPreset::Flat,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub display: DisplayMode,
    pub show_edges: bool,
    pub show_grid: bool,
    pub background: Background,
    pub lighting: LightingPreset,
    pub show_info: bool,
    pub point_size: f32,
    pub loop_animation: bool,
    pub recent: Vec<PathBuf>,
    /// Blender program chosen for .blend files (None: found automatically).
    pub blender_path: Option<PathBuf>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            display: DisplayMode::Shaded,
            show_edges: true,
            show_grid: true,
            background: Background::Dark,
            lighting: LightingPreset::Studio,
            show_info: false,
            point_size: 3.0,
            loop_animation: true,
            recent: Vec::new(),
            blender_path: None,
        }
    }
}

impl Settings {
    pub fn add_recent(&mut self, path: PathBuf) {
        self.recent.retain(|p| p != &path);
        self.recent.insert(0, path);
        self.recent.truncate(MAX_RECENT);
    }
}
