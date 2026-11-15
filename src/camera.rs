//! Orbit camera (turntable around a target point, Y up).

use crate::scene::Aabb;
use glam::{Mat4, Vec3};

#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub target: Vec3,
    pub distance: f32,
    /// Rotation around the vertical axis, radians.
    pub yaw: f32,
    /// Elevation, radians, within (-pi/2, pi/2).
    pub pitch: f32,
    /// Vertical field of view, radians.
    pub fov_y: f32,
    /// Radius of the framed model, used to bound zoom.
    pub scene_radius: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            target: Vec3::ZERO,
            distance: 3.0,
            yaw: 0.6,
            pitch: 0.4,
            fov_y: 40f32.to_radians(),
            scene_radius: 1.0,
        }
    }
}

impl Camera {
    fn direction(&self) -> Vec3 {
        Vec3::new(self.pitch.cos() * self.yaw.sin(), self.pitch.sin(), self.pitch.cos() * self.yaw.cos())
    }

    pub fn eye(&self) -> Vec3 {
        self.target + self.direction() * self.distance
    }

    pub fn view(&self) -> Mat4 {
        glam::camera::rh::view::look_at_mat4(self.eye(), self.target, Vec3::Y)
    }

    /// Reverse-Z perspective with an infinite far plane (best depth precision),
    /// WebGPU conventions (depth in [0, 1], Y up).
    pub fn proj(&self, aspect: f32) -> Mat4 {
        let near = (self.distance * 0.002).max(self.scene_radius * 1e-6).max(1e-6);
        glam::camera::rh::proj::directx::perspective_infinite_reverse(self.fov_y, aspect.max(1e-3), near)
    }

    /// Places the camera so the whole box fills the view, keeping the
    /// orientation: every corner of the box is fitted to the frustum (tighter
    /// than a bounding sphere, especially for flat or long objects).
    pub fn frame(&mut self, bounds: &Aabb, aspect: f32) {
        if bounds.is_empty() {
            return;
        }
        let radius = (bounds.size().length() * 0.5).max(1e-6);
        self.scene_radius = radius;
        let center = bounds.center();
        self.target = center;
        let (right, up, forward) = self.basis();
        let tan_v = (self.fov_y * 0.5).tan();
        let tan_h = tan_v * aspect.max(1e-3);
        let mut distance: f32 = 0.0;
        for i in 0..8 {
            let corner = Vec3::new(
                if i & 1 == 0 { bounds.min.x } else { bounds.max.x },
                if i & 2 == 0 { bounds.min.y } else { bounds.max.y },
                if i & 4 == 0 { bounds.min.z } else { bounds.max.z },
            );
            let r = corner - center;
            let towards_camera = -r.dot(forward);
            distance = distance
                .max(towards_camera + r.dot(right).abs() / tan_h)
                .max(towards_camera + r.dot(up).abs() / tan_v);
        }
        // 8% margin; never closer than the box itself.
        self.distance = (distance * 1.08).max(radius * 1.05);
    }

    /// Default 3/4 view, then frames the box.
    pub fn reset(&mut self, bounds: &Aabb, aspect: f32) {
        self.yaw = 0.6;
        self.pitch = 0.4;
        self.frame(bounds, aspect);
    }

    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx * 0.008;
        self.pitch = (self.pitch + dy * 0.008).clamp(-1.553, 1.553);
    }

    /// Pans by a screen-space delta in pixels.
    pub fn pan(&mut self, dx: f32, dy: f32, viewport_height: f32) {
        let world_per_pixel = 2.0 * self.distance * (self.fov_y * 0.5).tan() / viewport_height.max(1.0);
        let view = self.view();
        let right = Vec3::new(view.x_axis.x, view.y_axis.x, view.z_axis.x);
        let up = Vec3::new(view.x_axis.y, view.y_axis.y, view.z_axis.y);
        self.target += (-dx * right + dy * up) * world_per_pixel;
    }

    /// Multiplies the distance (zoom in < 1 < zoom out).
    pub fn zoom(&mut self, factor: f32) {
        let r = self.scene_radius.max(1e-6);
        self.distance = (self.distance * factor).clamp(r * 1e-4, r * 1e3);
    }

    /// Unit vectors of the camera frame in world space (right, up, forward).
    pub fn basis(&self) -> (Vec3, Vec3, Vec3) {
        let forward = -self.direction();
        let right = forward.cross(Vec3::Y).normalize_or(Vec3::X);
        let up = right.cross(forward);
        (right, up, forward)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn framing_contains_the_box() {
        let mut cam = Camera::default();
        let b = Aabb { min: Vec3::splat(-1.0), max: Vec3::splat(1.0) };
        cam.reset(&b, 16.0 / 9.0);
        let vp = cam.proj(16.0 / 9.0) * cam.view();
        for i in 0..8 {
            let p = Vec3::new(
                if i & 1 == 0 { -1.0 } else { 1.0 },
                if i & 2 == 0 { -1.0 } else { 1.0 },
                if i & 4 == 0 { -1.0 } else { 1.0 },
            );
            let c = vp.project_point3(p);
            assert!(c.x.abs() <= 1.0 && c.y.abs() <= 1.0, "corner {p} projected to {c}");
            assert!(c.z > 0.0 && c.z <= 1.0, "reverse-Z depth in (0,1], got {}", c.z);
        }
    }
}
