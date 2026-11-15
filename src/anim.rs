//! Animation sampling and pose evaluation (cheap CPU work: a few thousand
//! keyframe lookups per frame). Vertex deformation itself runs on the GPU.

use crate::scene::{Animation, Channel, Interpolation, Property, Scene, Transform};
use glam::{Mat4, Quat, Vec3};

/// Node transforms and morph weights for one moment of an animation.
pub struct Pose {
    pub locals: Vec<Transform>,
    pub globals: Vec<Mat4>,
    pub weights: Vec<Vec<f32>>,
}

/// Precomputed data to evaluate poses quickly.
pub struct Animator {
    order: Vec<usize>,
    rest: Vec<Transform>,
    rest_weights: Vec<Vec<f32>>,
    parents: Vec<Option<usize>>,
}

impl Animator {
    pub fn new(scene: &Scene) -> Self {
        let rest_weights = scene
            .nodes
            .iter()
            .map(|n| {
                let targets = n
                    .mesh
                    .and_then(|m| scene.meshes.get(m))
                    .map(|m| m.primitives.iter().map(|p| p.targets.len()).max().unwrap_or(0))
                    .unwrap_or(0);
                let mut w = n.weights.clone();
                w.resize(targets, 0.0);
                w
            })
            .collect();
        Self {
            order: scene.traversal_order(),
            rest: scene.nodes.iter().map(|n| n.transform).collect(),
            rest_weights,
            parents: scene.nodes.iter().map(|n| n.parent).collect(),
        }
    }

    pub fn rest_pose(&self) -> Pose {
        let mut pose = Pose {
            locals: self.rest.clone(),
            globals: vec![Mat4::IDENTITY; self.rest.len()],
            weights: self.rest_weights.clone(),
        };
        self.update_globals(&mut pose);
        pose
    }

    /// Evaluates `anim` at `time` into `pose` (reusing its allocations).
    pub fn evaluate(&self, anim: Option<&Animation>, time: f32, pose: &mut Pose) {
        pose.locals.clone_from(&self.rest);
        for (w, rest) in pose.weights.iter_mut().zip(&self.rest_weights) {
            w.clone_from(rest);
        }
        if let Some(anim) = anim {
            let mut buf = [0f32; 4];
            for ch in &anim.channels {
                let Some(local) = pose.locals.get_mut(ch.node) else { continue };
                let n = ch.property.components();
                sample(ch, time, &mut buf[..n]);
                match ch.property {
                    Property::Translation => local.translation = Vec3::from_slice(&buf[..3]),
                    Property::Scale => local.scale = Vec3::from_slice(&buf[..3]),
                    Property::Rotation => {
                        let q = Quat::from_slice(&buf[..4]);
                        if q.is_finite() && q.length_squared() > 1e-12 {
                            local.rotation = q.normalize();
                        }
                    }
                    Property::MorphWeight(t) => {
                        if let Some(w) = pose.weights.get_mut(ch.node).and_then(|w| w.get_mut(t as usize)) {
                            *w = buf[0];
                        }
                    }
                }
            }
        }
        self.update_globals(pose);
    }

    fn update_globals(&self, pose: &mut Pose) {
        for &n in &self.order {
            let local = pose.locals[n].matrix();
            pose.globals[n] = match self.parents[n] {
                Some(p) => pose.globals[p] * local,
                None => local,
            };
        }
    }
}

/// Samples a channel at `t` (clamped to its key range) into `out`.
pub fn sample(ch: &Channel, t: f32, out: &mut [f32]) {
    let n = out.len();
    let keys = ch.times.len();
    let cubic = ch.interpolation == Interpolation::CubicSpline;
    let stride = if cubic { 3 * n } else { n };
    let value = |k: usize| {
        let base = k * stride + if cubic { n } else { 0 };
        &ch.values[base..base + n]
    };
    if keys == 0 || ch.values.len() < keys * stride {
        return;
    }
    if keys == 1 || t <= ch.times[0] {
        out.copy_from_slice(value(0));
        return;
    }
    if t >= ch.times[keys - 1] {
        out.copy_from_slice(value(keys - 1));
        return;
    }
    let i = ch.times.partition_point(|&x| x <= t).saturating_sub(1).min(keys - 2);
    let (t0, t1) = (ch.times[i], ch.times[i + 1]);
    let dt = t1 - t0;
    let u = if dt > 0.0 { ((t - t0) / dt).clamp(0.0, 1.0) } else { 0.0 };
    let rotation = ch.property == Property::Rotation;

    match ch.interpolation {
        Interpolation::Step => out.copy_from_slice(value(i)),
        Interpolation::Linear => {
            let (a, b) = (value(i), value(i + 1));
            if rotation {
                let qa = Quat::from_slice(a);
                let qb = Quat::from_slice(b);
                let q = qa.slerp(qb, u);
                out.copy_from_slice(&q.to_array());
            } else {
                for k in 0..n {
                    out[k] = a[k] + (b[k] - a[k]) * u;
                }
            }
        }
        Interpolation::CubicSpline => {
            let p0 = value(i);
            let p1 = value(i + 1);
            let m0 = &ch.values[i * stride + 2 * n..i * stride + 3 * n]; // out-tangent of key i
            let m1 = &ch.values[(i + 1) * stride..(i + 1) * stride + n]; // in-tangent of key i+1
            let (u2, u3) = (u * u, u * u * u);
            let h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
            let h10 = u3 - 2.0 * u2 + u;
            let h01 = -2.0 * u3 + 3.0 * u2;
            let h11 = u3 - u2;
            for k in 0..n {
                out[k] = h00 * p0[k] + h10 * dt * m0[k] + h01 * p1[k] + h11 * dt * m1[k];
            }
            if rotation {
                let q = Quat::from_slice(out).normalize();
                out.copy_from_slice(&q.to_array());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn channel(property: Property, interpolation: Interpolation, times: Vec<f32>, values: Vec<f32>) -> Channel {
        Channel { node: 0, property, interpolation, times, values }
    }

    #[test]
    fn linear_translation() {
        let ch = channel(Property::Translation, Interpolation::Linear, vec![0.0, 2.0], vec![0., 0., 0., 2., 4., 6.]);
        let mut out = [0.0; 3];
        sample(&ch, 1.0, &mut out);
        assert_eq!(out, [1.0, 2.0, 3.0]);
        sample(&ch, 5.0, &mut out);
        assert_eq!(out, [2.0, 4.0, 6.0]);
        sample(&ch, -1.0, &mut out);
        assert_eq!(out, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn step_and_rotation() {
        let ch = channel(Property::Scale, Interpolation::Step, vec![0.0, 1.0], vec![1., 1., 1., 2., 2., 2.]);
        let mut out = [0.0; 3];
        sample(&ch, 0.99, &mut out);
        assert_eq!(out, [1.0, 1.0, 1.0]);

        let q1 = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let mut values = Quat::IDENTITY.to_array().to_vec();
        values.extend(q1.to_array());
        let ch = channel(Property::Rotation, Interpolation::Linear, vec![0.0, 1.0], values);
        let mut out = [0.0; 4];
        sample(&ch, 0.5, &mut out);
        let q = Quat::from_slice(&out);
        assert!(q.angle_between(Quat::from_rotation_y(std::f32::consts::FRAC_PI_4)) < 1e-4);
    }

    #[test]
    fn cubic_spline_hits_keys() {
        // Scalar weight with zero tangents: value at keys must match exactly.
        let ch = channel(
            Property::MorphWeight(0),
            Interpolation::CubicSpline,
            vec![0.0, 1.0],
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        );
        let mut out = [0.0];
        sample(&ch, 0.0, &mut out);
        assert_eq!(out[0], 0.0);
        sample(&ch, 1.0, &mut out);
        assert_eq!(out[0], 1.0);
        sample(&ch, 0.5, &mut out);
        assert!((out[0] - 0.5).abs() < 1e-6);
    }
}
