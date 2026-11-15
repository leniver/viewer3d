//! STEP / IGES / BREP through OpenCASCADE (crate `occt-step`).

use super::util::Cancelled;
use super::{Format, LoadCtx};
use crate::scene::*;
use anyhow::{Result, anyhow};
use glam::{Mat4, Vec4};
use std::collections::HashMap;

pub fn load(ctx: &LoadCtx, format: Format) -> Result<Scene> {
    let occt_format = match format {
        Format::Step => occt_step::Format::Step,
        Format::Iges => occt_step::Format::Iges,
        _ => occt_step::Format::Brep,
    };
    ctx.progress.set_stage("Reading and tessellating CAD geometry");
    let progress = |fraction: f64| {
        ctx.progress.set(fraction as f32 * 0.9);
        !ctx.progress.is_cancelled()
    };
    let model = occt_step::load(ctx.path, occt_format, &occt_step::Options::default(), &progress).map_err(|e| match e {
        occt_step::Error::Cancelled => anyhow!(Cancelled),
        occt_step::Error::Failed(msg) => anyhow!(msg),
    })?;
    ctx.progress.check()?;

    let mut scene = Scene {
        up_axis: UpAxis::Z,
        unit: (format != Format::Brep).then(|| "mm".to_owned()),
        ..Default::default()
    };
    let default_material = scene.add_material(Material {
        name: "CAD default".into(),
        base_color: [0.60, 0.62, 0.66, 1.0],
        metallic: 0.0,
        roughness: 0.45,
        ..Default::default()
    });
    let edge_material = scene.add_material(Material {
        name: "Edges".into(),
        base_color: [0.02, 0.02, 0.025, 1.0],
        unlit: true,
        ..Default::default()
    });
    let mut color_materials: HashMap<[u32; 4], usize> = HashMap::new();

    for m in model.meshes {
        let mut mesh = Mesh::default();
        for g in &m.groups {
            let start = g.first_index as usize;
            let end = (start + g.index_count as usize).min(m.indices.len());
            if start >= end {
                continue;
            }
            let mut p = compact(&m.positions, &m.normals, &m.indices[start..end]);
            p.material = Some(match g.color {
                None => default_material,
                Some(rgba) => *color_materials.entry(rgba.map(f32::to_bits)).or_insert_with(|| {
                    scene.materials.push(Material {
                        name: "CAD colour".into(),
                        base_color: rgba,
                        roughness: 0.45,
                        alpha_mode: if rgba[3] < 0.999 { AlphaMode::Blend } else { AlphaMode::Opaque },
                        ..Default::default()
                    });
                    scene.materials.len() - 1
                }),
            });
            mesh.primitives.push(p);
        }
        if !m.edges.is_empty() {
            let mut e = Primitive::new(Topology::Lines);
            e.role = PrimitiveRole::FeatureEdges;
            e.indices = (0..m.edges.len() as u32).collect();
            e.positions = m.edges;
            e.material = Some(edge_material);
            mesh.primitives.push(e);
        }
        scene.meshes.push(mesh);
    }

    for inst in model.instances {
        let r = inst.matrix;
        let matrix = Mat4::from_cols(
            Vec4::new(r[0][0] as f32, r[1][0] as f32, r[2][0] as f32, 0.0),
            Vec4::new(r[0][1] as f32, r[1][1] as f32, r[2][1] as f32, 0.0),
            Vec4::new(r[0][2] as f32, r[1][2] as f32, r[2][2] as f32, 0.0),
            Vec4::new(r[0][3] as f32, r[1][3] as f32, r[2][3] as f32, 1.0),
        );
        scene.add_node(
            Node {
                name: inst.name,
                transform: Transform::from_matrix(&matrix),
                mesh: Some(inst.mesh),
                ..Default::default()
            },
            None,
        );
    }
    Ok(scene)
}

/// Extracts the vertices used by a range of triangles.
fn compact(positions: &[[f32; 3]], normals: &[[f32; 3]], indices: &[u32]) -> Primitive {
    let mut remap = vec![u32::MAX; positions.len()];
    let mut p = Primitive::new(Topology::Triangles);
    p.indices.reserve(indices.len());
    for &i in indices {
        let Some(slot) = remap.get_mut(i as usize) else { continue };
        if *slot == u32::MAX {
            *slot = p.positions.len() as u32;
            p.positions.push(positions[i as usize]);
            p.normals.push(normals.get(i as usize).copied().unwrap_or([0.0, 0.0, 1.0]));
        }
        p.indices.push(*slot);
    }
    p
}
