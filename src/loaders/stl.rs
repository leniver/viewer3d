//! STL loader (binary and ASCII), with per-facet colours (VisCAM/SolidView
//! and Materialise conventions). Vertices are welded and normals rebuilt with
//! a crease angle, so curved CAD surfaces look smooth while edges stay sharp.

use super::LoadCtx;
use super::util;
use crate::scene::*;
use anyhow::{Context, Result, bail};
use rayon::prelude::*;

pub fn load(ctx: &LoadCtx) -> Result<Scene> {
    let data = std::fs::read(ctx.path).with_context(|| format!("cannot read '{}'", ctx.path.display()))?;
    ctx.progress.set(0.1);
    let (corners, colors) = parse(&data)?;
    drop(data);
    ctx.progress.check()?;
    ctx.progress.set(0.4);
    ctx.progress.set_stage("Building mesh");
    let mesh = build_mesh(&corners, colors.as_deref());
    ctx.progress.set(0.9);

    let mut scene = Scene { up_axis: UpAxis::Z, unit: None, ..Default::default() };
    let material = scene.add_material(Material { name: "STL".into(), ..Default::default() });
    let mut mesh = mesh;
    mesh.material = Some(material);
    scene.meshes.push(Mesh { name: file_stem(ctx), primitives: vec![mesh] });
    scene.add_node(Node { name: file_stem(ctx), mesh: Some(0), ..Default::default() }, None);
    Ok(scene)
}

fn file_stem(ctx: &LoadCtx) -> String {
    ctx.path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
}

/// Triangle corners (3 per facet) and optional per-facet colours.
pub type Parsed = (Vec<[f32; 3]>, Option<Vec<[u8; 4]>>);

pub fn parse(data: &[u8]) -> Result<Parsed> {
    if is_binary(data) {
        parse_binary(data)
    } else {
        Ok((parse_ascii(data)?, None))
    }
}

fn is_binary(data: &[u8]) -> bool {
    if data.len() < 84 {
        return false;
    }
    let count = u32::from_le_bytes([data[80], data[81], data[82], data[83]]) as u64;
    // Size check first: many binary files start with "solid" too.
    if 84 + count * 50 == data.len() as u64 {
        return true;
    }
    let head = String::from_utf8_lossy(&data[..data.len().min(512)]);
    !(head.trim_start().starts_with("solid") && head.contains("facet"))
}

fn parse_binary(data: &[u8]) -> Result<Parsed> {
    let count = u32::from_le_bytes([data[80], data[81], data[82], data[83]]) as usize;
    let available = (data.len() - 84) / 50;
    if available == 0 {
        bail!("the binary STL file contains no triangles");
    }
    let count = count.min(available);
    let header = &data[..80];
    // Materialise Magics: "COLOR=" + RGBA default colour in the header, and
    // per-facet colour valid when bit 15 is 0 (RGB in bits 0-4/5-9/10-14).
    let materialise = header.windows(6).position(|w| w == b"COLOR=").map(|p| {
        let c = &header[(p + 6).min(76)..];
        [c[0], c[1], c[2], c.get(3).copied().unwrap_or(255)]
    });
    let records = &data[84..84 + count * 50];

    let corners: Vec<[f32; 3]> = records
        .par_chunks_exact(50)
        .flat_map_iter(|r| {
            let f = |o: usize| f32::from_le_bytes([r[o], r[o + 1], r[o + 2], r[o + 3]]);
            [[f(12), f(16), f(20)], [f(24), f(28), f(32)], [f(36), f(40), f(44)]]
        })
        .collect();

    let attrs: Vec<u16> = records.par_chunks_exact(50).map(|r| u16::from_le_bytes([r[48], r[49]])).collect();
    let expand5 = |v: u16| ((v as u32 * 255 + 15) / 31) as u8;
    let colors: Option<Vec<[u8; 4]>> = match materialise {
        Some(default) => {
            let any = attrs.iter().any(|a| a & 0x8000 == 0);
            any.then(|| {
                attrs
                    .iter()
                    .map(|&a| {
                        if a & 0x8000 == 0 {
                            [expand5(a & 31), expand5((a >> 5) & 31), expand5((a >> 10) & 31), 255]
                        } else {
                            default
                        }
                    })
                    .collect()
            })
        }
        None => {
            // VisCAM / SolidView: colour valid when bit 15 is 1, blue in the low bits.
            let valid = attrs.iter().filter(|a| *a & 0x8000 != 0).count();
            (valid > 0 && valid * 2 >= attrs.len()).then(|| {
                attrs
                    .iter()
                    .map(|&a| {
                        if a & 0x8000 != 0 {
                            [expand5((a >> 10) & 31), expand5((a >> 5) & 31), expand5(a & 31), 255]
                        } else {
                            [204, 204, 204, 255]
                        }
                    })
                    .collect()
            })
        }
    };
    Ok((corners, colors))
}

fn parse_ascii(data: &[u8]) -> Result<Vec<[f32; 3]>> {
    let text = String::from_utf8_lossy(data);
    let mut corners = Vec::new();
    for line in text.lines() {
        let line = line.trim_start();
        let Some(rest) = line.strip_prefix("vertex") else { continue };
        let mut it = rest.split_ascii_whitespace().map(|t| t.parse::<f32>());
        match (it.next(), it.next(), it.next()) {
            (Some(Ok(x)), Some(Ok(y)), Some(Ok(z))) => corners.push([x, y, z]),
            _ => bail!("malformed vertex line in ASCII STL: '{}'", line.trim()),
        }
    }
    corners.truncate(corners.len() / 3 * 3);
    if corners.is_empty() {
        bail!("the STL file contains no triangles");
    }
    Ok(corners)
}

/// Welds identical corners and computes crease-angle normals.
pub fn build_mesh(corners: &[[f32; 3]], facet_colors: Option<&[[u8; 4]]>) -> Primitive {
    // Per-corner colour key so differently coloured facets do not share vertices.
    let corner_colors: Option<Vec<[u8; 4]>> =
        facet_colors.map(|c| c.iter().flat_map(|&col| [col, col, col]).collect());
    let corner_keys: Option<Vec<u32>> = corner_colors.as_ref().map(|c| c.iter().map(|&k| u32::from_le_bytes(k)).collect());

    let (unique, remap) = util::weld_positions(corners, None);
    let smoothed = util::smooth_normals(&unique, &remap, corner_keys.as_deref(), 40.0);

    let mut p = Primitive::new(Topology::Triangles);
    if let Some(colors) = &corner_colors {
        p.colors = smoothed.source_corner.iter().map(|&c| colors[c as usize]).collect();
    }
    p.positions = smoothed.positions;
    p.normals = smoothed.normals;
    p.indices = smoothed.indices;
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binary_stl(tris: &[[[f32; 3]; 3]], header: &[u8], attr: u16) -> Vec<u8> {
        let mut d = vec![0u8; 80];
        d[..header.len()].copy_from_slice(header);
        d.extend((tris.len() as u32).to_le_bytes());
        for t in tris {
            d.extend([0u8; 12]);
            for v in t {
                for c in v {
                    d.extend(c.to_le_bytes());
                }
            }
            d.extend(attr.to_le_bytes());
        }
        d
    }

    const TWO_TRIS: [[[f32; 3]; 3]; 2] = [
        [[0., 0., 0.], [1., 0., 0.], [1., 1., 0.]],
        [[0., 0., 0.], [1., 1., 0.], [0., 1., 0.]],
    ];

    #[test]
    fn binary_with_solid_header_is_binary() {
        let d = binary_stl(&TWO_TRIS, b"solid but actually binary", 0);
        let (corners, colors) = parse(&d).unwrap();
        assert_eq!(corners.len(), 6);
        assert!(colors.is_none());
        let p = build_mesh(&corners, None);
        // Flat quad: 4 welded vertices, all normals +Z.
        assert_eq!(p.positions.len(), 4);
        assert!(p.normals.iter().all(|n| (n[2] - 1.0).abs() < 1e-6));
    }

    #[test]
    fn viscam_colors() {
        // Valid bit + pure red (bits 10-14).
        let d = binary_stl(&TWO_TRIS, b"binary", 0x8000 | (31 << 10));
        let (_, colors) = parse(&d).unwrap();
        assert_eq!(colors.unwrap()[0], [255, 0, 0, 255]);
    }

    #[test]
    fn ascii() {
        let text = "solid t\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid t\n";
        let (corners, _) = parse(text.as_bytes()).unwrap();
        assert_eq!(corners, vec![[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]);
    }
}
