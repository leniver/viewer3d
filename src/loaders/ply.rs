//! PLY loader (ASCII, binary little/big endian). Meshes and point clouds,
//! vertex normals/colours/UVs, per-face texture coordinates and colours,
//! and the MeshLab `TextureFile` comment.

use super::LoadCtx;
use super::util::{self, ImageSource};
use crate::scene::*;
use anyhow::{Context, Result, anyhow, bail};
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Scalar {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

impl Scalar {
    fn parse(s: &str) -> Option<Scalar> {
        Some(match s {
            "char" | "int8" => Scalar::I8,
            "uchar" | "uint8" => Scalar::U8,
            "short" | "int16" => Scalar::I16,
            "ushort" | "uint16" => Scalar::U16,
            "int" | "int32" => Scalar::I32,
            "uint" | "uint32" => Scalar::U32,
            "float" | "float32" => Scalar::F32,
            "double" | "float64" => Scalar::F64,
            _ => return None,
        })
    }

    fn size(self) -> usize {
        match self {
            Scalar::I8 | Scalar::U8 => 1,
            Scalar::I16 | Scalar::U16 => 2,
            Scalar::I32 | Scalar::U32 | Scalar::F32 => 4,
            Scalar::F64 => 8,
        }
    }

    #[inline]
    fn read(self, b: &[u8], big_endian: bool) -> f64 {
        macro_rules! rd {
            ($t:ty, $n:expr) => {{
                let mut a = [0u8; $n];
                a.copy_from_slice(&b[..$n]);
                (if big_endian { <$t>::from_be_bytes(a) } else { <$t>::from_le_bytes(a) }) as f64
            }};
        }
        match self {
            Scalar::I8 => b[0] as i8 as f64,
            Scalar::U8 => b[0] as f64,
            Scalar::I16 => rd!(i16, 2),
            Scalar::U16 => rd!(u16, 2),
            Scalar::I32 => rd!(i32, 4),
            Scalar::U32 => rd!(u32, 4),
            Scalar::F32 => rd!(f32, 4),
            Scalar::F64 => rd!(f64, 8),
        }
    }

    /// Maps a colour component to 0..255.
    fn color(self, v: f64) -> u8 {
        let n = match self {
            Scalar::F32 | Scalar::F64 => v * 255.0,
            Scalar::U16 | Scalar::I16 => v / 257.0,
            _ => v,
        };
        n.round().clamp(0.0, 255.0) as u8
    }
}

#[derive(Clone, Debug)]
enum Kind {
    Scalar(Scalar),
    List { count: Scalar, item: Scalar },
}

#[derive(Clone, Debug)]
struct Property {
    name: String,
    kind: Kind,
}

#[derive(Clone, Debug)]
struct Element {
    name: String,
    count: usize,
    props: Vec<Property>,
}

impl Element {
    fn find(&self, names: &[&str]) -> Option<usize> {
        names.iter().find_map(|n| self.props.iter().position(|p| p.name == *n))
    }

    /// Record size when every property is a scalar.
    fn fixed_size(&self) -> Option<usize> {
        self.props
            .iter()
            .map(|p| match p.kind {
                Kind::Scalar(s) => Some(s.size()),
                Kind::List { .. } => None,
            })
            .sum()
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Encoding {
    Ascii,
    Binary { big_endian: bool },
}

struct Header {
    encoding: Encoding,
    elements: Vec<Element>,
    texture: Option<String>,
    body_offset: usize,
}

fn parse_header(data: &[u8]) -> Result<Header> {
    let end = data
        .windows(10)
        .position(|w| w == b"end_header")
        .ok_or_else(|| anyhow!("not a PLY file (missing end_header)"))?;
    let mut body_offset = end + 10;
    // Line ending after end_header: \n, \r\n or (rarely) \r.
    if data.get(body_offset) == Some(&b'\r') {
        body_offset += 1;
    }
    if data.get(body_offset) == Some(&b'\n') {
        body_offset += 1;
    }
    let text = String::from_utf8_lossy(&data[..end]);
    let mut lines = text.lines();
    if lines.next().map(str::trim) != Some("ply") {
        bail!("not a PLY file");
    }
    let mut encoding = None;
    let mut elements: Vec<Element> = Vec::new();
    let mut texture = None;
    for line in lines {
        let mut t = line.split_ascii_whitespace();
        match t.next() {
            Some("format") => {
                encoding = Some(match t.next() {
                    Some("ascii") => Encoding::Ascii,
                    Some("binary_little_endian") => Encoding::Binary { big_endian: false },
                    Some("binary_big_endian") => Encoding::Binary { big_endian: true },
                    other => bail!("unsupported PLY format {other:?}"),
                })
            }
            Some("comment") | Some("obj_info") => {
                let rest: Vec<&str> = t.collect();
                if rest.first().is_some_and(|w| w.eq_ignore_ascii_case("TextureFile")) && rest.len() > 1 {
                    texture = Some(rest[1..].join(" "));
                }
            }
            Some("element") => {
                let name = t.next().unwrap_or_default().to_owned();
                let count = t.next().and_then(|c| c.parse().ok()).ok_or_else(|| anyhow!("bad element line '{line}'"))?;
                elements.push(Element { name, count, props: Vec::new() });
            }
            Some("property") => {
                let el = elements.last_mut().ok_or_else(|| anyhow!("property before element"))?;
                let ty = t.next().unwrap_or_default();
                let kind = if ty == "list" {
                    let count = t.next().and_then(Scalar::parse).ok_or_else(|| anyhow!("bad list '{line}'"))?;
                    let item = t.next().and_then(Scalar::parse).ok_or_else(|| anyhow!("bad list '{line}'"))?;
                    Kind::List { count, item }
                } else {
                    Kind::Scalar(Scalar::parse(ty).ok_or_else(|| anyhow!("unknown type in '{line}'"))?)
                };
                el.props.push(Property { name: t.next().unwrap_or_default().to_owned(), kind });
            }
            _ => {}
        }
    }
    Ok(Header { encoding: encoding.ok_or_else(|| anyhow!("missing PLY format line"))?, elements, texture, body_offset })
}

/// Values of one record: scalars, and list items flattened.
#[derive(Default)]
struct Record {
    scalars: Vec<f64>,
    lists: Vec<Vec<f64>>,
}

/// Reader positioned in the body.
struct Body<'a> {
    data: &'a [u8],
    pos: usize,
    encoding: Encoding,
}

impl<'a> Body<'a> {
    fn binary_scalar(&mut self, s: Scalar, big: bool) -> Result<f64> {
        let n = s.size();
        let b = self.data.get(self.pos..self.pos + n).ok_or_else(|| anyhow!("unexpected end of PLY data"))?;
        self.pos += n;
        Ok(s.read(b, big))
    }

    fn ascii_token(&mut self) -> Result<f64> {
        let d = self.data;
        while self.pos < d.len() && d[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
        let start = self.pos;
        while self.pos < d.len() && !d[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
        let tok = std::str::from_utf8(&d[start..self.pos]).map_err(|_| anyhow!("invalid PLY text"))?;
        tok.parse::<f64>().map_err(|_| anyhow!("invalid number '{tok}' in PLY data"))
    }

    fn value(&mut self, s: Scalar) -> Result<f64> {
        match self.encoding {
            Encoding::Ascii => self.ascii_token(),
            Encoding::Binary { big_endian } => self.binary_scalar(s, big_endian),
        }
    }

    fn record(&mut self, el: &Element, rec: &mut Record) -> Result<()> {
        rec.scalars.clear();
        let mut li = 0;
        for p in &el.props {
            match p.kind {
                Kind::Scalar(s) => {
                    let v = self.value(s)?;
                    rec.scalars.push(v);
                }
                Kind::List { count, item } => {
                    let n = self.value(count)?;
                    if !(0.0..=1_000_000.0).contains(&n) {
                        bail!("invalid list length {n} in PLY data");
                    }
                    if rec.lists.len() <= li {
                        rec.lists.push(Vec::new());
                    }
                    let list = &mut rec.lists[li];
                    list.clear();
                    for _ in 0..n as usize {
                        list.push(self.value(item)?);
                    }
                    li += 1;
                }
            }
        }
        Ok(())
    }

    /// Skips an element quickly when possible.
    fn skip(&mut self, el: &Element) -> Result<()> {
        if let (Encoding::Binary { .. }, Some(size)) = (self.encoding, el.fixed_size()) {
            self.pos = self.pos.checked_add(size * el.count).filter(|&p| p <= self.data.len()).ok_or_else(|| anyhow!("truncated PLY file"))?;
            return Ok(());
        }
        let mut rec = Record::default();
        for _ in 0..el.count {
            self.record(el, &mut rec)?;
        }
        Ok(())
    }
}

#[derive(Default)]
struct Vertices {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    colors: Vec<[u8; 4]>,
    uvs: Vec<[f32; 2]>,
}

/// Column indices of the vertex properties we use.
struct VertexLayout {
    pos: [usize; 3],
    normal: Option<[usize; 3]>,
    color: Option<([usize; 3], Option<usize>)>,
    uv: Option<[usize; 2]>,
}

impl VertexLayout {
    fn new(el: &Element) -> Result<Self> {
        let p = |n: &[&str]| el.find(n);
        let pos = [p(&["x"]), p(&["y"]), p(&["z"])];
        let [Some(x), Some(y), Some(z)] = pos else { bail!("PLY vertices have no x/y/z") };
        let tri = |a: &[&str], b: &[&str], c: &[&str]| match (p(a), p(b), p(c)) {
            (Some(a), Some(b), Some(c)) => Some([a, b, c]),
            _ => None,
        };
        let normal = tri(&["nx", "normal_x"], &["ny", "normal_y"], &["nz", "normal_z"]);
        let color = tri(&["red", "r", "diffuse_red"], &["green", "g", "diffuse_green"], &["blue", "b", "diffuse_blue"])
            .map(|rgb| (rgb, p(&["alpha", "a", "diffuse_alpha"])));
        let uv = match (p(&["s", "u", "texture_u", "texture_s"]), p(&["t", "v", "texture_v", "texture_t"])) {
            (Some(u), Some(v)) => Some([u, v]),
            _ => None,
        };
        Ok(Self { pos: [x, y, z], normal, color, uv })
    }

    fn scalar_type(el: &Element, i: usize) -> Scalar {
        match el.props[i].kind {
            Kind::Scalar(s) => s,
            Kind::List { item, .. } => item,
        }
    }
}

fn read_vertices(body: &mut Body, el: &Element) -> Result<Vertices> {
    let layout = VertexLayout::new(el)?;
    let ctype = layout.color.map(|(rgb, _)| VertexLayout::scalar_type(el, rgb[0])).unwrap_or(Scalar::U8);
    let n = el.count;

    // Converts one record (as scalars) into the output arrays.
    let convert = |s: &dyn Fn(usize) -> f64| {
        let pos = [s(layout.pos[0]) as f32, s(layout.pos[1]) as f32, s(layout.pos[2]) as f32];
        let normal = layout.normal.map(|i| [s(i[0]) as f32, s(i[1]) as f32, s(i[2]) as f32]);
        let color = layout.color.map(|(rgb, a)| {
            [ctype.color(s(rgb[0])), ctype.color(s(rgb[1])), ctype.color(s(rgb[2])), a.map_or(255, |a| ctype.color(s(a)))]
        });
        let uv = layout.uv.map(|i| [s(i[0]) as f32, 1.0 - s(i[1]) as f32]);
        (pos, normal, color, uv)
    };

    type Row = ([f32; 3], Option<[f32; 3]>, Option<[u8; 4]>, Option<[f32; 2]>);
    let rows: Vec<Row> = match (body.encoding, el.fixed_size()) {
        (Encoding::Binary { big_endian }, Some(size)) => {
            let bytes = body.data.get(body.pos..body.pos + size * n).ok_or_else(|| anyhow!("truncated PLY vertex data"))?;
            body.pos += size * n;
            let mut offsets = Vec::with_capacity(el.props.len());
            let mut o = 0;
            for p in &el.props {
                offsets.push(o);
                if let Kind::Scalar(s) = p.kind {
                    o += s.size();
                }
            }
            let types: Vec<Scalar> = (0..el.props.len()).map(|i| VertexLayout::scalar_type(el, i)).collect();
            bytes
                .par_chunks_exact(size)
                .map(|r| convert(&|i: usize| types[i].read(&r[offsets[i]..], big_endian)))
                .collect()
        }
        (Encoding::Ascii, Some(_)) => {
            // One record per line: split lines, then parse in parallel.
            let mut lines = Vec::with_capacity(n);
            let d = body.data;
            while lines.len() < n && body.pos < d.len() {
                let start = body.pos;
                while body.pos < d.len() && d[body.pos] != b'\n' {
                    body.pos += 1;
                }
                let line = &d[start..body.pos];
                body.pos += 1;
                if line.iter().any(|c| !c.is_ascii_whitespace()) {
                    lines.push(line);
                }
            }
            if lines.len() < n {
                bail!("truncated PLY vertex data");
            }
            let nprops = el.props.len();
            lines
                .par_iter()
                .map(|line| -> Result<Row> {
                    let text = std::str::from_utf8(line).map_err(|_| anyhow!("invalid PLY text"))?;
                    let mut vals = [0f64; 32];
                    let mut k = 0;
                    for tok in text.split_ascii_whitespace().take(nprops.min(32)) {
                        vals[k] = tok.parse().map_err(|_| anyhow!("invalid number '{tok}' in PLY data"))?;
                        k += 1;
                    }
                    if k < nprops.min(32) {
                        bail!("short vertex line in PLY data");
                    }
                    Ok(convert(&|i: usize| vals.get(i).copied().unwrap_or(0.0)))
                })
                .collect::<Result<_>>()?
        }
        _ => {
            let mut rec = Record::default();
            let mut rows = Vec::with_capacity(n);
            for _ in 0..n {
                body.record(el, &mut rec)?;
                rows.push(convert(&|i: usize| rec.scalars.get(scalar_slot(el, i)).copied().unwrap_or(0.0)));
            }
            rows
        }
    };

    let mut v = Vertices { positions: Vec::with_capacity(n), ..Default::default() };
    let has_n = layout.normal.is_some();
    let has_c = layout.color.is_some();
    let has_uv = layout.uv.is_some();
    for (p, nrm, c, uv) in rows {
        v.positions.push(p);
        if has_n {
            v.normals.push(nrm.unwrap_or([0.0, 1.0, 0.0]));
        }
        if has_c {
            v.colors.push(c.unwrap_or([255; 4]));
        }
        if has_uv {
            v.uvs.push(uv.unwrap_or([0.0; 2]));
        }
    }
    Ok(v)
}

/// Position of property `i` among the scalar properties of a record.
fn scalar_slot(el: &Element, i: usize) -> usize {
    el.props[..i].iter().filter(|p| matches!(p.kind, Kind::Scalar(_))).count()
}

#[derive(Default)]
struct Faces {
    /// Triangulated vertex indices.
    indices: Vec<u32>,
    /// Per corner texture coordinates (from a `texcoord` list).
    corner_uvs: Vec<[f32; 2]>,
    /// Per triangle colour.
    tri_colors: Vec<[u8; 4]>,
}

fn read_faces(body: &mut Body, el: &Element) -> Result<Faces> {
    let list_pos = |names: &[&str]| -> Option<usize> {
        let i = el.find(names)?;
        matches!(el.props[i].kind, Kind::List { .. }).then(|| el.props[..i].iter().filter(|p| matches!(p.kind, Kind::List { .. })).count())
    };
    let idx_list = list_pos(&["vertex_indices", "vertex_index"]).ok_or_else(|| anyhow!("PLY faces have no vertex_indices"))?;
    let tc_list = list_pos(&["texcoord", "texture_coordinates"]);
    let color = match (el.find(&["red"]), el.find(&["green"]), el.find(&["blue"])) {
        (Some(r), Some(g), Some(b)) => Some([scalar_slot(el, r), scalar_slot(el, g), scalar_slot(el, b)]),
        _ => None,
    };
    let ctype = el.find(&["red"]).map(|i| VertexLayout::scalar_type(el, i)).unwrap_or(Scalar::U8);

    let mut faces = Faces { indices: Vec::with_capacity(el.count * 3), ..Default::default() };
    let mut rec = Record::default();
    for _ in 0..el.count {
        body.record(el, &mut rec)?;
        let idx = &rec.lists[idx_list];
        if idx.len() < 3 {
            continue;
        }
        let tc = tc_list.and_then(|l| rec.lists.get(l)).filter(|t| t.len() >= idx.len() * 2);
        for k in 1..idx.len() - 1 {
            for c in [0, k, k + 1] {
                faces.indices.push(idx[c] as u32);
                if let Some(tc) = tc {
                    faces.corner_uvs.push([tc[2 * c] as f32, 1.0 - tc[2 * c + 1] as f32]);
                }
            }
            if let Some([r, g, b]) = color {
                let s = &rec.scalars;
                faces.tri_colors.push([ctype.color(s[r]), ctype.color(s[g]), ctype.color(s[b]), 255]);
            }
        }
    }
    Ok(faces)
}

pub fn load(ctx: &LoadCtx) -> Result<Scene> {
    let data = std::fs::read(ctx.path).with_context(|| format!("cannot read '{}'", ctx.path.display()))?;
    let header = parse_header(&data)?;
    let mut body = Body { data: &data, pos: header.body_offset, encoding: header.encoding };
    ctx.progress.set(0.05);

    let mut vertices = None;
    let mut faces = None;
    for el in &header.elements {
        ctx.progress.check()?;
        match el.name.as_str() {
            "vertex" if vertices.is_none() => vertices = Some(read_vertices(&mut body, el)?),
            "face" if faces.is_none() && el.count > 0 => faces = Some(read_faces(&mut body, el)?),
            _ => body.skip(el)?,
        }
        ctx.progress.set(0.05 + 0.6 * body.pos as f32 / data.len().max(1) as f32);
    }
    drop(data);
    let v = vertices.ok_or_else(|| anyhow!("the PLY file has no vertices"))?;
    ctx.progress.set_stage("Building mesh");

    let mut scene = Scene { up_axis: UpAxis::Y, ..Default::default() };
    let mut material = Material { name: "PLY".into(), ..Default::default() };

    // Texture referenced through the MeshLab comment.
    if let Some(tex) = &header.texture {
        match util::resolve_texture_path(ctx.dir, &[tex]) {
            Some(path) => {
                if let Some(Some(img)) = util::decode_images(vec![(tex.clone(), ImageSource::File(path))], ctx.warnings()).pop() {
                    scene.images.push(img);
                    scene.textures.push(Texture { image: 0, sampler: Sampler::default() });
                    material.base_color_texture = Some(TextureRef::new(0));
                }
            }
            None => ctx.warn(format!("Texture '{tex}' was not found")),
        }
    }

    let prim = match faces.filter(|f| !f.indices.is_empty()) {
        None => {
            let mut p = Primitive::new(Topology::Points);
            p.positions = v.positions;
            p.normals = v.normals;
            p.colors = v.colors;
            material.unlit = true;
            p
        }
        Some(f) => build_mesh(v, f),
    };
    if !prim.colors.is_empty() {
        material.base_color = [1.0, 1.0, 1.0, 1.0];
    }
    if material.base_color_texture.is_some() {
        material.base_color = [1.0, 1.0, 1.0, 1.0];
    }
    let mut prim = prim;
    prim.material = Some(scene.add_material(material));
    let name = ctx.path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    scene.meshes.push(Mesh { name: name.clone(), primitives: vec![prim] });
    scene.add_node(Node { name, mesh: Some(0), ..Default::default() }, None);
    Ok(scene)
}

fn build_mesh(v: Vertices, mut f: Faces) -> Primitive {
    let n = v.positions.len();
    util::sanitize_indices(&mut f.indices, n, 3);
    let per_corner = !f.corner_uvs.is_empty() && f.corner_uvs.len() == f.indices.len();
    let per_face_color = !f.tri_colors.is_empty() && f.tri_colors.len() * 3 == f.indices.len();
    let mut p = Primitive::new(Topology::Triangles);

    if v.normals.len() == n && !per_corner && !per_face_color {
        // Use the file's normals as they are.
        p.positions = v.positions;
        p.normals = v.normals;
        p.colors = v.colors;
        p.uvs = v.uvs;
        p.indices = f.indices;
        return p;
    }

    // Corner keys keep corners with different UVs/colours apart.
    let keys: Option<Vec<u32>> = (per_corner || per_face_color).then(|| {
        let mut table: std::collections::HashMap<(u64, u32), u32> = std::collections::HashMap::new();
        (0..f.indices.len())
            .map(|c| {
                let uv = if per_corner {
                    let [u, w] = f.corner_uvs[c];
                    (u.to_bits() as u64) << 32 | w.to_bits() as u64
                } else {
                    0
                };
                let col = if per_face_color { u32::from_le_bytes(f.tri_colors[c / 3]) } else { 0 };
                let next = table.len() as u32;
                *table.entry((uv, col)).or_insert(next)
            })
            .collect()
    });
    let s = util::smooth_normals(&v.positions, &f.indices, keys.as_deref(), 60.0);
    let src_vertex = |c: u32| f.indices[c as usize] as usize;
    if per_corner {
        p.uvs = s.source_corner.iter().map(|&c| f.corner_uvs[c as usize]).collect();
    } else if v.uvs.len() == n {
        p.uvs = s.source_corner.iter().map(|&c| v.uvs[src_vertex(c)]).collect();
    }
    if per_face_color {
        p.colors = s.source_corner.iter().map(|&c| f.tri_colors[c as usize / 3]).collect();
    } else if v.colors.len() == n {
        p.colors = s.source_corner.iter().map(|&c| v.colors[src_vertex(c)]).collect();
    }
    if v.normals.len() == n {
        // Keep file normals where present even when splitting vertices.
        p.normals = s.source_corner.iter().map(|&c| v.normals[src_vertex(c)]).collect();
    } else {
        p.normals = s.normals;
    }
    p.positions = s.positions;
    p.indices = s.indices;
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load_bytes(data: &[u8]) -> (Option<Vertices>, Option<Faces>) {
        let header = parse_header(data).unwrap();
        let mut body = Body { data, pos: header.body_offset, encoding: header.encoding };
        let (mut v, mut f) = (None, None);
        for el in &header.elements {
            match el.name.as_str() {
                "vertex" => v = Some(read_vertices(&mut body, el).unwrap()),
                "face" => f = Some(read_faces(&mut body, el).unwrap()),
                _ => body.skip(el).unwrap(),
            }
        }
        (v, f)
    }

    #[test]
    fn ascii_quad_with_colors() {
        let text = b"ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0 255 0 0\n1 0 0 0 255 0\n1 1 0 0 0 255\n0 1 0 255 255 255\n4 0 1 2 3\n";
        let (v, f) = load_bytes(text);
        let v = v.unwrap();
        let f = f.unwrap();
        assert_eq!(v.positions.len(), 4);
        assert_eq!(v.colors[1], [0, 255, 0, 255]);
        assert_eq!(f.indices, vec![0, 1, 2, 0, 2, 3]);
    }

    #[test]
    fn binary_point_cloud_with_extra_element() {
        let mut d = b"ply\r\nformat binary_little_endian 1.0\r\nelement vertex 2\r\nproperty double x\r\nproperty double y\r\nproperty double z\r\nelement camera 1\r\nproperty float view_px\r\nend_header\r\n".to_vec();
        for v in [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0] {
            d.extend(v.to_le_bytes());
        }
        d.extend(7.0f32.to_le_bytes());
        let (v, f) = load_bytes(&d);
        assert_eq!(v.unwrap().positions, vec![[1., 2., 3.], [4., 5., 6.]]);
        assert!(f.is_none());
    }

    #[test]
    fn big_endian_faces() {
        let mut d = b"ply\nformat binary_big_endian 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar uint vertex_indices\nend_header\n".to_vec();
        for v in [0f32, 0., 0., 1., 0., 0., 0., 1., 0.] {
            d.extend(v.to_be_bytes());
        }
        d.push(3);
        for i in [0u32, 1, 2] {
            d.extend(i.to_be_bytes());
        }
        let (_, f) = load_bytes(&d);
        assert_eq!(f.unwrap().indices, vec![0, 1, 2]);
    }
}
