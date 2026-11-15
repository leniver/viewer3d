// Builds a line-list index buffer (3 edges per triangle) from a triangle list.

struct Params {
    triangle_count: u32,
    row_width: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> triangles: array<u32>;
@group(0) @binding(2) var<storage, read_write> lines: array<u32>;

@compute @workgroup_size(64)
fn wire(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x + gid.y * params.row_width;
    if (t >= params.triangle_count) {
        return;
    }
    let a = triangles[t * 3u];
    let b = triangles[t * 3u + 1u];
    let c = triangles[t * 3u + 2u];
    let o = t * 6u;
    lines[o] = a;
    lines[o + 1u] = b;
    lines[o + 2u] = b;
    lines[o + 3u] = c;
    lines[o + 4u] = c;
    lines[o + 5u] = a;
}
