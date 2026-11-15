// GPU vertex deformation: morph targets then linear blend skinning.
// Writes positions + normals straight into the vertex buffer used for drawing.

struct Params {
    vertex_count: u32,
    target_count: u32,
    skinned: u32,
    row_width: u32, // invocations per dispatch row (2D dispatch for huge meshes)
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> base_vertices: array<f32>;   // 6 floats per vertex
@group(0) @binding(2) var<storage, read> morph: array<f32>;           // [target][vertex][6]
@group(0) @binding(3) var<storage, read> weights: array<f32>;         // per target
@group(0) @binding(4) var<storage, read> skin: array<u32>;            // per vertex: joints (2 x u16x2), weights (4 x f32 bits)
@group(0) @binding(5) var<storage, read> joints: array<mat4x4<f32>>;
@group(0) @binding(6) var<storage, read_write> out_vertices: array<f32>;

@compute @workgroup_size(64)
fn deform(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * params.row_width;
    if (i >= params.vertex_count) {
        return;
    }
    let b = i * 6u;
    var p = vec3<f32>(base_vertices[b], base_vertices[b + 1u], base_vertices[b + 2u]);
    var n = vec3<f32>(base_vertices[b + 3u], base_vertices[b + 4u], base_vertices[b + 5u]);

    for (var t = 0u; t < params.target_count; t = t + 1u) {
        let w = weights[t];
        if (w != 0.0) {
            let o = (t * params.vertex_count + i) * 6u;
            p += w * vec3<f32>(morph[o], morph[o + 1u], morph[o + 2u]);
            n += w * vec3<f32>(morph[o + 3u], morph[o + 4u], morph[o + 5u]);
        }
    }

    if (params.skinned != 0u) {
        let s = i * 6u;
        let j01 = skin[s];
        let j23 = skin[s + 1u];
        let w = vec4<f32>(bitcast<f32>(skin[s + 2u]), bitcast<f32>(skin[s + 3u]),
                          bitcast<f32>(skin[s + 4u]), bitcast<f32>(skin[s + 5u]));
        let m = joints[j01 & 0xffffu] * w.x + joints[j01 >> 16u] * w.y
              + joints[j23 & 0xffffu] * w.z + joints[j23 >> 16u] * w.w;
        p = (m * vec4<f32>(p, 1.0)).xyz;
        n = (m * vec4<f32>(n, 0.0)).xyz;
    }

    let len2 = dot(n, n);
    n = select(vec3<f32>(0.0, 1.0, 0.0), n * inverseSqrt(len2), len2 > 1e-20);
    out_vertices[b] = p.x;
    out_vertices[b + 1u] = p.y;
    out_vertices[b + 2u] = p.z;
    out_vertices[b + 3u] = n.x;
    out_vertices[b + 4u] = n.y;
    out_vertices[b + 5u] = n.z;
}
