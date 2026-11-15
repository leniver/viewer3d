// Scene rendering: PBR surfaces, lines, wireframe, points, grid, background.

struct Frame {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    key_dir: vec4<f32>,      // xyz: direction towards the light, w: intensity
    key_color: vec4<f32>,
    fill_dir: vec4<f32>,
    fill_color: vec4<f32>,
    sky_color: vec4<f32>,    // w: ambient intensity
    ground_color: vec4<f32>,
    bg_top: vec4<f32>,
    bg_bottom: vec4<f32>,
    grid_params: vec4<f32>,  // x: height, y: cell size, z: fade radius, w: unused
    grid_center: vec4<f32>,  // x, z: centre, w: half extent
    grid_color: vec4<f32>,
    wire_color: vec4<f32>,
    viewport: vec4<f32>,     // width, height, point size (px), flat shading
    params: vec4<f32>,       // exposure, line depth bias, unused, unused
};

@group(0) @binding(0) var<uniform> frame: Frame;

struct MaterialU {
    base_color: vec4<f32>,
    emissive: vec4<f32>,
    params: vec4<f32>,       // metallic, roughness, normal scale, occlusion strength
    params2: vec4<f32>,      // alpha cutoff, alpha mode (0 opaque, 1 mask, 2 blend), unlit, glossiness
    channels: vec4<u32>,     // metallic channel, roughness channel, texture mask, unused
    uv: array<vec4<f32>, 12>,// per texture slot: two rows of a 2x3 UV transform
};

@group(1) @binding(0) var<uniform> material: MaterialU;
@group(1) @binding(1) var t_base: texture_2d<f32>;
@group(1) @binding(2) var t_metal: texture_2d<f32>;
@group(1) @binding(3) var t_rough: texture_2d<f32>;
@group(1) @binding(4) var t_normal: texture_2d<f32>;
@group(1) @binding(5) var t_occl: texture_2d<f32>;
@group(1) @binding(6) var t_emis: texture_2d<f32>;
@group(1) @binding(7) var s_base: sampler;
@group(1) @binding(8) var s_metal: sampler;
@group(1) @binding(9) var s_rough: sampler;
@group(1) @binding(10) var s_normal: sampler;
@group(1) @binding(11) var s_occl: sampler;
@group(1) @binding(12) var s_emis: sampler;

struct DrawU {
    model: mat4x4<f32>,
    normal0: vec4<f32>,
    normal1: vec4<f32>,
    normal2: vec4<f32>,
    flags: vec4<u32>,
};

@group(2) @binding(0) var<uniform> draw: DrawU;

const PI: f32 = 3.14159265359;

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
};

fn transform_vertex(pos: vec3<f32>, nrm: vec3<f32>, uv: vec2<f32>, color: vec4<f32>) -> VOut {
    var o: VOut;
    let world = draw.model * vec4<f32>(pos, 1.0);
    o.world = world.xyz;
    o.clip = frame.view_proj * world;
    let nm = mat3x3<f32>(draw.normal0.xyz, draw.normal1.xyz, draw.normal2.xyz);
    o.normal = nm * nrm;
    o.uv = uv;
    o.color = vec4<f32>(srgb_to_linear(color.rgb), color.a);
    return o;
}

// ---------------------------------------------------------------- surfaces

@vertex
fn vs_full(@location(0) pos: vec3<f32>, @location(1) nrm: vec3<f32>,
           @location(2) uv: vec2<f32>, @location(3) color: vec4<f32>) -> VOut {
    return transform_vertex(pos, nrm, uv, color);
}

@vertex
fn vs_basic(@location(0) pos: vec3<f32>, @location(1) nrm: vec3<f32>) -> VOut {
    return transform_vertex(pos, nrm, vec2<f32>(0.0), vec4<f32>(1.0));
}

fn tex_uv(slot: u32, uv: vec2<f32>) -> vec2<f32> {
    let r0 = material.uv[slot * 2u];
    let r1 = material.uv[slot * 2u + 1u];
    let h = vec3<f32>(uv, 1.0);
    return vec2<f32>(dot(r0.xyz, h), dot(r1.xyz, h));
}

fn has_tex(slot: u32) -> bool {
    return (material.channels.z & (1u << slot)) != 0u;
}

fn d_ggx(nh: f32, a: f32) -> f32 {
    let a2 = a * a;
    let d = nh * nh * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

fn v_smith(nv: f32, nl: f32, a: f32) -> f32 {
    let a2 = a * a;
    let gv = nl * sqrt(nv * nv * (1.0 - a2) + a2);
    let gl = nv * sqrt(nl * nl * (1.0 - a2) + a2);
    return 0.5 / max(gv + gl, 1e-5);
}

fn f_schlick(f0: vec3<f32>, vh: f32) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - vh, 5.0);
}

// Analytic approximation of the split-sum environment BRDF (Karis).
fn env_brdf(f0: vec3<f32>, rough: f32, nv: f32) -> vec3<f32> {
    let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let r = rough * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * nv)) * r.x + r.y;
    let ab = vec2<f32>(-1.04, 1.04) * a004 + r.zw;
    return f0 * ab.x + vec3<f32>(ab.y);
}

// Khronos PBR Neutral tone mapper: keeps base colours faithful.
fn tonemap(c_in: vec3<f32>) -> vec3<f32> {
    let start = 0.8 - 0.04;
    let desat = 0.15;
    let x = min(c_in.r, min(c_in.g, c_in.b));
    let offset = select(0.04, x - 6.25 * x * x, x < 0.08);
    var c = c_in - vec3<f32>(offset);
    let peak = max(c.r, max(c.g, c.b));
    if (peak < start) {
        return c;
    }
    let d = 1.0 - start;
    let new_peak = 1.0 - d * d / (peak + d - start);
    c = c * (new_peak / peak);
    let g = 1.0 - 1.0 / (desat * (peak - new_peak) + 1.0);
    return mix(c, vec3<f32>(new_peak), g);
}

fn environment(dir: vec3<f32>) -> vec3<f32> {
    let t = smoothstep(-0.6, 0.8, dir.y);
    return mix(frame.ground_color.rgb, frame.sky_color.rgb, t);
}

fn direct_light(n: vec3<f32>, v: vec3<f32>, nv: f32, l: vec3<f32>, radiance: vec3<f32>,
                diffuse: vec3<f32>, f0: vec3<f32>, a: f32) -> vec3<f32> {
    let nl = dot(n, l);
    if (nl <= 0.0) {
        return vec3<f32>(0.0);
    }
    let h = normalize(l + v);
    let nh = max(dot(n, h), 0.0);
    let vh = max(dot(v, h), 0.0);
    let f = f_schlick(f0, vh);
    let spec = d_ggx(nh, a) * v_smith(nv, nl, a) * f;
    let diff = (vec3<f32>(1.0) - f) * diffuse / PI;
    return (diff + spec) * radiance * nl;
}

@fragment
fn fs_main(in: VOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    // Derivatives and texture samples first, in uniform control flow.
    let dp1 = dpdx(in.world);
    let dp2 = dpdy(in.world);
    let uv_n = tex_uv(3u, in.uv);
    let duv1 = dpdx(uv_n);
    let duv2 = dpdy(uv_n);
    let base_tex = textureSample(t_base, s_base, tex_uv(0u, in.uv));
    let metal_tex = textureSample(t_metal, s_metal, tex_uv(1u, in.uv));
    let rough_tex = textureSample(t_rough, s_rough, tex_uv(2u, in.uv));
    let normal_tex = textureSample(t_normal, s_normal, uv_n);
    let occl_tex = textureSample(t_occl, s_occl, tex_uv(4u, in.uv));
    let emis_tex = textureSample(t_emis, s_emis, tex_uv(5u, in.uv));

    let base = material.base_color * in.color * base_tex;
    let alpha_mode = material.params2.y;
    if (alpha_mode > 0.5 && alpha_mode < 1.5 && base.a < material.params2.x) {
        discard;
    }
    let out_alpha = select(1.0, base.a, alpha_mode > 1.5);
    let emissive = material.emissive.rgb * emis_tex.rgb;

    if (material.params2.z > 0.5 || frame.viewport.w > 0.5) {
        return vec4<f32>(tonemap((base.rgb + emissive) * frame.params.x), out_alpha);
    }

    var n = normalize(in.normal);
    if (!front) {
        n = -n;
    }
    if (has_tex(3u)) {
        let tn = vec3<f32>((normal_tex.xy * 2.0 - 1.0) * material.params.z, normal_tex.z * 2.0 - 1.0);
        let dp2perp = cross(dp2, n);
        let dp1perp = cross(n, dp1);
        let t = dp2perp * duv1.x + dp1perp * duv2.x;
        let b = dp2perp * duv1.y + dp1perp * duv2.y;
        let len2 = max(dot(t, t), dot(b, b));
        if (len2 > 1e-24) {
            let inv = inverseSqrt(len2);
            // glTF: +Y of the normal map points towards decreasing v.
            let tbn = mat3x3<f32>(t * inv, -b * inv, n);
            n = normalize(tbn * tn);
        }
    }

    var metallic = material.params.x * metal_tex[material.channels.x];
    var rough = material.params.y * rough_tex[material.channels.y];
    if (material.params2.w > 0.5) {
        rough = 1.0 - rough;
    }
    metallic = clamp(metallic, 0.0, 1.0);
    rough = clamp(rough, 0.045, 1.0);
    let a = rough * rough;

    let v = normalize(frame.camera_pos.xyz - in.world);
    let nv = max(dot(n, v), 1e-4);
    let f0 = mix(vec3<f32>(0.04), base.rgb, metallic);
    let diffuse = base.rgb * (1.0 - metallic);

    var color = direct_light(n, v, nv, frame.key_dir.xyz, frame.key_color.rgb * frame.key_dir.w, diffuse, f0, a);
    color += direct_light(n, v, nv, frame.fill_dir.xyz, frame.fill_color.rgb * frame.fill_dir.w, diffuse, f0, a);

    let ao = select(1.0, mix(1.0, occl_tex.r, material.params.w), has_tex(4u));
    let irradiance = environment(n);
    let r = reflect(-v, n);
    let specular_env = mix(environment(r), irradiance, rough);
    color += (diffuse * irradiance + specular_env * env_brdf(f0, rough, nv)) * ao * frame.sky_color.w;
    color += emissive;

    return vec4<f32>(tonemap(color * frame.params.x), out_alpha);
}

// ------------------------------------------------------------------- lines

fn biased(o_in: VOut) -> VOut {
    var o = o_in;
    // Reverse-Z: a larger depth is closer; pull lines slightly towards the eye.
    o.clip.z = o.clip.z * (1.0 + frame.params.y);
    return o;
}

@vertex
fn vs_lines_full(@location(0) pos: vec3<f32>, @location(3) color: vec4<f32>) -> VOut {
    return biased(transform_vertex(pos, vec3<f32>(0.0, 1.0, 0.0), vec2<f32>(0.0), color));
}

@vertex
fn vs_lines_basic(@location(0) pos: vec3<f32>) -> VOut {
    return biased(transform_vertex(pos, vec3<f32>(0.0, 1.0, 0.0), vec2<f32>(0.0), vec4<f32>(1.0)));
}

@fragment
fn fs_lines(in: VOut) -> @location(0) vec4<f32> {
    let c = material.base_color * in.color;
    return vec4<f32>(tonemap(c.rgb), c.a);
}

@fragment
fn fs_wire(in: VOut) -> @location(0) vec4<f32> {
    return frame.wire_color;
}

// ------------------------------------------------------------------ points

fn point_vertex(vi: u32, pos: vec3<f32>, nrm: vec3<f32>, color: vec4<f32>) -> VOut {
    let corner = vec2<f32>(f32(vi & 1u), f32(vi >> 1u)) * 2.0 - 1.0;
    var o = transform_vertex(pos, nrm, corner, color);
    let size = frame.viewport.z;
    o.clip = vec4<f32>(o.clip.xy + corner * size / frame.viewport.xy * o.clip.w, o.clip.zw);
    return o;
}

@vertex
fn vs_points_full(@builtin(vertex_index) vi: u32, @location(0) pos: vec3<f32>,
                  @location(1) nrm: vec3<f32>, @location(3) color: vec4<f32>) -> VOut {
    return point_vertex(vi, pos, nrm, color);
}

@vertex
fn vs_points_basic(@builtin(vertex_index) vi: u32, @location(0) pos: vec3<f32>,
                   @location(1) nrm: vec3<f32>) -> VOut {
    return point_vertex(vi, pos, nrm, vec4<f32>(1.0));
}

@fragment
fn fs_points(in: VOut) -> @location(0) vec4<f32> {
    if (dot(in.uv, in.uv) > 1.0) {
        discard;
    }
    let c = material.base_color * in.color;
    return vec4<f32>(tonemap(c.rgb), 1.0);
}

// -------------------------------------------------------------------- grid

struct GridOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
};

@vertex
fn vs_grid(@builtin(vertex_index) vi: u32) -> GridOut {
    let corner = vec2<f32>(f32(vi & 1u), f32(vi >> 1u)) * 2.0 - 1.0;
    let ext = frame.grid_center.w;
    let world = vec3<f32>(frame.grid_center.x + corner.x * ext, frame.grid_params.x, frame.grid_center.z + corner.y * ext);
    var o: GridOut;
    o.world = world;
    o.clip = frame.view_proj * vec4<f32>(world, 1.0);
    return o;
}

fn grid_lines(coord: vec2<f32>, spacing: f32) -> f32 {
    let c = coord / spacing;
    let w = max(fwidth(c), vec2<f32>(1e-6));
    let g = abs(fract(c - 0.5) - 0.5) / w;
    let line = 1.0 - min(min(g.x, g.y), 1.0);
    // Fade lines out where cells get smaller than a few pixels (no moire).
    return line * (1.0 - smoothstep(0.15, 0.45, max(w.x, w.y)));
}

@fragment
fn fs_grid(in: GridOut) -> @location(0) vec4<f32> {
    let cell = frame.grid_params.y;
    let minor = grid_lines(in.world.xz, cell);
    let major = grid_lines(in.world.xz, cell * 10.0);
    let dist = length(in.world.xz - frame.grid_center.xz);
    let fade = 1.0 - smoothstep(frame.grid_params.z * 0.25, frame.grid_params.z, dist);
    let a = max(minor * 0.35, major) * fade * frame.grid_color.a;
    if (a < 0.002) {
        discard;
    }
    return vec4<f32>(frame.grid_color.rgb, a);
}

// -------------------------------------------------------------- background

struct BgOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_background(@builtin(vertex_index) vi: u32) -> BgOut {
    let p = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u)) * 2.0 - 1.0;
    var o: BgOut;
    o.clip = vec4<f32>(p, 0.0, 1.0);
    o.uv = p * 0.5 + 0.5;
    return o;
}

@fragment
fn fs_background(in: BgOut) -> @location(0) vec4<f32> {
    let t = smoothstep(0.0, 1.0, in.uv.y);
    let vignette = 1.0 - 0.18 * dot(in.uv - 0.5, in.uv - 0.5);
    return vec4<f32>(mix(frame.bg_bottom.rgb, frame.bg_top.rgb, t) * vignette, 1.0);
}
