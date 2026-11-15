//! Tessellation of CAD files (STEP, IGES, BREP) through OpenCASCADE.
//!
//! The heavy lifting happens in `cpp/occt_shim.cpp`; this module copies its
//! result into plain Rust data.

use std::ffi::{CStr, CString, c_char, c_void};
use std::path::Path;

#[repr(C)]
struct OcctGroup {
    first_index: u32,
    index_count: u32,
    rgba: [f32; 4],
    has_color: i32,
}

#[repr(C)]
struct OcctMeshData {
    positions: *const f32,
    normals: *const f32,
    vertex_count: u32,
    indices: *const u32,
    index_count: u32,
    groups: *const OcctGroup,
    group_count: u32,
    edge_positions: *const f32,
    edge_vertex_count: u32,
}

#[repr(C)]
struct OcctInstanceData {
    mesh: u32,
    matrix: [f64; 12],
    name: *const c_char,
}

#[repr(C)]
struct OcctResult {
    _private: [u8; 0],
}

type ProgressFn = unsafe extern "C" fn(user: *mut c_void, fraction: f64) -> i32;

unsafe extern "C" {
    fn occt_shim_load(
        path: *const c_char,
        format: i32,
        lin_rel: f64,
        angle: f64,
        progress: Option<ProgressFn>,
        user: *mut c_void,
        err: *mut c_char,
        err_len: usize,
    ) -> *mut OcctResult;
    fn occt_shim_mesh_count(r: *const OcctResult) -> u32;
    fn occt_shim_mesh(r: *const OcctResult, i: u32) -> OcctMeshData;
    fn occt_shim_instance_count(r: *const OcctResult) -> u32;
    fn occt_shim_instance(r: *const OcctResult, i: u32) -> OcctInstanceData;
    fn occt_shim_free(r: *mut OcctResult);
}

/// CAD exchange format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Step,
    Iges,
    Brep,
}

/// Tessellation quality.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// Maximum chordal deviation, relative to the model's bounding-box diagonal.
    pub linear_deflection: f64,
    /// Maximum angle between adjacent facets, in radians.
    pub angular_deflection: f64,
}

impl Default for Options {
    fn default() -> Self {
        Self { linear_deflection: 0.0005, angular_deflection: 0.3 }
    }
}

/// Triangles sharing one colour (`None` = no colour in the file).
#[derive(Clone, Debug)]
pub struct Group {
    pub first_index: u32,
    pub index_count: u32,
    /// Linear RGB + alpha.
    pub color: Option<[f32; 4]>,
}

/// One tessellated part definition, shared by all its instances.
#[derive(Clone, Debug, Default)]
pub struct Mesh {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
    pub groups: Vec<Group>,
    /// Feature edges as a line list.
    pub edges: Vec<[f32; 3]>,
}

/// Placement of a part in the assembly.
#[derive(Clone, Debug)]
pub struct Instance {
    pub mesh: usize,
    /// Row-major 3x4 affine matrix (part to model space).
    pub matrix: [[f64; 4]; 3],
    pub name: String,
}

#[derive(Clone, Debug, Default)]
pub struct Model {
    pub meshes: Vec<Mesh>,
    pub instances: Vec<Instance>,
}

/// Error returned when loading fails or is cancelled.
#[derive(Debug)]
pub enum Error {
    Cancelled,
    Failed(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Cancelled => f.write_str("loading cancelled"),
            Error::Failed(msg) => f.write_str(msg),
        }
    }
}

impl std::error::Error for Error {}

struct ProgressCtx<'a> {
    callback: &'a (dyn Fn(f64) -> bool + Sync),
}

/// Called by OCCT (possibly from its worker threads, but serialised by OCCT).
unsafe extern "C" fn progress_trampoline(user: *mut c_void, fraction: f64) -> i32 {
    // SAFETY: `user` points to the ProgressCtx living on `load`'s stack for the
    // whole duration of the native call.
    let ctx = unsafe { &*(user as *const ProgressCtx<'_>) };
    let keep_going = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (ctx.callback)(fraction)))
        .unwrap_or(false);
    if keep_going { 0 } else { 1 }
}

/// Reads and tessellates a CAD file. `progress` receives a fraction in [0, 1]
/// and returns `false` to cancel.
pub fn load(
    path: &Path,
    format: Format,
    options: &Options,
    progress: &(dyn Fn(f64) -> bool + Sync),
) -> Result<Model, Error> {
    let utf8 = path
        .to_str()
        .ok_or_else(|| Error::Failed("the file path is not valid Unicode".into()))?;
    let c_path = CString::new(utf8).map_err(|_| Error::Failed("invalid file path".into()))?;
    let format_code = match format {
        Format::Step => 0,
        Format::Iges => 1,
        Format::Brep => 2,
    };
    let ctx = ProgressCtx { callback: progress };
    let mut err = vec![0 as c_char; 1024];

    // SAFETY: all pointers are valid for the duration of the call; the shim
    // copies the path and never keeps `ctx` after returning.
    let raw = unsafe {
        occt_shim_load(
            c_path.as_ptr(),
            format_code,
            options.linear_deflection,
            options.angular_deflection,
            Some(progress_trampoline),
            &ctx as *const ProgressCtx<'_> as *mut c_void,
            err.as_mut_ptr(),
            err.len(),
        )
    };
    if raw.is_null() {
        // SAFETY: the shim always NUL-terminates the error buffer.
        let msg = unsafe { CStr::from_ptr(err.as_ptr()) }.to_string_lossy().into_owned();
        return Err(if msg == "cancelled" { Error::Cancelled } else { Error::Failed(msg) });
    }

    struct Guard(*mut OcctResult);
    impl Drop for Guard {
        fn drop(&mut self) {
            // SAFETY: pointer returned by occt_shim_load, freed exactly once.
            unsafe { occt_shim_free(self.0) }
        }
    }
    let guard = Guard(raw);
    // SAFETY: `guard.0` stays valid until the guard drops at the end of scope;
    // every slice is built from pointer/length pairs reported by the shim.
    unsafe { Ok(copy_model(guard.0)) }
}

unsafe fn slice<'a, T>(ptr: *const T, len: usize) -> &'a [T] {
    if ptr.is_null() || len == 0 { &[] } else { unsafe { std::slice::from_raw_parts(ptr, len) } }
}

unsafe fn vec3s(ptr: *const f32, count: u32) -> Vec<[f32; 3]> {
    let flat = unsafe { slice(ptr, count as usize * 3) };
    flat.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
}

unsafe fn copy_model(r: *const OcctResult) -> Model {
    let mut model = Model::default();
    let mesh_count = unsafe { occt_shim_mesh_count(r) };
    model.meshes.reserve(mesh_count as usize);
    for i in 0..mesh_count {
        let d = unsafe { occt_shim_mesh(r, i) };
        let groups = unsafe { slice(d.groups, d.group_count as usize) }
            .iter()
            .map(|g| Group {
                first_index: g.first_index,
                index_count: g.index_count,
                color: (g.has_color != 0).then_some(g.rgba),
            })
            .collect();
        model.meshes.push(Mesh {
            positions: unsafe { vec3s(d.positions, d.vertex_count) },
            normals: unsafe { vec3s(d.normals, d.vertex_count) },
            indices: unsafe { slice(d.indices, d.index_count as usize) }.to_vec(),
            groups,
            edges: unsafe { vec3s(d.edge_positions, d.edge_vertex_count) },
        });
    }
    let instance_count = unsafe { occt_shim_instance_count(r) };
    model.instances.reserve(instance_count as usize);
    for i in 0..instance_count {
        let d = unsafe { occt_shim_instance(r, i) };
        let m = d.matrix;
        let name = if d.name.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(d.name) }.to_string_lossy().into_owned()
        };
        model.instances.push(Instance {
            mesh: d.mesh as usize,
            matrix: [[m[0], m[1], m[2], m[3]], [m[4], m[5], m[6], m[7]], [m[8], m[9], m[10], m[11]]],
            name,
        });
    }
    model
}
