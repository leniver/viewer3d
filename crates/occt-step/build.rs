//! Builds OpenCASCADE (once, shared by all profiles) and the small C++ shim
//! that turns STEP / IGES / BREP files into triangle meshes.

use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
};

/// Bumped when the OCCT build configuration changes, to force a rebuild.
const OCCT_MARKER: &str = ".viewer3d-occt-7.8.1-ok";

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=cpp/occt_shim.cpp");
    println!("cargo:rerun-if-env-changed=CMAKE");

    println!("cargo:rerun-if-env-changed=OCCT_BUILD_JOBS");

    let occt = occt_dir();
    std::fs::create_dir_all(&occt).expect("cannot create the OCCT directory");
    let marker = occt.join(OCCT_MARKER);

    // The OCCT tree is shared by every profile and by IDE `cargo check` runs:
    // hold an exclusive lock so two builds never compile it at the same time
    // (that doubles CPU and memory use and corrupts the build directory).
    let lock = std::fs::File::create(occt.join(".build.lock")).expect("cannot create OCCT lock file");
    lock.lock().expect("cannot lock the OCCT build directory");
    if !marker.exists() {
        ensure_cmake();
        let jobs = build_jobs();
        // SAFETY: build scripts are single-threaded; nothing else reads the
        // environment concurrently.
        unsafe {
            // CMake >= 4 refuses OCCT's `cmake_minimum_required(VERSION 3.1)`.
            env::set_var("CMAKE_POLICY_VERSION_MINIMUM", "3.5");
        }
        if env::var("TARGET").is_ok_and(|t| t.contains("msvc")) {
            prepare_msvc_cmake(jobs);
        } else {
            unsafe { env::set_var("NUM_JOBS", jobs.to_string()) };
        }
        build_occt_into(&occt);
        std::fs::write(&marker, b"ok").expect("cannot write OCCT build marker");
    }
    drop(lock);

    let include_dir = occt.join("include");
    let lib_dir = occt.join("lib");

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .file("cpp/occt_shim.cpp")
        .include(&include_dir)
        .define("OCCT_STATIC_BUILD", None)
        .warnings(false);
    if build.get_compiler().is_like_msvc() {
        build.flag("/std:c++17").flag("/utf-8").flag("/bigobj");
    } else {
        build.flag("-std=c++17");
    }
    build.compile("occt_shim");

    link_occt(&lib_dir);
}

/// Links every OCCT toolkit that was built. MSVC resolves symbols regardless of
/// order; GNU-style linkers get the libraries inside a group.
fn link_occt(lib_dir: &Path) {
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    let target = env::var("TARGET").unwrap_or_default();
    let msvc = target.contains("msvc");

    let mut libs: Vec<String> = std::fs::read_dir(lib_dir)
        .expect("OCCT lib directory missing")
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            if msvc {
                name.strip_suffix(".lib").map(str::to_owned)
            } else {
                name.strip_prefix("lib")
                    .and_then(|n| n.strip_suffix(".a"))
                    .map(str::to_owned)
            }
        })
        .filter(|n| n.starts_with("TK"))
        .collect();
    libs.sort();

    if msvc {
        for lib in &libs {
            println!("cargo:rustc-link-lib=static={lib}");
        }
        for sys in ["user32", "advapi32", "gdi32", "ws2_32", "psapi", "shell32", "ole32"] {
            println!("cargo:rustc-link-lib=dylib={sys}");
        }
    } else {
        println!("cargo:rustc-link-arg=-Wl,--start-group");
        for lib in &libs {
            println!("cargo:rustc-link-arg={}", lib_dir.join(format!("lib{lib}.a")).display());
        }
        println!("cargo:rustc-link-arg=-Wl,--end-group");
        let cxx = if target.contains("apple") { "c++" } else { "stdc++" };
        println!("cargo:rustc-link-lib=dylib={cxx}");
        if !target.contains("apple") {
            println!("cargo:rustc-link-lib=dylib=pthread");
            println!("cargo:rustc-link-lib=dylib=dl");
        }
    }
}

/// MSVC-only fixes for the OCCT CMake build:
/// - With the Visual Studio generator the `cmake` crate overrides
///   CMAKE_<LANG>_FLAGS_RELEASE with flags that have no optimisation level (it
///   also filters /O flags out of CFLAGS), which would leave OCCT unoptimised.
///   A toolchain file runs after those cache entries are set and restores them.
/// - Pins the Visual Studio instance to the stable one found by vswhere, so a
///   preview install does not compile OCCT with a newer toolset than the one
///   that links the final executable.
/// - Bounds parallelism: by default MSBuild builds `NUM_JOBS` projects at once
///   and OCCT adds `/MP` (one compiler per core) to each, i.e. up to cores²
///   compiler processes. Build one project at a time with `/MP<jobs>` instead;
///   the release flags come after OCCT's `/MP` on the command line, so the
///   bounded value wins.
fn prepare_msvc_cmake(jobs: usize) {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let toolchain = out_dir.join("occt-msvc-release.cmake");
    let flags = format!("/nologo /MD /O2 /Ob2 /DNDEBUG /Brepro /MP{jobs}");
    let content = format!(
        "set(CMAKE_C_FLAGS_RELEASE \"{flags}\" CACHE STRING \"\" FORCE)\n\
         set(CMAKE_CXX_FLAGS_RELEASE \"{flags}\" CACHE STRING \"\" FORCE)\n"
    );
    std::fs::write(&toolchain, content).expect("cannot write CMake toolchain file");
    // SAFETY: see main().
    unsafe {
        env::set_var("NUM_JOBS", "1");
        env::set_var("CMAKE_TOOLCHAIN_FILE", toolchain.to_string_lossy().replace('\\', "/"));
        if env::var_os("CMAKE_GENERATOR_INSTANCE").is_none()
            && let Some(vs) = find_vs_installation() {
                env::set_var("CMAKE_GENERATOR_INSTANCE", vs);
            }
    }
}

/// Resolves `.` and `..` components without touching the file system.
/// Where OCCT is built and installed: `OCCT` in the build directory root
/// (normally `target`; `target/<triple>` for cross builds), so every profile
/// shares one build. The root is found above the `build` folder that holds the
/// build-script outputs, whatever Cargo's layout below it:
/// `<profile>/build/<crate>-<hash>/out` until Cargo 1.9x, then
/// `<profile>/build/<crate>/<hash>/out`.
///
/// The path is cleaned up lexically only: canonicalize() yields `\\?\` paths on
/// Windows, which cl.exe does not accept as include directories.
fn occt_dir() -> PathBuf {
    let out_dir = normalize(Path::new(&env::var_os("OUT_DIR").expect("OUT_DIR is not set")));
    out_dir
        .ancestors()
        .find(|dir| dir.file_name().is_some_and(|name| name == "build"))
        .and_then(|build| build.parent()?.parent())
        .map(|root| root.join("OCCT"))
        // Unknown layout: occt-sys's own choice (may be one build per profile).
        .unwrap_or_else(|| normalize(&occt_sys::occt_path()))
}

/// Runs `occt_sys::build_occt()` so that it installs into `occt`.
///
/// occt-sys installs into `OUT_DIR/../../../../OCCT`, which assumes Cargo's
/// former layout; with the newer one it lands in each profile's folder (one
/// OCCT build per profile, outside `occt`). OUT_DIR is therefore pointed four
/// levels below `occt`'s parent for the duration of the call.
fn build_occt_into(occt: &Path) {
    assert_eq!(occt.file_name().and_then(|n| n.to_str()), Some("OCCT"), "unexpected OCCT folder {}", occt.display());
    // Real folders: on Linux, `..` is resolved through existing directories.
    let anchor = occt.join("out-dir-anchor").join("a").join("b");
    std::fs::create_dir_all(&anchor).expect("cannot create the OCCT out-dir anchor");
    let real_out_dir = env::var_os("OUT_DIR").expect("OUT_DIR is not set");
    // SAFETY: build scripts are single-threaded; nothing else reads the
    // environment concurrently.
    unsafe { env::set_var("OUT_DIR", &anchor) };
    assert_eq!(normalize(&occt_sys::occt_path()), occt, "occt-sys no longer installs into OUT_DIR/../../../../OCCT");
    occt_sys::build_occt();
    unsafe { env::set_var("OUT_DIR", real_out_dir) };
}

fn normalize(path: &Path) -> PathBuf {
    use std::path::Component;
    let mut out = PathBuf::new();
    for c in path.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Number of compiler processes for the OCCT build: `OCCT_BUILD_JOBS` if set,
/// otherwise the jobs cargo grants this build (`NUM_JOBS`, which honours
/// `build.jobs` in .cargo/config.toml), otherwise all logical CPUs but two.
fn build_jobs() -> usize {
    let parse = |name: &str| env::var(name).ok().and_then(|v| v.trim().parse::<usize>().ok());
    let fallback = std::thread::available_parallelism().map_or(1, |n| n.get().saturating_sub(2));
    parse("OCCT_BUILD_JOBS").or_else(|| parse("NUM_JOBS")).unwrap_or(fallback).max(1)
}

/// The `cmake` crate needs a cmake executable. If none is on the PATH, use the
/// one bundled with Visual Studio (found through vswhere).
fn ensure_cmake() {
    if env::var_os("CMAKE").is_some() {
        return;
    }
    if Command::new("cmake").arg("--version").output().is_ok() {
        return;
    }
    if let Some(path) = find_vs_cmake() {
        // SAFETY: build scripts are single-threaded at this point.
        unsafe { env::set_var("CMAKE", &path) };
        return;
    }
    panic!(
        "CMake was not found. Install CMake (https://cmake.org) or the \
         'C++ CMake tools' Visual Studio component, or set the CMAKE environment variable."
    );
}

fn find_vs_cmake() -> Option<PathBuf> {
    let install = find_vs_installation()?;
    let cmake = install.join(r"Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe");
    cmake.exists().then_some(cmake)
}

/// Latest stable (non-preview) Visual Studio installation with the C++ tools.
fn find_vs_installation() -> Option<PathBuf> {
    if !cfg!(windows) {
        return None;
    }
    let program_files = env::var_os("ProgramFiles(x86)")?;
    let vswhere = Path::new(&program_files).join("Microsoft Visual Studio/Installer/vswhere.exe");
    let out = Command::new(vswhere)
        .args([
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ])
        .output()
        .ok()?;
    let install = String::from_utf8_lossy(&out.stdout).lines().next()?.trim().to_owned();
    (!install.is_empty()).then(|| PathBuf::from(install))
}
