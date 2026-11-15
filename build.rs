fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=assets/icon.ico");
    let target = std::env::var("TARGET").unwrap_or_default();
    if target.contains("windows") {
        // Executable icon and version information (Explorer, "Open with").
        let mut res = winresource::WindowsResource::new();
        res.set_icon("assets/icon.ico")
            .set("ProductName", "3D Viewer")
            .set("FileDescription", "3D Viewer")
            .set("OriginalFilename", "viewer3d.exe")
            .set("CompanyName", "leniver")
            .set("LegalCopyright", "Copyright (c) 2026 leniver");
        if let Err(e) = res.compile() {
            println!("cargo:warning=could not embed the Windows icon: {e}");
        }
    }
    if target.contains("windows") && target.contains("msvc") {
        // Export the GPU preference symbols defined in main.rs so NVIDIA
        // Optimus / AMD PowerXpress drivers pick the discrete GPU.
        println!("cargo:rustc-link-arg-bins=/EXPORT:NvOptimusEnablement,DATA");
        println!("cargo:rustc-link-arg-bins=/EXPORT:AmdPowerXpressRequestHighPerformance,DATA");
    }
}
