//! Windows integration: per-user "Open with" registration for 3D files.
//!
//! Windows 10/11 do not let programs make themselves the default handler;
//! registering the app makes it appear in "Open with", where the user can pick
//! "Always use this app". Everything is written under HKEY_CURRENT_USER (no
//! administrator rights needed) and can be removed with [`unregister`].

#[cfg(windows)]
mod imp {
    use crate::loaders::{self, Format};
    use std::io;
    use winreg::RegKey;
    use winreg::enums::HKEY_CURRENT_USER;

    const PROG_ID: &str = "Viewer3D.Model";
    const APP_NAME: &str = "3D Viewer";

    fn extensions() -> impl Iterator<Item = &'static str> {
        loaders::SUPPORTED_EXTENSIONS
            .iter()
            .copied()
            .filter(|e| Format::from_extension(e).is_some_and(loaders::format_available))
    }

    fn classes() -> io::Result<RegKey> {
        Ok(RegKey::predef(HKEY_CURRENT_USER).create_subkey("Software\\Classes")?.0)
    }

    fn exe() -> io::Result<(String, String)> {
        let path = std::env::current_exe()?;
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .ok_or_else(|| io::Error::other("cannot determine the executable name"))?;
        Ok((path.to_string_lossy().into_owned(), name))
    }

    pub fn register() -> io::Result<()> {
        let (exe, exe_name) = exe()?;
        let command = format!("\"{exe}\" \"%1\"");
        let classes = classes()?;

        let (prog, _) = classes.create_subkey(PROG_ID)?;
        prog.set_value("", &"3D model")?;
        prog.create_subkey("DefaultIcon")?.0.set_value("", &format!("\"{exe}\",0"))?;
        prog.create_subkey("shell\\open\\command")?.0.set_value("", &command)?;

        let (app, _) = classes.create_subkey(format!("Applications\\{exe_name}"))?;
        app.set_value("FriendlyAppName", &APP_NAME)?;
        app.create_subkey("shell\\open\\command")?.0.set_value("", &command)?;
        let (types, _) = app.create_subkey("SupportedTypes")?;

        for ext in extensions() {
            types.set_value(format!(".{ext}"), &"")?;
            classes.create_subkey(format!(".{ext}\\OpenWithProgids"))?.0.set_value(PROG_ID, &"")?;
        }
        notify_shell();
        Ok(())
    }

    pub fn unregister() -> io::Result<()> {
        let (_, exe_name) = exe()?;
        let classes = classes()?;
        for ext in extensions() {
            if let Ok(key) = classes.open_subkey_with_flags(format!(".{ext}\\OpenWithProgids"), winreg::enums::KEY_ALL_ACCESS) {
                let _ = key.delete_value(PROG_ID);
            }
        }
        let ignore_missing = |r: io::Result<()>| match r {
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
            other => other,
        };
        ignore_missing(classes.delete_subkey_all(PROG_ID))?;
        ignore_missing(classes.delete_subkey_all(format!("Applications\\{exe_name}")))?;
        notify_shell();
        Ok(())
    }

    pub fn is_registered() -> bool {
        classes().is_ok_and(|c| c.open_subkey(format!("{PROG_ID}\\shell\\open\\command")).is_ok())
    }

    /// Opens Settings > Default apps so the user can make it the default.
    pub fn open_default_apps_settings() {
        let _ = std::process::Command::new("explorer.exe").arg("ms-settings:defaultapps").spawn();
    }

    fn notify_shell() {
        unsafe extern "system" {
            fn SHChangeNotify(event: i32, flags: u32, item1: *const core::ffi::c_void, item2: *const core::ffi::c_void);
        }
        const SHCNE_ASSOCCHANGED: i32 = 0x0800_0000;
        const SHCNF_IDLIST: u32 = 0;
        // SAFETY: documented call with null item pointers for this event.
        unsafe { SHChangeNotify(SHCNE_ASSOCCHANGED, SHCNF_IDLIST, std::ptr::null(), std::ptr::null()) };
    }
}

#[cfg(windows)]
pub use imp::*;

/// (working set, private) memory of this process in MB, for diagnostics.
/// Private bytes include commitments for GPU memory; the working set is RAM.
#[cfg(windows)]
pub fn memory_mb() -> (f64, f64) {
    #[repr(C)]
    #[derive(Default)]
    struct Counters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set: usize,
        working_set: usize,
        quota_peak_paged_pool: usize,
        quota_paged_pool: usize,
        quota_peak_non_paged_pool: usize,
        quota_non_paged_pool: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
        private_usage: usize,
    }
    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut core::ffi::c_void;
        fn K32GetProcessMemoryInfo(process: *mut core::ffi::c_void, counters: *mut Counters, cb: u32) -> i32;
    }
    let mut c = Counters { cb: std::mem::size_of::<Counters>() as u32, ..Default::default() };
    // SAFETY: valid pointer to a correctly sized PROCESS_MEMORY_COUNTERS_EX.
    let ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    let mb = |b: usize| b as f64 / (1024.0 * 1024.0);
    if ok != 0 { (mb(c.working_set), mb(c.private_usage)) } else { (0.0, 0.0) }
}

#[cfg(not(windows))]
pub fn memory_mb() -> (f64, f64) {
    (0.0, 0.0)
}

/// Logs the process memory at a loading stage when VIEWER3D_MEMLOG is set.
pub fn memlog(stage: &str) {
    if std::env::var_os("VIEWER3D_MEMLOG").is_some() {
        let (ws, private) = memory_mb();
        log::info!("memory after {stage}: {ws:.0} MB working set, {private:.0} MB private");
    }
}

#[cfg(not(windows))]
mod imp_other {
    use std::io;

    pub fn register() -> io::Result<()> {
        Err(io::Error::other("file registration is only available on Windows"))
    }

    pub fn unregister() -> io::Result<()> {
        register()
    }

    pub fn is_registered() -> bool {
        false
    }

    pub fn open_default_apps_settings() {}
}

#[cfg(not(windows))]
pub use imp_other::*;
