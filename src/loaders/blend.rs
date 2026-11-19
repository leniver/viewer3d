//! Blender files (.blend), opened through an installed Blender.
//!
//! A .blend file is a dump of Blender's memory whose layout changes with every
//! version, so only Blender reads it reliably. Blender is started in the
//! background to export the file to glTF (see `blend_export.py`), then the glTF
//! loader reads the result. Blender runs with factory settings and with Python
//! auto-run disabled: neither the user's add-ons nor scripts stored in the
//! file are executed.

use super::{LoadCtx, gltf};
use crate::scene::Scene;
use anyhow::{Context, Result, anyhow, bail};
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::{Duration, SystemTime};

/// Exporter script run inside Blender.
const SCRIPT: &str = include_str!("blend_export.py");

/// Environment variable naming the Blender program to use.
pub const BLENDER_ENV: &str = "VIEWER3D_BLENDER";

/// No usable Blender: the user has to install or choose one.
#[derive(Debug)]
pub struct BlenderNotFound(pub String);

impl std::fmt::Display for BlenderNotFound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for BlenderNotFound {}

#[derive(Clone, Debug, PartialEq)]
pub struct Blender {
    pub path: PathBuf,
    /// (major, minor, patch) read from the program file, when available.
    pub version: Option<(u32, u32, u32)>,
}

impl Blender {
    pub fn label(&self) -> String {
        match self.version {
            Some((a, b, c)) => format!("Blender {a}.{b}.{c}"),
            None => "Blender".to_owned(),
        }
    }
}

/// The Blender to use: `chosen` (the user's choice) if given, else the program
/// named by `VIEWER3D_BLENDER`, else the newest installed version.
pub fn find_blender(chosen: Option<&Path>) -> Result<Blender, BlenderNotFound> {
    if let Some(p) = chosen {
        return describe(p).ok_or_else(|| {
            BlenderNotFound(format!("The Blender chosen for .blend files was not found ({}). Choose it again.", p.display()))
        });
    }
    if let Some(p) = std::env::var_os(BLENDER_ENV).filter(|v| !v.is_empty()) {
        let p = PathBuf::from(p);
        return describe(&p)
            .ok_or_else(|| BlenderNotFound(format!("{BLENDER_ENV} names a program that does not exist ({})", p.display())));
    }
    newest(candidates()).ok_or_else(|| {
        BlenderNotFound(
            "Opening .blend files needs Blender, which was not found. Install it from blender.org, \
             or choose its program file (blender.exe)."
                .to_owned(),
        )
    })
}

/// Checks that `path` is a Blender program (the launcher is replaced by the
/// program it starts, whose output and exit code the viewer needs).
pub fn describe(path: &Path) -> Option<Blender> {
    let path = program(path);
    path.is_file().then(|| Blender { version: file_version(&path), path })
}

/// True when `path` looks like Blender's program file (used to check a choice).
pub fn is_blender_program(path: &Path) -> bool {
    program(path).file_stem().is_some_and(|s| s.to_string_lossy().to_ascii_lowercase().starts_with("blender"))
        && describe(path).is_some()
}

fn program(path: &Path) -> PathBuf {
    let launcher = path
        .file_name()
        .is_some_and(|n| n.to_string_lossy().eq_ignore_ascii_case("blender-launcher.exe"));
    if launcher { path.with_file_name("blender.exe") } else { path.to_path_buf() }
}

/// The highest version among the candidates that exist (the first one wins a tie).
fn newest(candidates: Vec<PathBuf>) -> Option<Blender> {
    let mut seen = Vec::new();
    let mut best: Option<Blender> = None;
    for c in candidates {
        let key = program(&c).to_string_lossy().to_lowercase();
        if seen.contains(&key) {
            continue;
        }
        seen.push(key);
        if let Some(b) = describe(&c)
            && best.as_ref().is_none_or(|cur| b.version > cur.version)
        {
            best = Some(b);
        }
    }
    best
}

fn path_candidates(name: &str) -> Vec<PathBuf> {
    std::env::var_os("PATH").map(|p| std::env::split_paths(&p).map(|d| d.join(name)).collect()).unwrap_or_default()
}

#[cfg(windows)]
fn candidates() -> Vec<PathBuf> {
    let mut out = registry_candidates();
    for var in ["ProgramW6432", "ProgramFiles", "ProgramFiles(x86)"] {
        let Some(dir) = std::env::var_os(var) else { continue };
        let dir = PathBuf::from(dir);
        if let Ok(entries) = std::fs::read_dir(dir.join("Blender Foundation")) {
            out.extend(entries.flatten().map(|e| e.path().join("blender.exe")));
        }
        out.push(dir.join("Steam\\steamapps\\common\\Blender\\blender.exe"));
    }
    out.extend(path_candidates("blender.exe"));
    out
}

/// Blender's installations known to Windows: the program that opens .blend
/// files, then the installed programs list.
#[cfg(windows)]
fn registry_candidates() -> Vec<PathBuf> {
    use winreg::RegKey;
    use winreg::enums::{HKEY_CLASSES_ROOT, HKEY_CURRENT_USER, HKEY_LOCAL_MACHINE};
    let mut out = Vec::new();
    let classes = RegKey::predef(HKEY_CLASSES_ROOT);
    let mut prog_ids = Vec::new();
    if let Ok(id) = classes.open_subkey(".blend").and_then(|k| k.get_value::<String, _>("")) {
        prog_ids.push(id);
    }
    prog_ids.push("blendfile".to_owned());
    for id in prog_ids {
        if let Ok(command) = classes.open_subkey(format!("{id}\\shell\\open\\command")).and_then(|k| k.get_value::<String, _>(""))
            && let Some(p) = command_program(&command)
        {
            out.push(p);
        }
    }
    for (root, key) in [
        (HKEY_LOCAL_MACHINE, "SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall"),
        (HKEY_LOCAL_MACHINE, "SOFTWARE\\WOW6432Node\\Microsoft\\Windows\\CurrentVersion\\Uninstall"),
        (HKEY_CURRENT_USER, "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall"),
    ] {
        let Ok(apps) = RegKey::predef(root).open_subkey(key) else { continue };
        for name in apps.enum_keys().flatten() {
            let Ok(app) = apps.open_subkey(&name) else { continue };
            let display: String = app.get_value("DisplayName").unwrap_or_default();
            let location: String = app.get_value("InstallLocation").unwrap_or_default();
            if display.starts_with("Blender") && !location.is_empty() {
                out.push(Path::new(&expand_env(&location)).join("blender.exe"));
            }
        }
    }
    out
}

/// Program of a shell command line: `"C:\path\app.exe" "%1"` -> `C:\path\app.exe`.
#[cfg_attr(not(windows), allow(dead_code))]
fn command_program(command: &str) -> Option<PathBuf> {
    let command = command.trim();
    let program = match command.strip_prefix('"') {
        Some(rest) => rest.split('"').next()?,
        None => command.split_whitespace().next()?,
    };
    (!program.is_empty()).then(|| PathBuf::from(expand_env(program)))
}

/// Expands `%NAME%` environment variables (registry values may contain them).
#[cfg_attr(not(windows), allow(dead_code))]
fn expand_env(s: &str) -> String {
    let mut out = String::new();
    let mut rest = s;
    while let Some(start) = rest.find('%') {
        out.push_str(&rest[..start]);
        let after = &rest[start + 1..];
        match after.find('%') {
            Some(end) if end > 0 => {
                let name = &after[..end];
                match std::env::var(name) {
                    Ok(v) => out.push_str(&v),
                    Err(_) => out.push_str(&rest[start..start + end + 2]),
                }
                rest = &after[end + 1..];
            }
            _ => {
                out.push('%');
                rest = after;
            }
        }
    }
    out.push_str(rest);
    out
}

#[cfg(not(windows))]
fn candidates() -> Vec<PathBuf> {
    let mut out = path_candidates("blender");
    out.extend(["/usr/bin/blender", "/usr/local/bin/blender", "/snap/bin/blender"].map(PathBuf::from));
    if let Ok(entries) = std::fs::read_dir("/opt") {
        let mut dirs: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.file_name().is_some_and(|n| n.to_string_lossy().to_lowercase().starts_with("blender")))
            .collect();
        dirs.sort();
        out.extend(dirs.into_iter().rev().map(|d| d.join("blender")));
    }
    out
}

/// Version from the program's version resource (e.g. 5.2.3).
#[cfg(windows)]
fn file_version(path: &Path) -> Option<(u32, u32, u32)> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::{GetFileVersionInfoSizeW, GetFileVersionInfoW, VS_FIXEDFILEINFO, VerQueryValueW};
    let wide: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
    // SAFETY: plain Win32 calls on buffers of the sizes they report; the fixed
    // info is copied out with an unaligned read after checking its length.
    unsafe {
        let size = GetFileVersionInfoSizeW(wide.as_ptr(), std::ptr::null_mut());
        if size == 0 {
            return None;
        }
        let mut data = vec![0u8; size as usize];
        if GetFileVersionInfoW(wide.as_ptr(), 0, size, data.as_mut_ptr().cast()) == 0 {
            return None;
        }
        let root = [u16::from(b'\\'), 0];
        let mut info: *mut core::ffi::c_void = std::ptr::null_mut();
        let mut len = 0u32;
        if VerQueryValueW(data.as_ptr().cast(), root.as_ptr(), &mut info, &mut len) == 0
            || info.is_null()
            || (len as usize) < std::mem::size_of::<VS_FIXEDFILEINFO>()
        {
            return None;
        }
        let fixed = std::ptr::read_unaligned(info as *const VS_FIXEDFILEINFO);
        (fixed.dwSignature == 0xFEEF_04BD)
            .then_some((fixed.dwFileVersionMS >> 16, fixed.dwFileVersionMS & 0xFFFF, fixed.dwFileVersionLS >> 16))
    }
}

#[cfg(not(windows))]
fn file_version(_path: &Path) -> Option<(u32, u32, u32)> {
    None
}

// ---------------------------------------------------------------------------
// Conversion
// ---------------------------------------------------------------------------

pub fn load(ctx: &LoadCtx) -> Result<Scene> {
    let result = convert_and_load(ctx);
    ctx.progress.set_range(0.0, 1.0);
    result
}

fn convert_and_load(ctx: &LoadCtx) -> Result<Scene> {
    let blender = find_blender(ctx.options.blender.as_deref())?;
    let work = WorkDir::new()?;
    let out = work.0.join("model.glb");
    let run = run_blender(&blender, ctx.path, &work.0, &out, ctx)?;

    ctx.progress.set_range(0.6, 1.0);
    ctx.progress.set_stage("Reading the converted model");
    let inner = LoadCtx { path: &out, dir: &work.0, progress: ctx.progress, options: ctx.options, warnings: Mutex::new(Vec::new()) };
    let mut scene = gltf::load(&inner)?;
    for w in inner.warnings.into_inner().unwrap_or_default() {
        ctx.warn(w);
    }

    // The clip of the active actions (what Blender's Play button shows) first.
    if let Some(name) = &run.timeline
        && let Some(i) = scene.animations.iter().position(|a| &a.name == name)
    {
        let clip = scene.animations.remove(i);
        scene.animations.insert(0, clip);
    }
    let by = run.version.as_ref().map_or_else(|| blender.label(), |v| format!("Blender {v}"));
    if let Some(newer) = &run.newer {
        ctx.warn(format!("The file was saved with Blender {newer}, newer than the {by} used to open it: some content may be missing"));
    }
    for note in run.notes {
        ctx.warn(format!("Blender: {note}"));
    }
    scene.converted_with = Some(by);
    Ok(scene)
}

/// What the export script reported.
#[derive(Default, Debug)]
struct Run {
    version: Option<String>,
    timeline: Option<String>,
    newer: Option<String>,
    notes: Vec<String>,
    error: Option<String>,
    done: bool,
    /// Last lines of Blender's own output, for error messages.
    tail: Vec<String>,
}

impl Run {
    fn line(&mut self, line: &str, ctx: &LoadCtx) {
        let Some(tagged) = line.strip_prefix("VIEWER3D:") else {
            if line.contains("Read blend:") {
                ctx.progress.set_stage("Blender: reading the file");
                ctx.progress.set(0.25);
            }
            log::debug!("Blender: {line}");
            if !line.trim().is_empty() {
                if self.tail.len() == 40 {
                    self.tail.remove(0);
                }
                self.tail.push(line.trim().to_owned());
            }
            return;
        };
        let (tag, text) = tagged.split_once(':').unwrap_or((tagged, ""));
        let text = text.trim().to_owned();
        match tag {
            "VERSION" => self.version = Some(text),
            "STAGE" if text == "Exporting" => {
                ctx.progress.set_stage("Blender: exporting to glTF");
                ctx.progress.set(0.4);
            }
            "STAGE" => {
                ctx.progress.set_stage("Blender: preparing animations");
                ctx.progress.set(0.35);
            }
            "TIMELINE" => self.timeline = Some(text),
            "NEWER" => self.newer = Some(text),
            "NOTE" => self.notes.push(text),
            "ERROR" => self.error = Some(text),
            "DONE" => {
                self.done = true;
                ctx.progress.set(0.6);
            }
            _ => {}
        }
    }

    /// Best explanation of a failed run.
    fn failure(&self, status: Option<i32>) -> String {
        let blender_error = self
            .tail
            .iter()
            .rev()
            .find(|l| l.contains("Error") || l.contains("ERROR") || l.contains("rror:"))
            .map(|l| log_message(l).to_owned());
        match (&self.error, blender_error) {
            (Some(e), Some(b)) if !b.contains(e.as_str()) => format!("{e} ({b})"),
            (Some(e), _) => e.clone(),
            (None, Some(b)) => format!("Blender could not convert the file: {b}"),
            (None, None) => match status {
                Some(code) => format!("Blender stopped without converting the file (exit code {code})"),
                None => "Blender stopped without converting the file".to_owned(),
            },
        }
    }
}

/// Message of a Blender log line, without its time, category and level:
/// `00:02.687  reports | ERROR Failed to read…` -> `Failed to read…`.
fn log_message(line: &str) -> &str {
    let message = line.split_once("| ").map_or(line, |(_, m)| m).trim();
    ["ERROR:", "ERROR", "Error:"].iter().find_map(|p| message.strip_prefix(p)).unwrap_or(message).trim()
}

fn run_blender(blender: &Blender, file: &Path, work: &Path, out: &Path, ctx: &LoadCtx) -> Result<Run> {
    let script = work.join("blend_export.py");
    std::fs::write(&script, SCRIPT).context("cannot write the export script")?;
    // An absolute path never starts with '-', so Blender cannot take it for an option.
    let file = std::path::absolute(file).unwrap_or_else(|_| file.to_path_buf());

    ctx.progress.set_stage(&format!("Starting {}", blender.label()));
    ctx.progress.set(0.02);
    let mut cmd = Command::new(&blender.path);
    cmd.args(["--background", "--factory-startup", "--disable-autoexec", "-noaudio"])
        .arg(&file)
        .args(["--python-exit-code", "3", "--python"])
        .arg(&script)
        .arg("--")
        .arg(out)
        .current_dir(work)
        .env_remove("PYTHONPATH")
        .env_remove("PYTHONHOME")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(windows_sys::Win32::System::Threading::CREATE_NO_WINDOW);
    }
    let mut child = cmd.spawn().with_context(|| format!("cannot start Blender ({})", blender.path.display()))?;
    end_with_viewer(&child);

    let lines = read_lines(&mut child);
    let mut run = Run::default();
    let status = loop {
        match lines.recv_timeout(Duration::from_millis(50)) {
            Ok(line) => run.line(&line, ctx),
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                // Output closed: Blender is ending.
                if let Some(status) = child.try_wait()? {
                    break status;
                }
                std::thread::sleep(Duration::from_millis(20));
            }
        }
        if ctx.progress.is_cancelled() {
            let _ = child.kill();
            let _ = child.wait();
            ctx.progress.check()?;
        }
    };
    if status.success() && run.done && out.is_file() {
        Ok(run)
    } else {
        bail!(run.failure(status.code()))
    }
}

/// Lines of Blender's standard output and error, read on helper threads so
/// that neither pipe can fill up and block Blender.
fn read_lines(child: &mut Child) -> mpsc::Receiver<String> {
    let (tx, rx) = mpsc::channel();
    let streams: [Option<Box<dyn Read + Send>>; 2] = [
        child.stdout.take().map(|s| Box::new(s) as Box<dyn Read + Send>),
        child.stderr.take().map(|s| Box::new(s) as Box<dyn Read + Send>),
    ];
    for stream in streams.into_iter().flatten() {
        let tx = tx.clone();
        let spawned = std::thread::Builder::new().name("blender-output".into()).spawn(move || {
            let mut reader = BufReader::new(stream);
            let mut buf = Vec::new();
            loop {
                buf.clear();
                match reader.read_until(b'\n', &mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {
                        if tx.send(String::from_utf8_lossy(&buf).trim_end().to_owned()).is_err() {
                            break;
                        }
                    }
                }
            }
        });
        if let Err(e) = spawned {
            log::warn!("cannot read Blender's output: {e}");
        }
    }
    rx
}

/// Puts Blender in a job object that Windows closes when the viewer exits, so
/// a conversion still running then is stopped instead of running on unseen.
#[cfg(windows)]
fn end_with_viewer(child: &Child) {
    use std::os::windows::io::AsRawHandle;
    use std::sync::OnceLock;
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JobObjectExtendedLimitInformation, SetInformationJobObject,
    };
    // The handle is kept for the life of the process (as an integer: raw
    // handles are not Sync); Windows closes it on exit, which ends the job.
    static JOB: OnceLock<usize> = OnceLock::new();
    let job = *JOB.get_or_init(|| {
        // SAFETY: plain Win32 calls; `info` is a zero-initialised C struct.
        unsafe {
            let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            if job.is_null() {
                return 0;
            }
            let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            let ok = SetInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                (&raw const info).cast(),
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            );
            if ok == 0 {
                CloseHandle(job);
                return 0;
            }
            job as usize
        }
    });
    // SAFETY: both handles are valid: the job for the process lifetime, the
    // child's for the lifetime of `child`.
    if job == 0 || unsafe { AssignProcessToJobObject(job as HANDLE, child.as_raw_handle() as HANDLE) } == 0 {
        log::warn!("Blender will not be stopped automatically if the viewer exits during a conversion");
    }
}

/// Elsewhere a conversion still running when the viewer exits ends by itself;
/// its temporary folder is removed by a later run (see [`WorkDir::new`]).
#[cfg(not(windows))]
fn end_with_viewer(_child: &Child) {}

/// Temporary folder of one conversion, removed when dropped.
struct WorkDir(PathBuf);

const WORK_PREFIX: &str = "viewer3d-blend-";

impl WorkDir {
    fn new() -> Result<Self> {
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let base = std::env::temp_dir();
        remove_stale(&base);
        for _ in 0..100 {
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let dir = base.join(format!("{WORK_PREFIX}{}-{n}", std::process::id()));
            match std::fs::create_dir(&dir) {
                Ok(()) => return Ok(Self(dir)),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e).with_context(|| format!("cannot create a temporary folder in '{}'", base.display())),
            }
        }
        Err(anyhow!("cannot create a temporary folder in '{}'", base.display()))
    }
}

impl Drop for WorkDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Removes conversion folders left by a viewer that was killed (older than a
/// day, so that conversions of other running viewers are never touched).
fn remove_stale(base: &Path) {
    let Ok(entries) = std::fs::read_dir(base) else { return };
    let day = Duration::from_secs(24 * 3600);
    for e in entries.flatten() {
        if !e.file_name().to_string_lossy().starts_with(WORK_PREFIX) {
            continue;
        }
        let old = e
            .metadata()
            .and_then(|m| m.modified())
            .is_ok_and(|t| SystemTime::now().duration_since(t).is_ok_and(|age| age > day));
        if old {
            let _ = std::fs::remove_dir_all(e.path());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shell_command_program() {
        assert_eq!(
            command_program(r#""C:\Program Files\Blender Foundation\Blender 5.2\blender-launcher.exe" "%1""#),
            Some(PathBuf::from(r"C:\Program Files\Blender Foundation\Blender 5.2\blender-launcher.exe"))
        );
        assert_eq!(command_program(r"C:\Blender\blender.exe %1"), Some(PathBuf::from(r"C:\Blender\blender.exe")));
        assert_eq!(command_program("  "), None);
    }

    #[test]
    fn launcher_is_replaced_by_program() {
        assert_eq!(program(Path::new(r"C:\B\blender-launcher.exe")), PathBuf::from(r"C:\B\blender.exe"));
        assert_eq!(program(Path::new(r"C:\B\blender.exe")), PathBuf::from(r"C:\B\blender.exe"));
    }

    #[test]
    fn environment_expansion() {
        // SAFETY: test-only variable, not read by other tests.
        unsafe { std::env::set_var("VIEWER3D_TEST_DIR", r"C:\Apps") };
        assert_eq!(expand_env(r"%VIEWER3D_TEST_DIR%\Blender"), r"C:\Apps\Blender");
        assert_eq!(expand_env("100% sure"), "100% sure");
        assert_eq!(expand_env("%VIEWER3D_UNSET_VARIABLE%"), "%VIEWER3D_UNSET_VARIABLE%");
    }

    #[test]
    fn failure_message_prefers_script_error() {
        let mut run = Run::default();
        assert!(run.failure(Some(1)).contains("exit code 1"));
        run.tail.push(r#"Error: Cannot read file "x.blend": File format is not supported"#.into());
        assert_eq!(
            run.failure(Some(1)),
            r#"Blender could not convert the file: Cannot read file "x.blend": File format is not supported"#
        );
        run.error = Some("Blender could not open the file".into());
        assert_eq!(
            run.failure(Some(3)),
            r#"Blender could not open the file (Cannot read file "x.blend": File format is not supported)"#
        );
    }

    #[test]
    fn log_prefix_is_removed() {
        assert_eq!(
            log_message("00:02.687  reports          | ERROR Failed to read blend file 'x.blend': Missing DNA block"),
            "Failed to read blend file 'x.blend': Missing DNA block"
        );
        assert_eq!(log_message("Error: Not freed memory blocks"), "Not freed memory blocks");
        assert_eq!(log_message("plain text"), "plain text");
    }
}
