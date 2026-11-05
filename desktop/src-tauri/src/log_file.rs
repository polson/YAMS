use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

static LOG_PATH: OnceLock<PathBuf> = OnceLock::new();
static WRITE_LOCK: Mutex<()> = Mutex::new(());

/// Sets the log file inside `directory`. Only the first call has an effect.
pub fn start(directory: &Path) -> Result<PathBuf, String> {
    std::fs::create_dir_all(directory)
        .map_err(|error| format!("Could not create log folder {}: {error}", directory.display()))?;
    let path = directory.join("yams-desktop.log");
    Ok(LOG_PATH.get_or_init(|| path).clone())
}

pub fn path() -> Option<&'static Path> {
    LOG_PATH.get().map(PathBuf::as_path)
}

/// Appends one timestamped line to the log file. Falls back to stderr when no file is set or writing fails.
pub fn record(message: impl AsRef<str>) {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    let line = format!("[{seconds}] {}\n", message.as_ref());
    let written = match LOG_PATH.get() {
        Some(path) => {
            let _guard = WRITE_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .and_then(|mut file| file.write_all(line.as_bytes()))
                .is_ok()
        }
        None => false,
    };
    if !written {
        eprint!("{line}");
    }
}
