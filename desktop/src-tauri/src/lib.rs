use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::io::{BufRead, BufReader, Write};
use std::panic::{self, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::process::{ChildStdin, Command, Stdio};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread;

pub mod engine;
pub mod log_file;

use log_file::record;

/// Returns the last `limit` characters of `text`, trimmed of surrounding whitespace.
pub fn tail(text: &str, limit: usize) -> String {
    let trimmed = text.trim();
    let count = trimmed.chars().count();
    if count <= limit {
        trimmed.to_string()
    } else {
        trimmed.chars().skip(count - limit).collect()
    }
}

/// Adds captured error output to a failure message, or says that nothing was printed.
fn with_detail(message: String, stderr: &str) -> String {
    let stderr = stderr.trim();
    if stderr.is_empty() {
        format!("{message}. No error output was printed.")
    } else {
        format!("{message}\n{stderr}")
    }
}

fn with_log_pointer(message: String) -> String {
    match log_file::path() {
        Some(path) => format!("{message}\nDetails were saved to {}", path.display()),
        None => message,
    }
}

#[derive(Clone)]
pub struct Settings {
    pub python: String,
    pub project: String,
    pub cache: String,
    pub models: String,
    pub inference_source: String,
    pub device: String,
    pub model_profile: String,
    pub installer: Option<String>,
}

impl Settings {
    pub fn command(&self) -> Result<Command, String> {
        if !Path::new(&self.python).is_file() {
            return Err(format!("Choose an existing Python executable. Not found: {}", self.python));
        }
        let has_main = Path::new(&self.project).join("yams/__main__.py").is_file()
            || Path::new(&self.project).join("yass/__main__.py").is_file();
        if !has_main {
            return Err(format!(
                "Choose the YAMS project folder containing yams/__main__.py. Not found in: {}",
                self.project
            ));
        }
        if !["auto", "cpu", "cuda:0", "mps", "dml:0"].contains(&self.device.as_str()) {
            return Err(format!("Unsupported device: {}", self.device));
        }
        let mut python_path = self.project.clone();
        #[cfg(debug_assertions)]
        {
            let dev_root = Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .and_then(|p| p.parent());
            if let Some(dev_root) = dev_root {
                if dev_root.join("yams/__main__.py").is_file() || dev_root.join("yass/__main__.py").is_file() {
                    let sep = if cfg!(windows) { ";" } else { ":" };
                    python_path = format!("{}{sep}{}", dev_root.display(), self.project);
                }
            }
        }
        let module_name = if Path::new(&self.project).join("yams/__main__.py").is_file() {
            "yams"
        } else {
            "yass"
        };
        let mut command = Command::new(&self.python);
        command
            .current_dir(&self.cache)
            .env_remove("PYTHONHOME")
            .env("PYTHONPATH", python_path)
            .args(["-u", "-m", module_name]);
        command.env("PYTHONDONTWRITEBYTECODE", "1");
        let cache = Path::new(&self.cache);
        command
            .env("XDG_CACHE_HOME", &cache)
            .env("NUMBA_CACHE_DIR", cache.join("numba"))
            .env("TORCH_HOME", cache.join("torch"));
        let mut bin_paths = Vec::new();
        let cache_bin = cache.join("bin");
        if cache_bin.is_dir() {
            bin_paths.push(cache_bin);
        }
        let project_bin = Path::new(&self.project).join("bin");
        if project_bin.is_dir() {
            bin_paths.push(project_bin);
        }
        if let Some(sys_path) = std::env::var_os("PATH") {
            bin_paths.extend(std::env::split_paths(&sys_path));
        }
        if let Ok(new_path) = std::env::join_paths(bin_paths) {
            command.env("PATH", new_path);
        }
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x08000000);
        }
        Ok(command)
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Job {
    pub id: String,
    pub folder: String,
    #[serde(default)]
    pub output_folder: Option<String>,
    pub recover: bool,
}

#[derive(Serialize, Deserialize)]
pub struct Preview {
    pub folder: String,
    pub status: String,
    pub message: String,
    pub source: String,
    pub backup: String,
}

#[derive(Deserialize)]
struct WorkerEvent {
    event: String,
    folder: String,
    target: Option<String>,
    asset: Option<String>,
    received: Option<u64>,
    total: Option<u64>,
    pass_number: Option<u64>,
    fraction: Option<f64>,
    message: Option<String>,
}

impl WorkerEvent {
    fn matches_job(&self, folder: &Path) -> bool {
        if Path::new(&self.folder) != folder {
            return false;
        }
        match self.event.as_str() {
            "model_download" => matches!(
                (&self.asset, self.received, self.total),
                (Some(asset), Some(received), Some(total))
                    if !asset.is_empty() && total > 0 && received <= total
            ),
            "model_started" | "model_finished" => {
                self.target
                    .as_ref()
                    .is_some_and(|target| !target.is_empty())
                    && self.pass_number.is_some_and(|pass| pass > 0)
            }
            "model_progress" => {
                self.target
                    .as_ref()
                    .is_some_and(|target| !target.is_empty())
                    && self.pass_number.is_some_and(|pass| pass > 0)
                    && self
                        .fraction
                        .is_some_and(|fraction| (0.0..=1.0).contains(&fraction))
            }
            _ => true,
        }
    }
}

pub fn is_con_file(path: &Path) -> bool {
    if !path.is_file() {
        return false;
    }
    let mut file = match std::fs::File::open(path) {
        Ok(file) => file,
        Err(_) => return false,
    };
    use std::io::Read;
    let mut header = [0u8; 4];
    if file.read_exact(&mut header).is_err() {
        return false;
    }
    matches!(&header, b"CON " | b"LIVE" | b"PIRS")
}

const AUDIO_EXTENSIONS: &[&str] = &["opus", "ogg", "mp3", "wav", "flac", "aiff", "m4a", "aac"];

pub fn is_audio_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| AUDIO_EXTENSIONS.iter().any(|&e| e.eq_ignore_ascii_case(ext)))
        .unwrap_or(false)
}

fn is_stem_name(stem: &str) -> bool {
    matches!(
        stem.to_ascii_lowercase().as_str(),
        "song"
            | "guitar"
            | "bass"
            | "rhythm"
            | "keys"
            | "vocals"
            | "vocals_1"
            | "vocals_2"
            | "drums"
            | "drums_1"
            | "drums_2"
            | "drums_3"
            | "drums_4"
            | "crowd"
            | "song_clean"
            | "vocals_clean"
            | "vocals_1_clean"
            | "vocals_2_clean"
            | "crowd_clean"
            | "vocals_explicit"
            | "vocals_1_explicit"
            | "vocals_2_explicit"
            | "crowd_explicit"
            | "kick"
            | "snare"
            | "other"
    )
}

fn is_excon_dir(dir: &Path, entries: &[std::fs::DirEntry]) -> bool {
    let has_mogg = entries.iter().any(|entry| {
        entry
            .path()
            .extension()
            .map(|ext| ext.eq_ignore_ascii_case("mogg"))
            .unwrap_or(false)
    });
    if !has_mogg {
        return false;
    }
    dir.join("songs.dta").is_file()
        || dir.join("songs").join("songs.dta").is_file()
        || dir.parent().map(|p| p.join("songs.dta").is_file()).unwrap_or(false)
}

pub fn inspect(settings: &Settings, paths: Vec<String>) -> Result<Vec<Preview>, String> {
    settings.command()?;
    let mut pending: Vec<PathBuf> = paths.into_iter().map(PathBuf::from).collect();
    let mut seen = HashSet::new();
    let mut folders = Vec::new();
    while let Some(path) = pending.pop() {
        let canonical = path
            .canonicalize()
            .map_err(|error| format!("{}: {error}", path.display()))?;
        if canonical.is_file() {
            if is_con_file(&canonical) || is_audio_file(&canonical) {
                if seen.insert(canonical.clone()) {
                    folders.push(canonical.to_string_lossy().into_owned());
                }
                continue;
            } else {
                return Err(format!("Unsupported file: {}", path.display()));
            }
        }
        if !canonical.is_dir() {
            return Err(format!("Choose song folders, audio files, CON packages, or exCON folders: {}", path.display()));
        }
        if !seen.insert(canonical.clone()) {
            continue;
        }
        let entries = std::fs::read_dir(&canonical)
            .and_then(|directory| directory.collect::<Result<Vec<_>, _>>())
            .map_err(|error| format!("Could not read {}: {error}", canonical.display()))?;
        let is_song = entries.iter().any(|entry| {
            entry
                .file_name()
                .to_string_lossy()
                .eq_ignore_ascii_case("song.ini")
        });
        let is_excon = !is_song && is_excon_dir(&canonical, &entries);
        if is_song || is_excon {
            folders.push(canonical.to_string_lossy().into_owned());
        } else {
            let direct_audio: Vec<_> = entries
                .iter()
                .filter(|entry| {
                    entry.file_type().map(|t| t.is_file()).unwrap_or(false)
                        && is_audio_file(&entry.path())
                })
                .collect();
            let has_multiple_stems = direct_audio.len() >= 2
                && direct_audio.iter().any(|entry| {
                    let stem = entry.path().file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string();
                    is_stem_name(&stem) && stem.to_ascii_lowercase() != "song"
                });
            let has_subdirs = entries.iter().any(|entry| entry.file_type().map(|t| t.is_dir()).unwrap_or(false));
            if has_multiple_stems {
                folders.push(canonical.to_string_lossy().into_owned());
            } else if direct_audio.len() == 1 && !has_subdirs {
                folders.push(canonical.to_string_lossy().into_owned());
            } else {
                for audio_entry in direct_audio {
                    let audio_canonical = audio_entry.path().canonicalize().unwrap_or_else(|_| audio_entry.path());
                    if seen.insert(audio_canonical.clone()) {
                        folders.push(audio_canonical.to_string_lossy().into_owned());
                    }
                }
                for entry in entries {
                    let kind = entry
                        .file_type()
                        .map_err(|error| format!("Could not read {}: {error}", entry.path().display()))?;
                    let name = entry.file_name();
                    if kind.is_dir() && !name.to_string_lossy().starts_with('.') && name != "backup" {
                        pending.push(entry.path());
                    } else if kind.is_file() && is_con_file(&entry.path()) {
                        let con_canonical = entry.path().canonicalize().unwrap_or_else(|_| entry.path());
                        if seen.insert(con_canonical.clone()) {
                            folders.push(con_canonical.to_string_lossy().into_owned());
                        }
                    }
                }
            }
        }
    }
    if folders.is_empty() {
        return Ok(Vec::new());
    }
    let mut child = settings
        .command()?
        .arg("--inspect-list")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("Could not start Python at {}: {error}", settings.python))?;
    let mut input = child.stdin.take().unwrap();
    let sent = serde_json::to_writer(&mut input, &folders);
    drop(input);
    let output = child
        .wait_with_output()
        .map_err(|error| format!("Could not read the worker's preview output: {error}"))?;
    let stderr = tail(&String::from_utf8_lossy(&output.stderr), 2000);
    sent.map_err(|error| format!("Could not send songs to the worker: {error}"))?;
    let event: Value = match serde_json::from_slice(&output.stdout) {
        Ok(event) => event,
        Err(_) => {
            let message = format!("Worker did not return previews ({})", output.status);
            record(format!("{message}: {stderr}"));
            return Err(with_detail(message, &stderr));
        }
    };
    if !output.status.success() || event["event"] != "previews" {
        let message = event["message"]
            .as_str()
            .unwrap_or("Unexpected worker preview response")
            .to_string();
        record(format!("{message} ({}): {stderr}", output.status));
        return Err(with_detail(message, &stderr));
    }
    let mut previews: Vec<Preview> =
        serde_json::from_value(event["songs"].clone()).map_err(|error| format!("Worker previews were malformed: {error}"))?;
    let expected: HashSet<_> = folders.into_iter().collect();
    let received: HashSet<_> = previews
        .iter()
        .map(|preview| preview.folder.clone())
        .collect();
    if received != expected || previews.len() != received.len() {
        let missing = expected.difference(&received).count();
        let unexpected = received.difference(&expected).count();
        let duplicated = previews.len().saturating_sub(received.len());
        return Err(format!(
            "Worker returned previews for different songs: {missing} missing, {unexpected} unexpected, {duplicated} duplicated"
        ));
    }
    if let Some(preview) = previews.iter().find(|preview| {
        !["ready", "skipped", "error", "recovery_required"].contains(&preview.status.as_str())
    }) {
        return Err(format!(
            "Worker returned unknown preview status {:?} for {}",
            preview.status, preview.folder
        ));
    }
    previews.sort_by(|left, right| left.folder.cmp(&right.folder));
    Ok(previews)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tail_keeps_the_end_of_long_text() {
        assert_eq!(tail("  abcdef \n", 3), "def");
        assert_eq!(tail("ab", 3), "ab");
    }

    #[test]
    fn failure_details_say_when_nothing_was_printed() {
        assert_eq!(
            with_detail("failed".into(), "  "),
            "failed. No error output was printed."
        );
        assert_eq!(with_detail("failed".into(), "boom\n"), "failed\nboom");
    }
}

#[derive(Default)]
struct State {
    running: bool,
    stop: bool,
    input: Option<ChildStdin>,
}

#[derive(Clone, Default)]
pub struct Runner {
    state: Arc<Mutex<State>>,
}

impl Runner {
    /// Locks the batch state. A panic in another thread must not stop later batches or cancellation.
    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|poison| poison.into_inner())
    }

    pub fn running(&self) -> bool {
        self.state().running
    }

    pub fn reserve(&self, settings: &Settings, jobs: &[Job]) -> Result<(), String> {
        settings.command()?;
        if jobs.is_empty() {
            return Err("Choose at least one ready song".into());
        }
        let mut folders = HashSet::new();
        let mut ids = HashSet::new();
        for job in jobs {
            let path = Path::new(&job.folder)
                .canonicalize()
                .map_err(|error| format!("Could not open {}: {error}", job.folder))?;
            if !path.is_dir() && !is_con_file(&path) && !is_audio_file(&path) {
                return Err(format!("Not a song folder, audio file, or CON package: {}", job.folder));
            }
            if !folders.insert(path) {
                return Err(format!("The same song is queued twice: {}", job.folder));
            }
            if !ids.insert(&job.id) {
                return Err(format!("Two queued songs share the ID {}", job.id));
            }
        }
        let mut state = self.state();
        if state.running {
            return Err("A batch is already running".into());
        }
        state.running = true;
        state.stop = false;
        Ok(())
    }

    pub fn cancel(&self) -> Result<(), String> {
        let mut state = self.state();
        state.stop = true;
        if let Some(input) = state.input.as_mut() {
            let sent = input.write_all(b"cancel\n").and_then(|()| input.flush());
            if let Err(error) = sent {
                let message = format!("Could not send cancel to the worker: {error}");
                record(&message);
                return Err(message);
            }
        }
        Ok(())
    }

    pub fn run(&self, settings: Settings, jobs: Vec<Job>, emit: Arc<dyn Fn(Value) + Send + Sync>) {
        record(format!("Batch started: {} song(s), device {}, model profile {}", jobs.len(), settings.device, settings.model_profile));
        let outcome = panic::catch_unwind(AssertUnwindSafe(|| self.run_jobs(&settings, jobs, &emit)));
        {
            let mut state = self.state();
            state.running = false;
            state.input = None;
        }
        if outcome.is_err() {
            record("Batch stopped unexpectedly; remaining songs were not started");
        }
        record("Batch finished");
        emit(json!({"kind": "batch_finished"}));
    }

    fn run_jobs(&self, settings: &Settings, jobs: Vec<Job>, emit: &Arc<dyn Fn(Value) + Send + Sync>) {
        for job in jobs {
            if self.state().stop {
                record(format!("Job {} cancelled before it started", job.id));
                emit(
                    json!({"id": job.id, "kind": "finished", "status": "cancelled", "message": "Queue stopped before this song started"}),
                );
                continue;
            }
            record(format!("Job {} started: {}", job.id, job.folder));
            emit(json!({"id": job.id, "kind": "started"}));
            // A panic while separating one song must not stop the rest of the batch.
            let outcome = panic::catch_unwind(AssertUnwindSafe(|| self.run_job(settings, &job, emit.clone())));
            let (status, message) = match outcome {
                Ok(Ok(result)) => result,
                Ok(Err(message)) => ("error".into(), with_log_pointer(message)),
                Err(_) => (
                    "error".into(),
                    with_log_pointer("YAMS stopped unexpectedly while separating this song.".into()),
                ),
            };
            record(format!("Job {} finished with status {status}: {message}", job.id));
            emit(json!({"id": job.id, "kind": "finished", "status": status, "message": message}));
        }
    }

    fn run_job(
        &self,
        settings: &Settings,
        job: &Job,
        emit: Arc<dyn Fn(Value) + Send + Sync>,
    ) -> Result<(String, String), String> {
        let expected_folder = Path::new(&job.folder)
            .canonicalize()
            .map_err(|error| format!("Could not open {}: {error}", job.folder))?;
        let mut command = settings.command()?;
        command.arg(&expected_folder);
        if job.recover {
            command.arg("--recover");
        } else {
            command.args([
                "--models",
                &settings.models,
                "--model-profile",
                &settings.model_profile,
                "--inference-source",
                &settings.inference_source,
                "--device",
                &settings.device,
                "--control-stdin",
                "--download-models",
            ]);
            if let Some(output) = &job.output_folder {
                command.args(["--output-folder", output]);
            }
            if let Some(installer) = &settings.installer {
                command.args([
                    "--runtime-installer",
                    installer,
                    "--runtime-resources",
                    &settings.project,
                ]);
            }
        }
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|error| format!("Could not start the worker with {}: {error}", settings.python))?;
        let mut state = self.state();
        state.input = child.stdin.take();
        let stop = state.stop;
        drop(state);
        if stop {
            let _ = self.cancel();
        }
        let stderr = child.stderr.take().unwrap();
        let diagnostic_emit = emit.clone();
        let id = job.id.clone();
        let diagnostics = thread::spawn(move || {
            let mut recent = String::new();
            for line in BufReader::new(stderr).lines() {
                match line {
                    Ok(line) => {
                        let line: String = line.chars().take(2048).collect();
                        record(format!("worker {id}: {line}"));
                        recent.push_str(&line);
                        recent.push('\n');
                        if recent.len() > 8192 {
                            recent = recent
                                .chars()
                                .rev()
                                .take(4096)
                                .collect::<String>()
                                .chars()
                                .rev()
                                .collect();
                        }
                        diagnostic_emit(json!({"id": id, "kind": "diagnostic", "message": line}));
                    }
                    Err(error) => {
                        record(format!("worker {id}: could not read stderr: {error}"));
                        recent.push_str(&error.to_string());
                        break;
                    }
                }
            }
            recent
        });
        let mut terminal = String::new();
        let mut message = String::new();
        let mut protocol_error = None;
        let mut needs_recovery = false;
        for line in BufReader::new(child.stdout.take().unwrap()).lines() {
            let text = match line {
                Ok(text) => text,
                Err(error) => {
                    protocol_error.get_or_insert_with(|| format!("Could not read worker output: {error}"));
                    let _ = self.cancel();
                    continue;
                }
            };
            let parsed = serde_json::from_str::<Value>(&text).ok().and_then(|event| {
                serde_json::from_value::<WorkerEvent>(event.clone())
                    .ok()
                    .map(|worker| (event, worker))
            });
            match parsed {
                Some((event, worker)) if worker.matches_job(&expected_folder) => {
                    let name = worker.event.as_str();
                    if name == "models_error" || name == "runtime_error" {
                        self.state().stop = true;
                    }
                    if ["completed", "skipped", "recovered", "cancelled", "error"].contains(&name) {
                        terminal = name.into();
                    }
                    if let Some(detail) = worker.message.filter(|_| name != "stage") {
                        message = detail;
                    }
                    needs_recovery |= name == "recovery_required";
                    emit(json!({"id": job.id, "kind": "worker", "payload": event}));
                }
                _ => {
                    let excerpt: String = text.chars().take(300).collect();
                    protocol_error.get_or_insert_with(|| {
                        format!("Worker sent an unexpected line on stdout; cancellation requested: {excerpt}")
                    });
                    let _ = self.cancel();
                }
            }
        }
        let exit = child.wait().map_err(|error| format!("Could not wait for the worker: {error}"));
        self.state().input = None;
        let recent = diagnostics.join().unwrap_or_else(|_| {
            record(format!("Job {} lost its worker error output", job.id));
            String::new()
        });
        let exit = exit?;
        let (status, message) = if needs_recovery {
            ("recovery_required".to_string(), message)
        } else if let Some(error) = protocol_error {
            ("error".to_string(), with_detail(error, &recent))
        } else {
            let status = match (exit.code(), terminal.as_str()) {
                (Some(0), "completed") => "completed",
                (Some(0), "skipped") => "skipped",
                (Some(0), "recovered") => "recovered",
                (Some(130), "cancelled") => "cancelled",
                _ => "error",
            };
            let message = if status == "error" && message.is_empty() {
                with_detail(format!("Worker exited with {exit}"), &recent)
            } else {
                message
            };
            (status.to_string(), message)
        };
        if status == "error" {
            record(format!("Job {} worker output:\n{}", job.id, tail(&recent, 4000)));
            return Ok((status, with_log_pointer(message)));
        }
        Ok((status, message))
    }
}
