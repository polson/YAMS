use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{ChildStdin, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;

pub mod engine;

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
            return Err("Choose an existing Python executable".into());
        }
        let has_main = Path::new(&self.project).join("yams/__main__.py").is_file()
            || Path::new(&self.project).join("yass/__main__.py").is_file();
        if !has_main {
            return Err("Choose the YAMS project folder containing yams/__main__.py".into());
        }
        if !["auto", "cpu", "cuda:0", "mps"].contains(&self.device.as_str()) {
            return Err("Unsupported device".into());
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
            .map_err(|error| error.to_string())?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| error.to_string())?;
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
                    let kind = entry.file_type().map_err(|error| error.to_string())?;
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
        .map_err(|error| format!("Could not run Python: {error}"))?;
    let mut input = child.stdin.take().unwrap();
    let sent = serde_json::to_writer(&mut input, &folders);
    drop(input);
    let output = child
        .wait_with_output()
        .map_err(|error| error.to_string())?;
    sent.map_err(|error| error.to_string())?;
    let event: Value = serde_json::from_slice(&output.stdout).map_err(|_| {
        format!(
            "Worker did not return previews: {}",
            String::from_utf8_lossy(&output.stderr)
        )
    })?;
    if !output.status.success() || event["event"] != "previews" {
        return Err(event["message"]
            .as_str()
            .unwrap_or("Unexpected worker preview response")
            .into());
    }
    let mut previews: Vec<Preview> =
        serde_json::from_value(event["songs"].clone()).map_err(|error| error.to_string())?;
    let expected: HashSet<_> = folders.into_iter().collect();
    let received: HashSet<_> = previews
        .iter()
        .map(|preview| preview.folder.clone())
        .collect();
    if received != expected
        || previews.len() != received.len()
        || previews.iter().any(|preview| {
            !["ready", "skipped", "error", "recovery_required"].contains(&preview.status.as_str())
        })
    {
        return Err("Worker returned mismatched folder previews".into());
    }
    previews.sort_by(|left, right| left.folder.cmp(&right.folder));
    Ok(previews)
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
    pub fn running(&self) -> bool {
        self.state.lock().unwrap().running
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
                .map_err(|error| error.to_string())?;
            if (!path.is_dir() && !is_con_file(&path) && !is_audio_file(&path)) || !folders.insert(path) || !ids.insert(&job.id) {
                return Err("Each queued job must have a unique ID and folder".into());
            }
        }
        let mut state = self.state.lock().unwrap();
        if state.running {
            return Err("A batch is already running".into());
        }
        state.running = true;
        state.stop = false;
        Ok(())
    }

    pub fn cancel(&self) -> Result<(), String> {
        let mut state = self.state.lock().unwrap();
        state.stop = true;
        if let Some(input) = state.input.as_mut() {
            input
                .write_all(b"cancel\n")
                .map_err(|error| error.to_string())?;
            input.flush().map_err(|error| error.to_string())?;
        }
        Ok(())
    }

    pub fn run(&self, settings: Settings, jobs: Vec<Job>, emit: Arc<dyn Fn(Value) + Send + Sync>) {
        for job in jobs {
            if self.state.lock().unwrap().stop {
                emit(
                    json!({"id": job.id, "kind": "finished", "status": "cancelled", "message": "Queue stopped before this song started"}),
                );
                continue;
            }
            emit(json!({"id": job.id, "kind": "started"}));
            match self.run_job(&settings, &job, emit.clone()) {
                Ok((status, message)) => emit(
                    json!({"id": job.id, "kind": "finished", "status": status, "message": message}),
                ),
                Err(message) => emit(
                    json!({"id": job.id, "kind": "finished", "status": "error", "message": message}),
                ),
            }
        }
        let mut state = self.state.lock().unwrap();
        state.running = false;
        state.input = None;
        drop(state);
        emit(json!({"kind": "batch_finished"}));
    }

    fn run_job(
        &self,
        settings: &Settings,
        job: &Job,
        emit: Arc<dyn Fn(Value) + Send + Sync>,
    ) -> Result<(String, String), String> {
        let expected_folder = Path::new(&job.folder)
            .canonicalize()
            .map_err(|error| error.to_string())?;
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
            .map_err(|error| format!("Could not start worker: {error}"))?;
        let mut state = self.state.lock().unwrap();
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
            let parsed = line
                .map_err(|error| error.to_string())
                .and_then(|line| {
                    serde_json::from_str::<Value>(&line).map_err(|error| error.to_string())
                })
                .and_then(|event| {
                    serde_json::from_value::<WorkerEvent>(event.clone())
                        .map(|worker| (event, worker))
                        .map_err(|error| error.to_string())
                });
            match parsed {
                Ok((event, worker)) if worker.matches_job(&expected_folder) => {
                    let name = worker.event.as_str();
                    if name == "models_error" || name == "runtime_error" {
                        self.state.lock().unwrap().stop = true;
                    }
                    if ["completed", "skipped", "recovered", "cancelled", "error"].contains(&name) {
                        terminal = name.into();
                    }
                    if let Some(detail) = worker.message {
                        message = detail;
                    }
                    needs_recovery |= name == "recovery_required";
                    emit(json!({"id": job.id, "kind": "worker", "payload": event}));
                }
                _ => {
                    protocol_error = Some(
                        "Worker returned invalid progress; cancellation requested".to_string(),
                    );
                    let _ = self.cancel();
                }
            }
        }
        let exit = child.wait().map_err(|error| error.to_string());
        self.state.lock().unwrap().input = None;
        let recent = diagnostics.join().map_err(|_| "Diagnostic reader failed")?;
        let exit = exit?;
        if needs_recovery {
            return Ok(("recovery_required".into(), message));
        }
        if let Some(error) = protocol_error {
            return Ok(("error".into(), format!("{error}\n{recent}")));
        }
        let status = match (exit.code(), terminal.as_str()) {
            (Some(0), "completed") => "completed",
            (Some(0), "skipped") => "skipped",
            (Some(0), "recovered") => "recovered",
            (Some(130), "cancelled") => "cancelled",
            _ => "error",
        };
        if status == "error" && message.is_empty() {
            message = format!("Worker exited with {exit}: {recent}");
        }
        Ok((status.into(), message))
    }
}
