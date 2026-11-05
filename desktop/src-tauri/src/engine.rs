use crate::log_file::record;
use crate::{tail, with_detail, Settings};
use serde::{Deserialize, Serialize};
use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Preferences {
    pub device: String,
    #[serde(default)]
    pub available_devices: Vec<String>,
    #[serde(default = "default_model_profile")]
    pub model_profile: String,
    #[serde(default)]
    pub available_model_profiles: Vec<String>,
}

fn default_model_profile() -> String {
    "mega-8stem".into()
}

fn detect_available_model_profiles() -> Vec<String> {
    let mut profiles = vec!["mega-8stem".into()];
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        profiles.push("mlx-6stem".into());
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        profiles.push("torch-6stem".into());
        profiles.push("onnx-6stem".into());
    }
    profiles
}

impl Default for Preferences {
    fn default() -> Self {
        let available_devices = detect_available_devices();
        let device = available_devices
            .first()
            .cloned()
            .unwrap_or_else(|| "cpu".into());
        let available_model_profiles = detect_available_model_profiles();
        let model_profile = default_model_profile();
        Self {
            device,
            available_devices,
            model_profile,
            available_model_profiles,
        }
    }
}

#[cfg(target_os = "linux")]
fn has_linux_amd_gpu() -> bool {
    if std::path::Path::new("/dev/kfd").exists() {
        if let Ok(entries) = std::fs::read_dir("/sys/class/drm") {
            for entry in entries.flatten() {
                let vendor = entry.path().join("device/vendor");
                if let Ok(content) = std::fs::read_to_string(vendor) {
                    if content.trim().eq_ignore_ascii_case("0x1002") {
                        return true;
                    }
                }
            }
        }
    }
    false
}

#[cfg(target_os = "windows")]
fn has_windows_directml_gpu() -> bool {
    use std::os::windows::process::CommandExt;
    let mut cmd = Command::new("powershell");
    cmd.creation_flags(0x08000000);
    cmd.args([
        "-NoProfile",
        "-Command",
        "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name",
    ])
    .output()
    .map(|out| {
        let stdout = String::from_utf8_lossy(&out.stdout).to_lowercase();
        stdout.contains("amd")
            || stdout.contains("radeon")
            || stdout.contains("intel")
            || stdout.contains("geforce")
            || stdout.contains("rtx")
            || stdout.contains("gtx")
    })
    .unwrap_or_else(|error| {
        record(format!("DirectML GPU check skipped: could not run PowerShell: {error}"));
        false
    })
}

fn detect_available_devices() -> Vec<String> {
    let mut devices = Vec::new();
    #[cfg(target_os = "macos")]
    {
        devices.push("mps".into());
    }
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    {
        let mut cmd = Command::new("nvidia-smi");
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(0x08000000);
        }
        match cmd.arg("--id=0").output() {
            Ok(output) if output.status.success() => devices.push("cuda:0".into()),
            Ok(output) => record(format!("NVIDIA GPU not offered: nvidia-smi exited with {}", output.status)),
            Err(error) => record(format!("NVIDIA GPU not offered: could not run nvidia-smi: {error}")),
        }
    }
    #[cfg(target_os = "linux")]
    {
        if !devices.contains(&"cuda:0".to_string()) && has_linux_amd_gpu() {
            devices.push("cuda:0".into());
        }
    }
    #[cfg(target_os = "windows")]
    {
        if has_windows_directml_gpu() {
            devices.push("dml:0".into());
        }
    }
    devices.push("cpu".into());
    devices
}

pub struct Engine {
    python: String,
    resources: PathBuf,
    cache: PathBuf,
    models: PathBuf,
}

impl Engine {
    pub fn settings(&self, preferences: Preferences) -> Settings {
        let inference_source = if self.resources.join("inference").is_dir() {
            self.resources.join("inference")
        } else if self.resources.join("yams/inference").is_dir() {
            self.resources.join("yams/inference")
        } else {
            self.resources.join("yass/inference")
        };
        let models = match preferences.model_profile.as_str() {
            "mlx-6stem" => self.cache.join("models/mlx-6stem"),
            "torch-6stem" => self.cache.join("models/torch-6stem"),
            "onnx-6stem" => self.cache.join("models/onnx-6stem"),
            _ => self.models.clone(),
        };
        Settings {
            python: self.python.clone(),
            project: self.resources.to_string_lossy().into_owned(),
            cache: self.cache.to_string_lossy().into_owned(),
            models: models.to_string_lossy().into_owned(),
            inference_source: inference_source.to_string_lossy().into_owned(),
            device: preferences.device,
            model_profile: preferences.model_profile,
            installer: Path::new(&self.python)
                .starts_with(self.cache.join("runtime-v1"))
                .then(|| {
                    self.resources
                        .join(if cfg!(windows) {
                            "bin/uv.exe"
                        } else {
                            "bin/uv"
                        })
                        .to_string_lossy()
                        .into_owned()
                }),
        }
    }

    pub fn prepare(
        resources: PathBuf,
        cache: PathBuf,
        emit: impl Fn(&str),
    ) -> Result<Self, String> {
        let has_inference = resources.join("inference/utils/settings.py").is_file()
            || resources.join("yams/inference/utils/settings.py").is_file()
            || resources.join("yass/inference/utils/settings.py").is_file();
        let has_worker = resources.join("yams/__main__.py").is_file()
            || resources.join("yass/__main__.py").is_file();
        if !has_worker || !has_inference {
            return Err(format!(
                "Audio engine resources are missing in {}. Rebuild or reinstall YAMS.",
                resources.display()
            ));
        }
        std::fs::create_dir_all(&cache).map_err(|error| {
            format!("Could not create the audio engine folder {}: {error}", cache.display())
        })?;
        let environment = cache.join("runtime-v1");
        let python = environment.join(if cfg!(windows) {
            "Scripts/python.exe"
        } else {
            "bin/python"
        });
        emit("Checking audio engine…");
        let mut ready = checked_probe(&python, &resources, &cache, "--base");
        #[cfg(debug_assertions)]
        if ready.is_err() && std::env::var_os("YAMS_ENGINE_FRESH").or_else(|| std::env::var_os("YASS_ENGINE_FRESH")).is_none() {
            let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"));
            if let Some(home) = home {
                let development = PathBuf::from(home).join(if cfg!(windows) {
                    "miniconda3/envs/MusicSep-Lightning/python.exe"
                } else {
                    "miniconda3/envs/MusicSep-Lightning/bin/python"
                });
                ready = checked_probe(&development, &resources, &cache, "--inference");
            }
        }
        if ready.is_err() {
            emit("Preparing Python…");
            let uv = resources.join(if cfg!(windows) {
                "bin/uv.exe"
            } else {
                "bin/uv"
            });
            // The failure above is already in the log; the venv is recreated next.
            record(format!("Recreating the audio engine at {}", environment.display()));
            run_installer(
                &uv,
                &cache,
                &[
                    "venv",
                    "--clear",
                    "--managed-python",
                    "--python",
                    "3.13.10",
                    &environment.to_string_lossy(),
                ],
            )?;
            emit("Verifying Python…");
            ready = checked_probe(&python, &resources, &cache, "--base");
        }
        let python = ready.map_err(|error| format!("Audio engine setup failed: {error}"))?;
        emit("Checking FFmpeg…");
        ensure_ffmpeg(Path::new(&python), &resources, &cache, &emit)?;
        let models = cache.join("models/mega-v1");
        #[cfg(debug_assertions)]
        let models = std::env::var_os("YAMS_MODELS")
            .or_else(|| std::env::var_os("YASS_MODELS"))
            .map(PathBuf::from)
            .unwrap_or(models);
        Ok(Self {
            python,
            resources,
            cache,
            models,
        })
    }
}

fn ensure_ffmpeg(
    python: &Path,
    resources: &Path,
    cache: &Path,
    emit: &impl Fn(&str),
) -> Result<(), String> {
    let mut command = Command::new(python);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000);
    }
    let mut bin_paths = Vec::new();
    let cache_bin = cache.join("bin");
    if cache_bin.is_dir() {
        bin_paths.push(cache_bin);
    }
    let resources_bin = resources.join("bin");
    if resources_bin.is_dir() {
        bin_paths.push(resources_bin);
    }
    if let Some(sys_path) = std::env::var_os("PATH") {
        bin_paths.extend(std::env::split_paths(&sys_path));
    }
    if let Ok(new_path) = std::env::join_paths(bin_paths) {
        command.env("PATH", new_path);
    }
    command
        .env_remove("PYTHONHOME")
        .env("PYTHONPATH", resources)
        .arg("-m")
        .arg("yams.ffmpeg")
        .arg("--cache")
        .arg(cache)
        .arg("--resources")
        .arg(resources)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = command
        .spawn()
        .map_err(|error| {
            format!("Could not start Python at {} to check FFmpeg: {error}", python.display())
        })?;

    // Read stderr on its own thread so a child that writes a lot of it cannot block while stdout is read.
    let stderr_reader = child.stderr.take().map(|mut stderr| {
        std::thread::spawn(move || {
            let mut bytes = Vec::new();
            let _ = std::io::Read::read_to_end(&mut stderr, &mut bytes);
            String::from_utf8_lossy(&bytes).into_owned()
        })
    });

    if let Some(stdout) = child.stdout.take() {
        let reader = std::io::BufReader::new(stdout);
        for line in reader.lines() {
            let line = match line {
                Ok(line) => line,
                Err(error) => {
                    record(format!("FFmpeg check output could not be read: {error}"));
                    break;
                }
            };
            if let Some(msg) = line.strip_prefix("[ffmpeg_download] ") {
                if !msg.is_empty() {
                    emit(msg);
                }
            } else if let Some(msg) = line.strip_prefix("[ffmpeg_ready] ") {
                if !msg.is_empty() {
                    emit(msg);
                }
            }
        }
    }

    let status = child
        .wait()
        .map_err(|error| format!("FFmpeg check failed: {error}"))?;
    let stderr = stderr_reader
        .and_then(|reader| reader.join().ok())
        .unwrap_or_default();
    if !status.success() {
        let detail = tail(&stderr, 2000);
        let message = with_detail(format!("FFmpeg setup failed ({status})"), &detail);
        record(format!("FFmpeg check failed for {}: {message}", python.display()));
        return Err(message);
    }
    Ok(())
}

fn probe(python: &Path, resources: &Path, cache: &Path, mode: &str) -> Result<String, String> {
    let mut command = Command::new(python);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000);
    }
    let mut bin_paths = Vec::new();
    let cache_bin = cache.join("bin");
    if cache_bin.is_dir() {
        bin_paths.push(cache_bin);
    }
    let resources_bin = resources.join("bin");
    if resources_bin.is_dir() {
        bin_paths.push(resources_bin);
    }
    if let Some(sys_path) = std::env::var_os("PATH") {
        bin_paths.extend(std::env::split_paths(&sys_path));
    }
    if let Ok(new_path) = std::env::join_paths(bin_paths) {
        command.env("PATH", new_path);
    }
    let output = command
        .arg(resources.join("check_runtime.py"))
        .arg(mode)
        .env_remove("PYTHONHOME")
        .env_remove("PYTHONPATH")
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .env("PYTHONIOENCODING", "utf-8")
        .env("XDG_CACHE_HOME", cache)
        .env("NUMBA_CACHE_DIR", cache.join("numba"))
        .env("TORCH_HOME", cache.join("torch"))
        .output()
        .map_err(|error| format!("Could not start Python at {}: {error}", python.display()))?;
    if !output.status.success() {
        let stderr = tail(&String::from_utf8_lossy(&output.stderr), 2000);
        return Err(with_detail(
            format!("Python at {} failed its check ({})", python.display(), output.status),
            &stderr,
        ));
    }
    let executable = String::from_utf8(output.stdout).map_err(|error| {
        format!("Python at {} returned an invalid UTF-8 executable path: {error}", python.display())
    })?;
    Ok(executable.trim().to_string())
}

/// Runs `probe` and records a failure, so the reason survives when a later step replaces the result.
fn checked_probe(python: &Path, resources: &Path, cache: &Path, mode: &str) -> Result<String, String> {
    probe(python, resources, cache, mode).map_err(|error| {
        record(format!("Python check {mode} failed for {}: {error}", python.display()));
        error
    })
}

fn run_installer(uv: &Path, cache: &Path, arguments: &[&str]) -> Result<(), String> {
    let mut command = Command::new(uv);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000);
    }
    let output = command
        .arg("--no-config")
        .args(arguments)
        .env_remove("PYTHONHOME")
        .env_remove("PYTHONPATH")
        .env("UV_CACHE_DIR", cache.join("uv"))
        .env("UV_PYTHON_INSTALL_DIR", cache.join("python"))
        .env("UV_PYTHON_INSTALL_BIN", "0")
        .output()
        .map_err(|error| format!("Could not start the audio engine installer at {}: {error}", uv.display()))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let detail = tail(if stderr.trim().is_empty() { &stdout } else { &stderr }, 2000);
        let message = with_detail(format!("Audio engine installation failed ({})", output.status), &detail);
        record(format!("uv {} failed: {message}", arguments.join(" ")));
        return Err(message);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn probe_preserves_unicode_and_rejects_invalid_utf8() {
        use std::os::unix::fs::PermissionsExt;

        let directory = std::env::temp_dir().join(format!("yams-probe-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let python = directory.join("python");
        std::fs::write(
            &python,
            "#!/bin/sh\n[ \"$PYTHONIOENCODING\" = utf-8 ] || exit 1\nprintf 'C:\\\\Users\\\\José\\\\python.exe\\n'\n",
        ).unwrap();
        std::fs::set_permissions(&python, std::fs::Permissions::from_mode(0o700)).unwrap();
        assert_eq!(
            probe(&python, &directory, &directory, "--base").unwrap(),
            "C:\\Users\\José\\python.exe"
        );

        std::fs::write(&python, "#!/bin/sh\nprintf '\\351'\n").unwrap();
        let error = probe(&python, &directory, &directory, "--base").unwrap_err();
        assert!(error.contains("invalid UTF-8 executable path"));
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn detects_expected_devices() {
        let devices = detect_available_devices();
        assert!(devices.contains(&"cpu".to_string()));
        #[cfg(target_os = "macos")]
        {
            assert_eq!(devices.first().map(String::as_str), Some("mps"));
        }
    }

    #[test]
    fn detects_expected_model_profiles() {
        let profiles = detect_available_model_profiles();
        assert!(profiles.contains(&"mega-8stem".to_string()));
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            assert!(profiles.contains(&"mlx-6stem".to_string()));
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            assert!(profiles.contains(&"torch-6stem".to_string()));
            assert!(profiles.contains(&"onnx-6stem".to_string()));
        }
    }
}
