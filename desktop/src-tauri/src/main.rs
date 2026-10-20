#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde_json::Value;
use std::path::Path;
use std::sync::{Arc, Mutex};
use tauri::{ipc::Channel, Emitter, Manager, State};
use yams_desktop::engine::{Engine, Preferences};
use yams_desktop::{inspect, Job, Preview, Runner, Settings};

type EngineState = Arc<Mutex<Option<Engine>>>;

#[tauri::command]
fn default_settings() -> Preferences {
    Preferences::default()
}

#[tauri::command]
async fn prepare_engine(
    app: tauri::AppHandle,
    events: Channel<String>,
    engine: State<'_, EngineState>,
) -> Result<(), String> {
    let resources = app
        .path()
        .resource_dir()
        .map_err(|error| error.to_string())?
        .join("engine");
    let cache = app
        .path()
        .app_cache_dir()
        .map_err(|error| error.to_string())?;
    #[cfg(debug_assertions)]
    let cache = std::env::var_os("YAMS_ENGINE_CACHE")
        .or_else(|| std::env::var_os("YASS_ENGINE_CACHE"))
        .map(std::path::PathBuf::from)
        .unwrap_or(cache);
    let engine = engine.inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        let mut engine = engine.lock().unwrap();
        if engine.is_none() {
            *engine = Some(Engine::prepare(resources, cache, |message| {
                let _ = events.send(message.to_string());
            })?);
        }
        Ok(())
    })
    .await
    .map_err(|error| error.to_string())?
}

fn worker_settings(engine: &EngineState, preferences: Preferences) -> Result<Settings, String> {
    let engine = engine.lock().unwrap();
    engine
        .as_ref()
        .map(|engine| engine.settings(preferences))
        .ok_or_else(|| "Audio engine is not ready. Retry setup.".into())
}

#[tauri::command]
async fn inspect_folders(
    settings: Preferences,
    paths: Vec<String>,
    engine: State<'_, EngineState>,
) -> Result<Vec<Preview>, String> {
    let settings = worker_settings(engine.inner(), settings)?;
    tauri::async_runtime::spawn_blocking(move || inspect(&settings, paths))
        .await
        .map_err(|error| error.to_string())?
}

#[tauri::command]
fn start_batch(
    settings: Preferences,
    jobs: Vec<Job>,
    events: Channel<Value>,
    runner: State<'_, Runner>,
    engine: State<'_, EngineState>,
) -> Result<(), String> {
    let settings = worker_settings(engine.inner(), settings)?;
    runner.reserve(&settings, &jobs)?;
    let running = runner.inner().clone();
    let disconnected = running.clone();
    std::thread::spawn(move || {
        running.run(
            settings,
            jobs,
            Arc::new(move |event| {
                if events.send(event).is_err() {
                    let _ = disconnected.cancel();
                }
            }),
        );
    });
    Ok(())
}

#[tauri::command]
fn cancel_batch(runner: State<'_, Runner>) -> Result<(), String> {
    runner.cancel()
}

#[tauri::command]
async fn open_folder(folder: String) -> Result<(), String> {
    tauri::async_runtime::spawn_blocking(move || {
        let raw_path = Path::new(&folder);
        let path = raw_path
            .canonicalize()
            .or_else(|_| {
                let parent = raw_path.parent().unwrap_or(raw_path);
                let name = raw_path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                let stem = raw_path.file_stem().and_then(|s| s.to_str()).unwrap_or(name);
                let has_ext = raw_path.extension().is_some();
                let candidate = if has_ext { stem } else { &format!("{name}_unpacked") };
                parent.join(candidate).canonicalize()
            })
            .map_err(|error| error.to_string())?;

        if !path.exists() {
            return Err("The song folder or file no longer exists".into());
        }

        let is_dir = path.is_dir();

        #[cfg(target_os = "windows")]
        let path = {
            let text = path.to_string_lossy();
            if let Some(network) = text.strip_prefix(r"\\?\UNC\") {
                std::path::PathBuf::from(format!(r"\\{network}"))
            } else {
                std::path::PathBuf::from(text.strip_prefix(r"\\?\").unwrap_or(&text))
            }
        };

        #[cfg(target_os = "windows")]
        let mut command = {
            let mut cmd = std::process::Command::new("explorer.exe");
            if is_dir {
                cmd.arg(path);
            } else {
                cmd.arg(format!("/select,{}", path.display()));
            }
            cmd
        };

        #[cfg(target_os = "macos")]
        let mut command = {
            let mut cmd = std::process::Command::new("open");
            if !is_dir {
                cmd.arg("-R");
            }
            cmd.arg(path);
            cmd
        };

        #[cfg(target_os = "linux")]
        let mut command = {
            let mut cmd = std::process::Command::new("xdg-open");
            if is_dir {
                cmd.arg(path);
            } else {
                cmd.arg(path.parent().unwrap_or(&path));
            }
            cmd
        };

        let status = command
            .status()
            .map_err(|error| format!("Could not open item: {error}"))?;
        if !status.success() {
            return Err(format!("Could not open item: {status}"));
        }
        Ok(())
    })
    .await
    .map_err(|error| error.to_string())?
}

fn main() {
    std::env::remove_var("PYTHONHOME");
    std::env::remove_var("PYTHONPATH");
    #[cfg(target_os = "linux")]
    if std::env::var_os("WEBKIT_DISABLE_DMABUF_RENDERER").is_none() {
        std::env::set_var("WEBKIT_DISABLE_DMABUF_RENDERER", "1");
    }
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .manage(Runner::default())
        .manage(EngineState::default())
        .invoke_handler(tauri::generate_handler![
            default_settings,
            prepare_engine,
            inspect_folders,
            start_batch,
            cancel_batch,
            open_folder
        ])
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                let runner = window.state::<Runner>();
                if runner.running() {
                    api.prevent_close();
                    let _ = runner.cancel();
                    let _ = window.emit("closing_requested", ());
                }
            }
        })
        .build(tauri::generate_context!())
        .expect("Could not initialize YAMS desktop");
    app.run(|app, event| {
        if let tauri::RunEvent::ExitRequested { api, .. } = event {
            let runner = app.state::<Runner>();
            if runner.running() {
                api.prevent_exit();
                let _ = runner.cancel();
                let _ = app.emit("closing_requested", ());
            }
        }
    });
}
