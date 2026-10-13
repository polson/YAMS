use serde_json::Value;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use yams_desktop::{inspect, Job, Runner, Settings};

static NEXT: AtomicU64 = AtomicU64::new(0);

struct Fixture(PathBuf);

impl Fixture {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "yams-bridge-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(path.join("yams")).unwrap();
        fs::write(path.join("yams/__init__.py"), "").unwrap();
        fs::write(
            path.join("yams/__main__.py"),
            r#"import json, pathlib, sys
if '--inspect-list' in sys.argv:
    paths = json.load(sys.stdin)
    print(json.dumps(dict(event='previews', songs=[dict(folder=str(pathlib.Path(path).resolve()), status='ready', source='song.ogg', backup='backup/song.ogg', message='') for path in paths])))
    sys.exit(0)
folder = pathlib.Path(sys.argv[1]).resolve()
name = folder.name
def emit(event, **data):
    print(json.dumps(dict(event=event, folder=str(folder), **data)), flush=True)
if '--inspect' in sys.argv:
    emit('preview', status='ready', source='song.ogg', backup='backup/song.ogg')
elif '--recover' in sys.argv:
    emit('recovered', result='rolled_back')
elif name == 'cancel':
    emit('model_started', target='vocal', pass_number=1)
    assert sys.stdin.readline().strip() == 'cancel'
    emit('recovered', result='discarded_staging')
    emit('cancelled')
    sys.exit(130)
elif name == 'moved.con':
    assert sys.argv[1] == str(folder)
    emit('started', source=name)
    folder.rename(folder.with_suffix('.backup'))
    emit('completed', validation=dict(peak_error=0))
elif name == 'wrong-folder':
    print(json.dumps(dict(event='completed', folder=str(folder.parent))), flush=True)
    assert sys.stdin.readline().strip() == 'cancel'
    emit('cancelled')
    sys.exit(130)
elif name == 'invalid':
    print('invalid-json', flush=True)
    sys.stdin.readline()
    emit('cancelled')
    sys.exit(130)
elif name == 'incomplete-progress':
    emit('model_progress', fraction=0.5)
    assert sys.stdin.readline().strip() == 'cancel'
    emit('cancelled')
    sys.exit(130)
elif name in ('models-failure', 'runtime-failure'):
    emit('runtime_error' if name == 'runtime-failure' else 'models_error', message='download failed')
    emit('error', message='download failed')
    sys.exit(1)
elif name == 'failure':
    print('diagnostic café', file=sys.stderr)
    emit('error', message='model failed')
    sys.exit(1)
elif name == 'pending':
    emit('recovery_required', message='changed file retained')
    emit('error', message='install interrupted')
    sys.exit(1)
elif name == 'false-success':
    emit('completed')
    sys.exit(1)
elif name == 'skip':
    emit('skipped')
elif name == 'check-output-folder':
    assert '--output-folder' in sys.argv
    emit('completed', validation=dict(peak_error=0))
else:
    emit('completed', validation=dict(peak_error=0))
"#,
        )
        .unwrap();
        Self(path)
    }

    fn settings(&self) -> Settings {
        Settings {
            python: std::env::var("YAMS_TEST_PYTHON")
                .or_else(|_| std::env::var("YASS_TEST_PYTHON"))
                .unwrap_or_else(|_| "/usr/bin/python3".into()),
            project: self.0.to_string_lossy().into_owned(),
            cache: self.0.to_string_lossy().into_owned(),
            models: "models".into(),
            inference_source: "source".into(),
            device: "auto".into(),
            model_profile: "mega-8stem".into(),
            installer: None,
        }
    }

    fn job(&self, name: &str, recover: bool) -> Job {
        let folder = self.0.join(name);
        fs::create_dir_all(&folder).unwrap();
        fs::write(folder.join("song.ini"), "[song]").unwrap();
        Job {
            id: name.into(),
            folder: folder.to_string_lossy().into_owned(),
            output_folder: None,
            recover,
        }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn queue_continues_after_failure_and_checks_exit_status() {
    let fixture = Fixture::new();
    let jobs = ["failure", "skip", "success", "pending", "false-success"]
        .iter()
        .map(|name| fixture.job(name, false))
        .collect::<Vec<_>>();
    let runner = Runner::default();
    let events = Arc::new(Mutex::new(Vec::<Value>::new()));
    let received = events.clone();
    runner.reserve(&fixture.settings(), &jobs).unwrap();
    assert!(runner.reserve(&fixture.settings(), &jobs).is_err());
    runner.run(
        fixture.settings(),
        jobs,
        Arc::new(move |event| received.lock().unwrap().push(event)),
    );
    let events = events.lock().unwrap();
    let statuses = events
        .iter()
        .filter(|event| event["kind"] == "finished")
        .map(|event| event["status"].as_str().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        statuses,
        [
            "error",
            "skipped",
            "completed",
            "recovery_required",
            "error"
        ]
    );
    assert!(events
        .iter()
        .any(|event| event["message"] == "diagnostic café"));
    assert_eq!(events.last().unwrap()["kind"], "batch_finished");
    assert!(!runner.running());
}

#[test]
fn cancellation_waits_for_worker_and_stops_pending_jobs() {
    let fixture = Fixture::new();
    let jobs = vec![
        fixture.job("cancel", false),
        fixture.job("never-start", false),
    ];
    let runner = Runner::default();
    let cancellation = runner.clone();
    let events = Arc::new(Mutex::new(Vec::<Value>::new()));
    let received = events.clone();
    runner.reserve(&fixture.settings(), &jobs).unwrap();
    runner.run(
        fixture.settings(),
        jobs,
        Arc::new(move |event| {
            if event["payload"]["event"] == "model_started" {
                cancellation.cancel().unwrap();
            }
            received.lock().unwrap().push(event);
        }),
    );
    let events = events.lock().unwrap();
    assert_eq!(
        events
            .iter()
            .filter(|event| event["kind"] == "started")
            .count(),
        1
    );
    assert!(events
        .iter()
        .any(|event| event["payload"]["event"] == "recovered"));
    assert_eq!(
        events
            .iter()
            .filter(|event| event["status"] == "cancelled")
            .count(),
        2
    );
    assert!(!runner.running());
}

#[test]
fn protocol_failure_cancels_worker_and_stops_queue() {
    let fixture = Fixture::new();
    let jobs = vec![
        fixture.job("invalid", false),
        fixture.job("never-start", false),
    ];
    let runner = Runner::default();
    let events = Arc::new(Mutex::new(Vec::<Value>::new()));
    let received = events.clone();
    runner.reserve(&fixture.settings(), &jobs).unwrap();
    runner.run(
        fixture.settings(),
        jobs,
        Arc::new(move |event| received.lock().unwrap().push(event)),
    );
    let events = events.lock().unwrap();
    let statuses = events
        .iter()
        .filter(|event| event["kind"] == "finished")
        .map(|event| event["status"].as_str().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(statuses, ["error", "cancelled"]);
}

#[test]
fn incomplete_model_progress_is_rejected_before_forwarding() {
    let fixture = Fixture::new();
    let jobs = vec![fixture.job("incomplete-progress", false)];
    let runner = Runner::default();
    let events = Arc::new(Mutex::new(Vec::<Value>::new()));
    let received = events.clone();
    runner.reserve(&fixture.settings(), &jobs).unwrap();
    runner.run(
        fixture.settings(),
        jobs,
        Arc::new(move |event| received.lock().unwrap().push(event)),
    );
    let events = events.lock().unwrap();
    assert!(!events
        .iter()
        .any(|event| event["payload"]["event"] == "model_progress"));
    assert!(events
        .iter()
        .any(|event| event["kind"] == "finished" && event["status"] == "error"));
}

#[test]
fn discovery_deduplicates_parent_and_direct_song_paths() {
    let fixture = Fixture::new();
    let job = fixture.job("song", false);
    let previews = inspect(
        &fixture.settings(),
        vec![
            fixture.0.to_string_lossy().into_owned(),
            job.folder.clone(),
            job.folder,
        ],
    )
    .unwrap();
    assert_eq!(previews.len(), 1);
    assert_eq!(previews[0].status, "ready");
    assert_eq!(previews[0].backup, "backup/song.ogg");
}

#[test]
fn recovery_uses_worker_without_model_flags() {
    let fixture = Fixture::new();
    let jobs = vec![fixture.job("pending", true)];
    let runner = Runner::default();
    let events = Arc::new(Mutex::new(Vec::<Value>::new()));
    let received = events.clone();
    runner.reserve(&fixture.settings(), &jobs).unwrap();
    runner.run(
        fixture.settings(),
        jobs,
        Arc::new(move |event| received.lock().unwrap().push(event)),
    );
    assert!(events
        .lock()
        .unwrap()
        .iter()
        .any(|event| event["status"] == "recovered"));
}

#[test]
fn duplicate_folders_do_not_reserve_a_queue() {
    let fixture = Fixture::new();
    let job = fixture.job("song", false);
    let mut other = job.clone();
    other.id = "other".into();
    let runner = Runner::default();
    assert!(runner.reserve(&fixture.settings(), &[job, other]).is_err());
    assert!(!runner.running());
}

#[test]
fn model_setup_failure_stops_pending_jobs() {
    check_setup_failure("models-failure");
}

#[test]
fn runtime_setup_failure_stops_pending_jobs() {
    check_setup_failure("runtime-failure");
}

fn check_setup_failure(name: &str) {
    let fixture = Fixture::new();
    let jobs = vec![fixture.job(name, false), fixture.job("never-start", false)];
    let runner = Runner::default();
    let events = Arc::new(Mutex::new(Vec::<Value>::new()));
    let received = events.clone();
    runner.reserve(&fixture.settings(), &jobs).unwrap();
    runner.run(
        fixture.settings(),
        jobs,
        Arc::new(move |event| received.lock().unwrap().push(event)),
    );
    let events = events.lock().unwrap();
    assert_eq!(
        events
            .iter()
            .filter(|event| event["kind"] == "started")
            .count(),
        1
    );
    let statuses = events
        .iter()
        .filter(|event| event["kind"] == "finished")
        .map(|event| event["status"].as_str().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(statuses, ["error", "cancelled"]);
    assert!(!runner.running());
}

#[test]
fn output_folder_passed_to_worker() {
    let fixture = Fixture::new();
    let mut job = fixture.job("check-output-folder", false);
    job.output_folder = Some("/path/to/destination".into());
    let runner = Runner::default();
    let events = Arc::new(Mutex::new(Vec::<Value>::new()));
    let received = events.clone();
    runner.reserve(&fixture.settings(), &[job.clone()]).unwrap();
    runner.run(
        fixture.settings(),
        vec![job],
        Arc::new(move |event| received.lock().unwrap().push(event)),
    );
    let events = events.lock().unwrap();
    let statuses = events
        .iter()
        .filter(|event| event["kind"] == "finished")
        .map(|event| event["status"].as_str().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(statuses, ["completed"]);
}

#[test]
fn discovery_finds_con_files_directly_and_in_folders() {
    let fixture = Fixture::new();
    let con_file = fixture.0.join("sample.con");
    fs::write(&con_file, b"CON \x00\x00\x00\x00").unwrap();

    let subfolder = fixture.0.join("con_collection");
    fs::create_dir_all(&subfolder).unwrap();
    let nested_con = subfolder.join("nested.con");
    fs::write(&nested_con, b"LIVE\x00\x00\x00\x00").unwrap();

    let previews = inspect(
        &fixture.settings(),
        vec![
            con_file.to_string_lossy().into_owned(),
            subfolder.to_string_lossy().into_owned(),
        ],
    )
    .unwrap();
    assert_eq!(previews.len(), 2);

    let job = Job {
        id: "con-job".into(),
        folder: con_file.to_string_lossy().into_owned(),
        output_folder: None,
        recover: false,
    };
    let runner = Runner::default();
    assert!(runner.reserve(&fixture.settings(), &[job]).is_ok());
}

#[test]
fn discovery_finds_excon_folders_directly_and_nested() {
    let fixture = Fixture::new();

    // 1. Direct excon folder (songs.dta and .mogg inside it)
    let direct_excon = fixture.0.join("direct_excon");
    fs::create_dir_all(&direct_excon).unwrap();
    fs::write(direct_excon.join("songs.dta"), b"(song1 (name \"Song 1\"))").unwrap();
    fs::write(direct_excon.join("song1.mogg"), b"\x00\x00\x00\x00").unwrap();

    // 2. Nested excon collection (parent with songs.dta, subfolders with .mogg)
    let collection = fixture.0.join("excon_collection");
    fs::create_dir_all(&collection).unwrap();
    fs::write(collection.join("songs.dta"), b"(song2 (name \"Song 2\"))").unwrap();

    let sub_song = collection.join("sub_song");
    fs::create_dir_all(&sub_song).unwrap();
    fs::write(sub_song.join("sub_song.mogg"), b"\x00\x00\x00\x00").unwrap();

    let previews = inspect(
        &fixture.settings(),
        vec![
            direct_excon.to_string_lossy().into_owned(),
            collection.to_string_lossy().into_owned(),
        ],
    )
    .unwrap();
    assert_eq!(previews.len(), 2);
}

#[test]
fn discovery_finds_audio_files_and_albums() {
    let fixture = Fixture::new();

    // 1. Direct audio file
    let audio_file = fixture.0.join("solo_track.opus");
    fs::write(&audio_file, b"opus audio data").unwrap();

    // 2. Album folder with multiple tracks
    let album = fixture.0.join("album");
    fs::create_dir_all(&album).unwrap();
    fs::write(album.join("track1.mp3"), b"track1").unwrap();
    fs::write(album.join("track2.flac"), b"track2").unwrap();

    // 3. Multitrack stem folder
    let stem_folder = fixture.0.join("stems");
    fs::create_dir_all(&stem_folder).unwrap();
    fs::write(stem_folder.join("vocals.ogg"), b"vocals").unwrap();
    fs::write(stem_folder.join("drums.wav"), b"drums").unwrap();

    let previews = inspect(
        &fixture.settings(),
        vec![
            audio_file.to_string_lossy().into_owned(),
            album.to_string_lossy().into_owned(),
            stem_folder.to_string_lossy().into_owned(),
        ],
    )
    .unwrap();
    // 1 direct audio + 2 tracks in album + 1 stem folder unit = 4 items
    assert_eq!(previews.len(), 4);

    let job = Job {
        id: "audio-job".into(),
        folder: audio_file.to_string_lossy().into_owned(),
        output_folder: None,
        recover: false,
    };
    let runner = Runner::default();
    assert!(runner.reserve(&fixture.settings(), &[job]).is_ok());
}

#[test]
fn completion_after_source_move_keeps_queue_running() {
    let fixture = Fixture::new();
    let source = fixture.0.join("moved.con");
    fs::write(&source, "CON source").unwrap();
    let jobs = vec![
        Job {
            id: "moved.con".into(),
            folder: fixture.0.join(".").join("moved.con").to_string_lossy().into_owned(),
            output_folder: None,
            recover: false,
        },
        fixture.job("success", false),
    ];
    let runner = Runner::default();
    let events = Arc::new(Mutex::new(Vec::<Value>::new()));
    let received = events.clone();
    runner.reserve(&fixture.settings(), &jobs).unwrap();
    runner.run(
        fixture.settings(),
        jobs,
        Arc::new(move |event| received.lock().unwrap().push(event)),
    );
    let events = events.lock().unwrap();
    let statuses = events
        .iter()
        .filter(|event| event["kind"] == "finished")
        .map(|event| event["status"].as_str().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(statuses, ["completed", "completed"]);
    assert!(!source.exists());
    assert_eq!(
        fs::read_to_string(source.with_extension("backup")).unwrap(),
        "CON source"
    );
}

#[test]
fn mismatched_folder_cancels_worker_and_stops_queue() {
    let fixture = Fixture::new();
    let jobs = vec![
        fixture.job("wrong-folder", false),
        fixture.job("never-start", false),
    ];
    let runner = Runner::default();
    let events = Arc::new(Mutex::new(Vec::<Value>::new()));
    let received = events.clone();
    runner.reserve(&fixture.settings(), &jobs).unwrap();
    runner.run(
        fixture.settings(),
        jobs,
        Arc::new(move |event| received.lock().unwrap().push(event)),
    );
    let events = events.lock().unwrap();
    let statuses = events
        .iter()
        .filter(|event| event["kind"] == "finished")
        .map(|event| event["status"].as_str().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(statuses, ["error", "cancelled"]);
}
