import base64
import hashlib
import http.client
import json
import math
import os
import socket
import struct
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DESKTOP = ROOT / "desktop"


def request(port, method, path, data=None):
    body = None if data is None else json.dumps(data).encode()
    query = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=body, method=method,
                                   headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(query, timeout=40) as response:
        result = json.load(response)
    value = result.get("value")
    if isinstance(value, dict) and "error" in value:
        raise RuntimeError(value)
    return value


def main():
    with socket.socket() as available, socket.socket() as native:
        available.bind(("127.0.0.1", 0))
        port = available.getsockname()[1]
        native.bind(("127.0.0.1", 0))
        native_port = native.getsockname()[1]
    model_cache = tempfile.TemporaryDirectory(prefix="models-native-", dir=DESKTOP / ".cache")
    environment = os.environ.copy()
    models = Path(os.environ.get("YAMS_NATIVE_MODELS", DESKTOP / ".cache/native-smoke/models"))
    environment["YAMS_MODELS"] = str(models)
    if os.environ.get("YAMS_NATIVE_MODEL_DOWNLOAD") == "1":
        source_models = Path(os.environ["YAMS_NATIVE_SOURCE_MODELS"])
        for source in source_models.iterdir():
            if source.name != "BS-Rofo-SW-Fixed.yaml":
                os.link(source, Path(model_cache.name) / source.name)
        environment["YAMS_MODELS"] = model_cache.name
    driver = subprocess.Popen([str(DESKTOP / ".tools/bin/tauri-driver"), "--port", str(port), "--native-port", str(native_port)], env=environment,
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    session = None
    try:
        for attempt in range(100):
            try:
                status = request(port, "GET", "/status")
                if status["ready"]:
                    break
            except (urllib.error.URLError, http.client.RemoteDisconnected):
                if driver.poll() is not None:
                    raise RuntimeError(driver.stderr.read().decode())
            time.sleep(.1)
        else:
            raise RuntimeError("Native driver did not become ready")
        session = request(port, "POST", "/session", {"capabilities": {"alwaysMatch": {
            "tauri:options": {"application": str(DESKTOP / "src-tauri/target/debug/yams-desktop")}
        }}})["sessionId"]
        base = f"/session/{session}"

        def script(code, *args):
            return request(port, "POST", base + "/execute/sync", {"script": code, "args": args})

        def wait(code, timeout=30):
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if script(code):
                    return
                time.sleep(.1)
            raise AssertionError(script("return document.body.innerText"))

        wait("return document.querySelector('button.quiet') && !document.querySelector('button.quiet').disabled")
        config = request(port, "POST", base + "/execute/async", {
            "script": "const done=arguments[arguments.length-1]; window.__TAURI_INTERNALS__.invoke('default_settings').then(done).catch(error=>done({error:String(error)}));", "args": []
        })
        assert set(config) == {"device", "availableDevices", "modelProfile", "availableModelProfiles"}
        config.update({"python": "/obsolete/python", "project": "/obsolete/project", "inferenceSource": "/obsolete/source"})
        script("localStorage.setItem('yams-settings',JSON.stringify(arguments[0])); location.reload();", config)
        wait("return document.querySelector('.toggle input') && !document.querySelector('.toggle input').disabled")
        assert script("return Object.keys(JSON.parse(localStorage.getItem('yams-settings'))).sort()") == ["device", "exportFolder", "modelProfile"]
        assert script("return document.querySelector('.app-header .brand strong').textContent") == "YAMS"
        assert script("return document.querySelector('.brand-mark').getBoundingClientRect().width") == 36
        assert script("return document.querySelector('.titlebar').textContent.trim()") == ""
        assert script("return document.querySelectorAll('.titlebar button').length") == 3
        assert script("return document.querySelector('.titlebar [data-tauri-drag-region]').getBoundingClientRect().width") > 500
        decorated = request(port, "POST", base + "/execute/async", {
            "script": "const done = arguments[arguments.length - 1]; window.__TAURI_INTERNALS__.invoke('plugin:window|is_decorated', {label:'main'}).then(done).catch(error => done({error:String(error)}));", "args": []
        })
        assert decorated is False
        if os.environ.get("YAMS_NATIVE_WINDOW_CONTROLS") == "1":
            for double_click, expected in ((False, True), (False, False), (True, True), (True, False)):
                if double_click:
                    script("document.querySelector('.titlebar-drag').dispatchEvent(new MouseEvent('mousedown', {bubbles:true, button:0, detail:2}))")
                else:
                    script("document.querySelector('[aria-label=\"Maximize or restore window\"]').click()")
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    maximized = request(port, "POST", base + "/execute/async", {
                        "script": "const done = arguments[arguments.length - 1]; window.__TAURI_INTERNALS__.invoke('plugin:window|is_maximized', {label:'main'}).then(done).catch(error => done({error:String(error)}));", "args": []
                    })
                    if maximized is expected:
                        break
                    time.sleep(.1)
                else:
                    raise AssertionError(f"Expected native maximized state {expected}")
        assert script("return document.querySelectorAll('.path-field, .settings').length") == 0
        assert script("return !document.querySelector('.device-select select') || ['cuda:0', 'mps', 'cpu'].includes(document.querySelector('.device-select select').value)")
        assert script("return document.querySelector('.drop-zone h2').textContent") == "Drop song folders here"
        assert script("return document.querySelectorAll('.intro, .stem-strip, footer').length") == 0
        assert script("return document.querySelector('.drop-zone').getBoundingClientRect().height") >= 180
        assert script("return Math.min(...Array.from(document.querySelectorAll('body *')).filter(element => element.getClientRects().length && Array.from(element.childNodes).some(node => node.nodeType === 3 && node.textContent.trim())).map(element => parseFloat(getComputedStyle(element).fontSize)))") >= 14

        with tempfile.TemporaryDirectory(prefix="yams-native-") as temporary:
            collection = Path(temporary)
            ready = collection / "ready"
            skip = collection / "already-separated"
            pending = collection / "interrupted"
            for folder in (ready, skip, pending):
                folder.mkdir()
                (folder / "song.ini").write_text("[song]\nname = GUI fixture\n")
                (folder / "notes.mid").write_bytes(b"fixture chart")
            with wave.open(str(ready / "song.wav"), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(44100)
                audio.writeframes(b"".join(struct.pack("<h", int(4000 * math.sin(2 * math.pi * 440 * frame / 44100))) for frame in range(44100)))
            original_hash = hashlib.sha256((ready / "song.wav").read_bytes()).hexdigest()
            (skip / "vocals.ogg").write_bytes(b"existing stem")
            (pending / "song.ogg").write_bytes(b"original")
            (pending / ".yams-work").mkdir()
            request(port, "POST", base + "/execute/async", {
                "script": """const done = arguments[arguments.length - 1];
                    window.__TAURI_INTERNALS__.invoke('plugin:event|emit', {
                        event: 'tauri://drag-drop',
                        payload: { paths: [arguments[0], arguments[1]], position: {x: 0, y: 0} }
                    }).then(() => done(null)).catch(error => done({error: String(error)}));""",
                "args": [str(collection), str(ready)]
            })
            wait("return document.querySelectorAll('.song').length === 3")
            assert script("return document.querySelectorAll('[data-status=ready]').length") == 1
            assert script("return document.querySelectorAll('[data-status=skipped]').length") == 1
            assert script("return document.querySelectorAll('[data-status=recovery_required]').length") == 1
            assert not (ready / ".yams.lock").exists()
            assert script("return document.querySelectorAll('.remove:not(:disabled)').length") == 3
            assert script("return document.querySelectorAll('.open-folder:not(:disabled)').length") == 3
            script("const row = document.querySelector('[data-status=recovery_required]').closest('.song'); row.click(); if (!document.querySelector('.song-details')) throw new Error('Row click did not open information'); row.querySelector('.remove').click()")
            wait("return document.querySelectorAll('.song').length === 2 && !document.querySelector('.song-details')")
            assert (pending / ".yams-work").is_dir()
            assert (pending / "song.ogg").read_bytes() == b"original"
            request(port, "POST", base + "/execute/async", {
                "script": "const done = arguments[arguments.length - 1]; window.__TAURI_INTERNALS__.invoke('plugin:event|emit', {event:'tauri://drag-drop', payload:{paths:[arguments[0]],position:{x:0,y:0}}}).then(() => done(null)).catch(error => done({error:String(error)}));",
                "args": [str(pending)]
            })
            wait("return document.querySelectorAll('.song').length === 3 && !document.querySelector('.remove').disabled")
            script("Array.from(document.querySelectorAll('button')).find(button => button.textContent.startsWith('Recover ')).click()")
            wait("return document.querySelectorAll('[data-status=recovered]').length === 1 && !document.querySelector('.queue-footer .primary').disabled")
            assert not (pending / ".yams-work").exists()
            if os.environ.get("YAMS_NATIVE_INFERENCE") == "1":
                script("document.querySelector('.queue-footer .primary').click()")
                wait("return document.querySelectorAll('[data-status=running]').length === 1")
                assert script("return Array.from(document.querySelectorAll('button')).find(button => button.textContent === 'Clear').disabled")
                assert script("return Array.from(document.querySelectorAll('.remove')).every(button => button.disabled)")
                script("document.querySelector('.remove').click()")
                assert script("return document.querySelectorAll('.song').length") == 3
                assert script("return document.querySelector('[data-status=running]').closest('.song').querySelector('.timing').textContent.match(/^\\d+:[0-5]\\d/)")
                script("Array.from(document.querySelectorAll('button')).find(button => button.textContent === 'Clear').click()")
                assert script("return document.querySelectorAll('.song').length") == 3
                wait("return document.querySelectorAll('[data-status=completed]').length === 1", timeout=600 if os.environ.get("YAMS_ENGINE_FRESH") == "1" else 90)
                assert script("return Array.from(document.querySelectorAll('.song button')).some(button => button.getAttribute('aria-label') === 'Open folder')")
                opener_log = os.environ.get("YAMS_NATIVE_OPENER_LOG")
                if opener_log:
                    script("document.querySelector('[data-status=completed]').closest('.song').querySelector('.open-folder').click()")
                    deadline = time.monotonic() + 10
                    while not Path(opener_log).exists() and time.monotonic() < deadline:
                        time.sleep(.1)
                    assert Path(opener_log).read_text().strip() == str(ready.resolve())
                assert script("return Array.from(document.querySelectorAll('button')).find(button => button.textContent === 'Clear').disabled") is False
                assert hashlib.sha256((ready / "backup/song.wav").read_bytes()).hexdigest() == original_hash
                assert len(list(ready.glob("*.opus"))) == 8
                assert not list(ready.glob("*.wav"))
                if os.environ.get("YAMS_NATIVE_MODEL_DOWNLOAD") == "1":
                    registry = json.loads((ROOT / "yams/model_assets.json").read_text())
                    config = "BS-Rofo-SW-Fixed.yaml"
                    assert hashlib.sha256((Path(model_cache.name) / config).read_bytes()).hexdigest() == registry["torch_6stem_files"][config]["sha256"]
                    print("Native first-use missing-model download and verified CUDA separation passed")
                assert not (ready / ".yams-work").exists()
                assert script("return document.querySelectorAll('[data-status=completed]').length") == 1
                assert script("return Array.from(document.querySelectorAll('button')).some(button => button.textContent === 'Recheck')") is False
            else:
                script("document.querySelector('.queue-footer .primary').click()")
                wait("return document.querySelectorAll('[data-status=running]').length === 1")
                assert script("return Array.from(document.querySelectorAll('button')).find(button => button.textContent === 'Clear').disabled")
                assert script("return Array.from(document.querySelectorAll('.remove')).every(button => button.disabled)")
                script("document.querySelector('.remove').click()")
                assert script("return document.querySelectorAll('.song').length") == 3
                script("document.querySelector('.stop').click()")
                wait("return document.querySelectorAll('[data-status=cancelled]').length === 1 && !document.querySelector('.stop')")
                assert hashlib.sha256((ready / "song.wav").read_bytes()).hexdigest() == original_hash
                assert not (ready / ".yams-work").exists()
                wait("return Array.from(document.querySelectorAll('.song button')).some(button => button.textContent === 'Retry' && !button.disabled)")
                script("Array.from(document.querySelectorAll('.song')).find(row => row.querySelector('h3').textContent === 'ready').querySelectorAll('button').forEach(button => { if (button.textContent === 'Retry') button.click(); })")
                wait("return document.querySelectorAll('[data-status=running]').length === 1")
                assert script("return Array.from(document.querySelectorAll('button')).find(button => button.textContent === 'Clear').disabled")
                assert script("return Array.from(document.querySelectorAll('.remove')).every(button => button.disabled)")
                script("document.querySelector('.remove').click()")
                assert script("return document.querySelectorAll('.song').length") == 3
                script("document.querySelector('.stop').click()")
                wait("return document.querySelectorAll('[data-status=cancelled]').length === 1 && !document.querySelector('.stop')")
                assert hashlib.sha256((ready / "song.wav").read_bytes()).hexdigest() == original_hash
            if os.environ.get("YAMS_NATIVE_LARGE_QUEUE") == "1":
                bulk = collection / "bulk"
                bulk.mkdir()
                for index in range(9000):
                    folder = bulk / f"song-{index:05d}"
                    folder.mkdir()
                    (folder / "song.ini").write_text("[song]\n")
                    (folder / "notes.mid").write_bytes(b"chart")
                    (folder / "song.ogg").write_bytes(b"original")
                loaded = time.monotonic()
                request(port, "POST", base + "/execute/async", {
                    "script": "const done = arguments[arguments.length - 1]; window.__TAURI_INTERNALS__.invoke('plugin:event|emit', {event:'tauri://drag-drop', payload:{paths:[arguments[0]],position:{x:0,y:0}}}).then(() => done(null)).catch(error => done({error:String(error)}));",
                    "args": [str(bulk)]
                })
                wait("return document.querySelector('.queue .section-heading span').textContent.startsWith('9003 songs') && !Array.from(document.querySelectorAll('button')).find(button => button.textContent === 'Clear').disabled", timeout=60)
                assert script("return document.querySelectorAll('.song').length <= Math.ceil(document.querySelector('.song-list').clientHeight / 52) + 10")
                assert script("return Array.from(document.querySelectorAll('.song')).every(row => row.getBoundingClientRect().height === 52)")
                script("const list = document.querySelector('.song-list'); list.scrollTop = list.scrollHeight / 2")
                wait("return Array.from(document.querySelectorAll('.song h3')).some(row => row.textContent.startsWith('song-04'))")
                script("const list = document.querySelector('.song-list'); list.scrollTop = list.scrollHeight")
                wait("return Array.from(document.querySelectorAll('.song h3')).some(row => row.textContent === 'song-08999')")
                assert script("return document.querySelectorAll('.song').length") < 30
                print(json.dumps({"large_queue_records":9003,"bounded_native_rows":"passed","scroll_middle_and_end":"passed","preview_and_scroll_seconds":round(time.monotonic()-loaded,2)}))
            image = request(port, "GET", base + "/screenshot")
            screenshot = DESKTOP / ".cache/native-smoke.png"
            screenshot.parent.mkdir(parents=True, exist_ok=True)
            screenshot.write_bytes(base64.b64decode(image))
            assert script("return document.querySelectorAll('[role=alert]').length") == 0
            if os.environ.get("YAMS_NATIVE_WINDOW_CONTROLS") == "1":
                script("document.querySelector('[aria-label=\"Minimize window\"]').click()")
                minimized = request(port, "POST", base + "/execute/async", {
                    "script": "const done = arguments[arguments.length - 1]; setTimeout(() => window.__TAURI_INTERNALS__.invoke('plugin:window|is_minimized', {label:'main'}).then(done).catch(error => done({error:String(error)})), 300);", "args": []
                })
                assert minimized is True
            if os.environ.get("YAMS_NATIVE_INFERENCE") != "1":
                script("document.querySelector('.song-list').scrollTop = 0")
                wait("return Array.from(document.querySelectorAll('button')).some(button => button.textContent === 'Retry')")
                script("Array.from(document.querySelectorAll('button')).find(button => button.textContent === 'Retry').click()")
                wait("return document.querySelectorAll('[data-status=running]').length === 1")
            script("setTimeout(() => document.querySelector('.window-close').click(), 100)")
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                try:
                    if not request(port, "GET", base + "/window/handles"):
                        break
                except urllib.error.HTTPError as problem:
                    value = json.load(problem)["value"]
                    assert (value["error"] in ("invalid session id", "no such window")
                            or value.get("message") == "Session terminated without a reply"), value
                    break
                time.sleep(.1)
            else:
                raise AssertionError("Window did not close after stopping the worker")
            if os.environ.get("YAMS_NATIVE_INFERENCE") != "1":
                assert hashlib.sha256((ready / "song.wav").read_bytes()).hexdigest() == original_hash
                assert not (ready / ".yams-work").exists()
            print(json.dumps({"native_window": "passed", "folder_preview": "passed", "deduplication": "passed",
                              "existing_stem_skip": "passed", "recovery": "passed",
                              "custom_titlebar_and_safe_close": "passed",
                              "cancellation": "not requested" if os.environ.get("YAMS_NATIVE_INFERENCE") == "1" else "passed",
                              "inference": "passed" if os.environ.get("YAMS_NATIVE_INFERENCE") == "1" else "not requested",
                              "screenshot": str(screenshot)}))
    finally:
        if session is not None:
            try:
                request(port, "DELETE", f"/session/{session}")
            except (urllib.error.URLError, RuntimeError):
                pass
        driver.terminate()
        driver.wait(timeout=10)
        model_cache.cleanup()
        diagnostics = driver.stderr.read().decode()
        if diagnostics:
            print(diagnostics[-4000:])


if __name__ == "__main__":
    main()
