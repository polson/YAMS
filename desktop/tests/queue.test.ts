import test from 'node:test';
import assert from 'node:assert/strict';
import { mergePreviews, queueProgress, refreshSong, resolveExportTargets, rowWindow, ROW_HEIGHT, duration, timing, updateSong, scrollbarGeometry, scrollPositionFromThumb, type Song } from '../src/queue.ts';

const song: Song = { id: 'a', folder: '/songs/a', status: 'ready', source: 'song.opus', backup: 'backup/song.opus', message: '', detail: '', progress: 0, log: '', canRetry: false, startedAt: 0, finishedAt: 0, inferenceAt: 0 };

test('model progress stays below completion until exit-confirmed installation', () => {
  const progress = updateSong(song, { kind: 'worker', id: 'a', payload: { event: 'model_finished', folder: song.folder, target: 'drumsep', pass_number: 2 } });
  assert.equal(progress.progress, 95);
  assert.equal(progress.status, 'ready');
  const reported = updateSong(progress, { kind: 'worker', id: 'a', payload: { event: 'completed', folder: song.folder } });
  assert.notEqual(reported.status, 'completed');
  const finished = updateSong(reported, { kind: 'finished', id: 'a', status: 'completed', message: '' });
  assert.equal(finished.status, 'completed');
  assert.equal(finished.progress, 100);
});

test('recovery and failure are retained as actionable statuses', () => {
  const result = updateSong(song, { kind: 'finished', id: 'a', status: 'recovery_required', message: 'File changed; retained' });
  assert.equal(result.status, 'recovery_required');
  assert.equal(result.message, 'File changed; retained');
});

test('duplicate folder previews preserve existing results', () => {
  const result = mergePreviews([song], [{ ...song, status: 'skipped' }, { ...song, folder: '/songs/b' }]);
  assert.equal(result.length, 2);
  assert.equal(result[0].status, 'ready');
  assert.notEqual(result[1].id, song.id);
});

test('diagnostics stay bounded and do not alter song status', () => {
  const result = updateSong(song, { kind: 'diagnostic', id: 'a', message: 'x'.repeat(10000) });
  assert.equal(result.log.length, 8192);
  assert.equal(result.status, 'ready');
});

test('6-stem progress scales to 95% and displays clean detail', () => {
  const half = updateSong(song, { kind: 'worker', id: 'a', payload: { event: 'model_progress', folder: song.folder, target: '6stem', pass_number: 1, fraction: 0.5 } });
  assert.equal(half.progress, 47.5);
  assert.equal(half.detail, 'Separating 6 stems');
  const finish = updateSong(half, { kind: 'worker', id: 'a', payload: { event: 'model_finished', folder: song.folder, target: '6stem', pass_number: 1 } });
  assert.equal(finish.progress, 95);
});

test('6-stem progress preserves explicit worker detail', () => {
  const half = updateSong(song, { kind: 'worker', id: 'a', payload: { event: 'model_progress', folder: song.folder, target: '6stem', pass_number: 1, fraction: 0.5, detail: 'Separating 6 stems' } });
  assert.equal(half.progress, 47.5);
  assert.equal(half.detail, 'Separating 6 stems');
  const finish = updateSong(half, { kind: 'worker', id: 'a', payload: { event: 'model_finished', folder: song.folder, target: '6stem', pass_number: 1 } });
  assert.equal(finish.progress, 95);
  assert.equal(finish.detail, 'Separating 6 stems');
});

test('8-stem 2-pass progress scales across 6stem and drumsep smoothly to 95%', () => {
  const pass1 = updateSong(song, { kind: 'worker', id: 'a', payload: { event: 'model_progress', folder: song.folder, target: '6stem', pass_number: 1, total_passes: 2, fraction: 0.5, detail: 'Separating 6 stems' } });
  assert.equal(pass1.progress, 23.75);
  assert.equal(pass1.detail, 'Separating 6 stems');
  const pass1End = updateSong(pass1, { kind: 'worker', id: 'a', payload: { event: 'model_finished', folder: song.folder, target: '6stem', pass_number: 1, total_passes: 2, detail: 'Separating 6 stems' } });
  assert.equal(pass1End.progress, 47.5);
  const pass2 = updateSong(pass1End, { kind: 'worker', id: 'a', payload: { event: 'model_progress', folder: song.folder, target: 'drumsep', pass_number: 2, total_passes: 2, fraction: 0.5, detail: 'Separating drums' } });
  assert.equal(pass2.progress, 71.25);
  assert.equal(pass2.detail, 'Separating drums');
  const pass2End = updateSong(pass2, { kind: 'worker', id: 'a', payload: { event: 'model_finished', folder: song.folder, target: 'drumsep', pass_number: 2, total_passes: 2, detail: 'Separating drums' } });
  assert.equal(pass2End.progress, 95);
});

test('automatic refresh preserves results and exposes safe retry', () => {
  for (const status of ['cancelled', 'error', 'recovered'] as const) {
    const result = refreshSong({ ...song, status, message: 'Previous result' }, { ...song, status: 'ready' });
    assert.equal(result.status, status);
    assert.equal(result.message, 'Previous result');
    assert.equal(result.canRetry, true);
  }
  const pending = refreshSong({ ...song, status: 'error' }, { ...song, status: 'recovery_required', message: 'Interrupted installation' });
  assert.equal(pending.status, 'recovery_required');
  assert.equal(pending.canRetry, false);
});

test('successful jobs release diagnostic buffers', () => {
  const result = updateSong({ ...song, log: 'x'.repeat(8192) }, { kind: 'finished', id: 'a', status: 'completed', message: '' });
  assert.equal(result.log, '');
  assert.equal(result.canRetry, false);
});

test('9000-song collection keeps job identity across indexed progress updates', () => {
  const previews = Array.from({ length: 9000 }, (_, index) => ({ ...song, folder: `/songs/${index}` }));
  const collection = mergePreviews([], previews);
  const positions = new Map(collection.map((song, index) => [song.id, index]));
  assert.equal(positions.size, 9000);
  const active = collection[8999];
  const first = collection[0];
  for (let update = 0; update < 1000; update++) {
    const index = positions.get(active.id)!;
    Object.assign(collection[index], updateSong(collection[index], { kind: 'worker', id: active.id, payload: { event: 'model_progress', folder: active.folder, target: '6stem', pass_number: 1, fraction: update / 1000 } }));
  }
  assert.equal(collection[0], first);
  assert.equal(collection[8999], active);
});

test('timing begins on worker start, freezes on exit and resets on retry', () => {
  const started = updateSong(song, { kind: 'started', id: 'a' }, 1000);
  assert.equal(timing(started, 3000), '0:02 / —');
  const progress = { ...started, progress: 25 };
  assert.equal(timing(progress, 11000), '0:10 / 0:38');
  assert.equal(timing({ ...progress, progress: 95 }, 11000), '0:10 · ~15s left');
  const completed = updateSong(progress, { kind: 'finished', id: 'a', status: 'completed', message: '' }, 11000);
  assert.equal(timing(completed, 99999), '0:10');
  const retried = updateSong(completed, { kind: 'started', id: 'a' }, 20000);
  assert.equal(timing(retried, 20000), '0:00 / —');
});

test('virtual scrolling covers 9000 rows with bounded rendering and exact spacer geometry', () => {
  const collection = mergePreviews([], Array.from({ length: 9000 }, (_, index) => ({ ...song, folder: `/songs/${index}` })));
  for (const height of [0, 120, 400, 1000]) {
    for (let index = 0; index < collection.length; index += 37) {
      const window = rowWindow(collection.length, index * ROW_HEIGHT, height);
      assert(window.start <= index && window.end > index);
      assert(window.end - window.start <= Math.ceil(height / ROW_HEIGHT) + 10);
      assert.equal(window.top + (window.end - window.start) * ROW_HEIGHT + window.bottom, collection.length * ROW_HEIGHT);
    }
  }
  assert.deepEqual(rowWindow(0, 10000, 400), { start: 0, end: 0, top: 0, bottom: 0 });
});

test('scrollbar geometry is only active when rows overflow viewport', () => {
  assert.deepEqual(scrollbarGeometry(0, 0, 400), { isScrollable: false, thumbHeight: 0, thumbTop: 0 });
  assert.deepEqual(scrollbarGeometry(5, 0, 400), { isScrollable: false, thumbHeight: 0, thumbTop: 0 });
  assert.deepEqual(scrollbarGeometry(10, 0, 520), { isScrollable: false, thumbHeight: 0, thumbTop: 0 });

  const overflowing = scrollbarGeometry(10, 0, 400);
  assert.equal(overflowing.isScrollable, true);
  assert.equal(overflowing.thumbTop, 0);
  assert.equal(overflowing.thumbHeight, (400 / 520) * 400);

  const scrolledToEnd = scrollbarGeometry(10, 120, 400);
  assert.equal(scrolledToEnd.isScrollable, true);
  assert.equal(Math.round(scrolledToEnd.thumbTop + scrolledToEnd.thumbHeight), 400);

  const clampedOverscroll = scrollbarGeometry(10, 9999, 400);
  assert.equal(clampedOverscroll.thumbTop, scrolledToEnd.thumbTop);

  const massiveCollection = scrollbarGeometry(10000, 0, 400);
  assert.equal(massiveCollection.thumbHeight, 28);

  const mappedScroll = scrollPositionFromThumb(scrolledToEnd.thumbTop, 10, 400, scrolledToEnd.thumbHeight);
  assert.equal(Math.round(mappedScroll), 120);
});

test('download progress is displayed without inventing separation progress or completion', () => {
  const started = updateSong(song, { kind: 'started', id: 'a' }, 0);
  const downloading = updateSong(started, { kind: 'worker', id: 'a', payload: { event: 'model_download', folder: song.folder, asset: 'BS-Rofo-SW-Fixed.ckpt', received: 50, total: 100 } });
  assert.equal(downloading.detail, 'Downloading BS-Rofo-SW-Fixed.ckpt · 50%');
  assert.equal(downloading.progress, 0);
  assert.equal(downloading.status, 'running');
  assert.equal(timing(downloading, 1000), '0:01 / —');
});

test('download time stays in elapsed time and is excluded from inference rate estimates', () => {
  const started = updateSong(song, { kind: 'started', id: 'a' }, 0);
  const inference = updateSong(started, { kind: 'worker', id: 'a', payload: { event: 'model_started', folder: song.folder, target: '6stem', pass_number: 1 } }, 120000);
  assert.equal(timing({ ...inference, progress: 10 }, 130000), '2:10 / 3:35');
  const secondPass = updateSong(inference, { kind: 'worker', id: 'a', payload: { event: 'model_started', folder: song.folder, target: 'drumsep', pass_number: 2 } }, 240000);
  assert.equal(secondPass.inferenceAt, 120000);
});

test('durations use m:ss below an hour and h:mm:ss above it', () => {
  for (const [milliseconds, expected] of [[0, '0:00'], [59000, '0:59'], [60000, '1:00'], [3599000, '59:59'], [3600000, '1:00:00'], [3661000, '1:01:01'], [360000000, '100:00:00']] as const) {
    assert.equal(duration(milliseconds), expected);
  }
});

test('queueProgress tracks completed vs remaining and active fraction across 9000 songs', () => {
  assert.deepEqual(queueProgress([]), { completed: 0, remaining: 0, failed: 0, skipped: 0, total: 0, percent: 0 });

  const collection: Song[] = Array.from({ length: 9000 }, (_, index) => {
    let status: Song['status'] = 'ready';
    let progress = 0;
    if (index < 2000) {
      status = 'completed';
      progress = 100;
    } else if (index === 2000) {
      status = 'running';
      progress = 50;
    } else if (index >= 8900) {
      status = 'skipped';
    } else if (index >= 8850) {
      status = 'error';
    }
    return { ...song, id: String(index), status, progress };
  });

  const stats = queueProgress(collection);
  assert.equal(stats.total, 9000);
  assert.equal(stats.completed, 2000);
  assert.equal(stats.remaining, 6850);
  assert.equal(stats.failed, 50);
  assert.equal(stats.skipped, 100);
  assert.equal(stats.percent.toFixed(2), '22.48');
});

test('resolveExportTargets mirrors relative directories and disambiguates collisions', () => {
  const single = resolveExportTargets(['/Songs/Artist/TrackA'], '/Exports');
  assert.equal(single.get('/Songs/Artist/TrackA'), '/Exports/TrackA');

  const nested = resolveExportTargets([
    '/Songs/Rock/Pack1/TrackA',
    '/Songs/Rock/Pack2/TrackB',
  ], '/Exports');
  assert.equal(nested.get('/Songs/Rock/Pack1/TrackA'), '/Exports/Pack1/TrackA');
  assert.equal(nested.get('/Songs/Rock/Pack2/TrackB'), '/Exports/Pack2/TrackB');

  const colliding = resolveExportTargets([
    '/Drive1/TrackA',
    '/Drive2/TrackA',
  ], '/Exports');
  assert.equal(colliding.get('/Drive1/TrackA'), '/Exports/TrackA');
  assert.equal(colliding.get('/Drive2/TrackA'), '/Exports/TrackA (1)');

  const conPackages = resolveExportTargets([
    '/DLC/Rock/song1.con',
    '/DLC/Metal/song2.live',
  ], '/Exports');
  assert.equal(conPackages.get('/DLC/Rock/song1.con'), '/Exports/Rock/song1');
  assert.equal(conPackages.get('/DLC/Metal/song2.live'), '/Exports/Metal/song2');

  const audioFiles = resolveExportTargets([
    '/Audio/Downloads/solo.opus',
    '/Audio/Music/Album/track.flac',
  ], '/Exports');
  assert.equal(audioFiles.get('/Audio/Downloads/solo.opus'), '/Exports/Downloads/solo');
  assert.equal(audioFiles.get('/Audio/Music/Album/track.flac'), '/Exports/Music/Album/track');

  const singleAudio = resolveExportTargets(['/Downloads/song.mp3'], '/Exports');
  assert.equal(singleAudio.get('/Downloads/song.mp3'), '/Exports/song');

  const networkFolders = resolveExportTargets([
    '\\\\server\\share\\Songs\\Pack1\\track.opus',
    '\\\\server\\share\\Songs\\Pack2\\track.opus',
  ], '\\\\server\\share\\Exports');
  assert.equal(networkFolders.get('\\\\server\\share\\Songs\\Pack1\\track.opus'), '\\\\server\\share\\Exports\\Pack1\\track');
  assert.equal(networkFolders.get('\\\\server\\share\\Songs\\Pack2\\track.opus'), '\\\\server\\share\\Exports\\Pack2\\track');
});

test('worker destination updates song destination property', () => {
  const updated = updateSong(song, {
    kind: 'worker',
    id: 'a',
    payload: { event: 'started', folder: song.folder, destination: '/unpacked/dest' },
  });
  assert.equal(updated.destination, '/unpacked/dest');
});

test('post-processing events advance progress smoothly through 95% to 100% with explicit badges', () => {
  const current = { ...song, progress: 95, status: 'running' as const };
  const finalizing = updateSong(current, {
    kind: 'worker',
    id: 'a',
    payload: { event: 'finalizing', folder: song.folder, detail: 'Finalizing stems…' },
  });
  assert.equal(finalizing.progress, 95);
  assert.equal(finalizing.detail, 'Finalizing stems…');

  const enc1 = updateSong(finalizing, {
    kind: 'worker',
    id: 'a',
    payload: { event: 'encoding', folder: song.folder, asset: 'drums_1.opus', index: 1, total: 8 },
  });
  assert.equal(enc1.progress, 95.5);
  assert.equal(enc1.detail, 'Encoding Opus (1/8)…');

  const enc8 = updateSong(enc1, {
    kind: 'worker',
    id: 'a',
    payload: { event: 'encoding', folder: song.folder, asset: 'song.opus', index: 8, total: 8 },
  });
  assert.equal(enc8.progress, 99);
  assert.equal(enc8.detail, 'Encoding Opus (8/8)…');

  const installing = updateSong(enc8, {
    kind: 'worker',
    id: 'a',
    payload: { event: 'installing', folder: song.folder },
  });
  assert.equal(installing.progress, 99.5);
  assert.equal(installing.detail, 'Saving…');

  const completed = updateSong(installing, {
    kind: 'finished',
    id: 'a',
    status: 'completed',
    message: '',
  });
  assert.equal(completed.progress, 100);
});

