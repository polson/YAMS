export type Status = 'ready' | 'queued' | 'running' | 'completed' | 'skipped' | 'recovered' | 'cancelled' | 'error' | 'recovery_required';

export interface Settings {
  device: string;
  availableDevices?: string[];
  modelProfile?: string;
  availableModelProfiles?: string[];
  exportFolder?: string;
}

export interface Preview {
  folder: string;
  status: Status;
  message: string;
  source: string;
  backup: string;
}

export interface Song extends Preview {
  id: string;
  detail: string;
  progress: number;
  log: string;
  chunkSize?: number;
  canRetry: boolean;
  startedAt: number;
  finishedAt: number;
  inferenceAt: number;
  destination?: string;
}

interface WorkerEventFields {
  folder: string;
  source?: string;
  destination?: string;
  backup?: string;
  target?: string;
  name?: string;
  asset?: string;
  received?: number;
  total?: number;
  index?: number;
  pass_number?: number;
  total_passes?: number;
  fraction?: number;
  chunk_size?: number;
  seconds?: number;
  sample_rate?: number;
  frames?: number;
  channels?: number;
  reason?: string;
  detail?: string;
  message?: string;
  result?: string;
  codec?: string;
  peak_error?: number;
  rms_error?: number;
  encoder?: string;
  bitrate?: number;
  files?: Record<string, unknown>;
  validation?: AudioValidation;
}

type WorkerEvent = WorkerEventFields & (
  | { event: 'model_download'; asset: string; received: number; total: number }
  | { event: 'model_started'; target: string; pass_number: number }
  | { event: 'model_progress'; target: string; pass_number: number; fraction: number }
  | { event: 'model_finished'; target: string; pass_number: number }
  | { event: 'encoding'; index: number; total: number }
  | { event: 'started' | 'device' | 'decoded' | 'preparing_pass' | 'validated' | 'finalizing' | 'installing' | 'completed' | 'skipped' | 'recovered' | 'recovery_required' | 'cancelled' | 'error' | 'runtime_error' | 'models_error' | 'runtime_setup' | 'checking_models' | 'models_ready' }
);

interface AudioValidation {
  codec?: string;
  peak_error?: number;
  rms_error?: number;
  encoder?: string;
  bitrate?: number;
  files?: Record<string, unknown>;
}

export type BatchEvent =
  | { kind: 'batch_finished' }
  | { kind: 'started'; id: string }
  | { kind: 'finished'; id: string; status: Status; message: string }
  | { kind: 'diagnostic'; id: string; message: string }
  | { kind: 'worker'; id: string; payload: WorkerEvent };

const SKIP_REASONS: Record<string, string> = {
  already_contains_multitrack_stems: 'This song already contains multitrack stems; no separation was needed.',
  existing_instrument_or_alternate_stems: 'Existing instrument or alternate stems were found; no new stems were created.',
  already_separated_in_destination: 'The export destination already contains the separated stems.',
};

function skipReason(reason: string | undefined): string {
  return SKIP_REASONS[reason ?? ''] || 'The worker skipped this item without providing a reason.';
}

function appendLog(log: string, message: string): string {
  const entry = message.trim();
  if (!entry) { return log; }
  return `${log}${log && !log.endsWith('\n') ? '\n' : ''}${entry}\n`.slice(-8192);
}

function validationLog(validation: AudioValidation): string {
  const codec = validation.codec === 'pcm_f32le'
    ? 'PCM float'
    : validation.codec?.toUpperCase() || 'audio';
  const summary = [`Validated ${codec}`];
  if (validation.encoder) { summary.push(`encoder ${validation.encoder}`); }
  if (validation.bitrate !== undefined) { summary.push(`${validation.bitrate} bit/s`); }
  if (validation.peak_error !== undefined) {
    summary.push(`peak error ${validation.peak_error.toExponential(2)}`);
  }
  if (validation.rms_error !== undefined) {
    summary.push(`RMS error ${validation.rms_error.toExponential(2)}`);
  }
  const fileCount = validation.files ? Object.keys(validation.files).length : 0;
  if (fileCount) { summary.push(`${fileCount} files`); }
  return summary.join(' · ');
}

function workerLogMessage(payload: WorkerEvent): string | undefined {
  const pass = payload.pass_number === undefined ? '' : ` · pass ${payload.pass_number}`;
  const chunk = payload.chunk_size === undefined
    ? ''
    : ` · chunk size ${payload.chunk_size.toLocaleString()} samples`;
  switch (payload.event) {
    case 'started':
      return `Processing started${payload.source ? `: ${payload.source}` : '.'}`
        + (payload.destination ? `\nOutput: ${payload.destination}` : '');
    case 'device':
      return payload.name ? `Using device: ${payload.name}` : undefined;
    case 'decoded': {
      const { sample_rate, frames, channels } = payload;
      if (sample_rate === undefined || frames === undefined || channels === undefined) {
        return 'Audio decoded.';
      }
      const duration = (frames / sample_rate).toFixed(1);
      const channelLabel = channels === 1 ? 'channel' : 'channels';
      return `Audio decoded: ${sample_rate.toLocaleString()} Hz · ${channels} ${channelLabel}`
        + ` · ${duration} s · ${frames.toLocaleString()} frames`;
    }
    case 'model_started':
      return `Starting ${payload.target || 'model'}${pass}${chunk}`;
    case 'model_finished': {
      const elapsed = payload.seconds === undefined ? '' : ` in ${payload.seconds.toFixed(1)} s`;
      return `Finished ${payload.target || 'model'}${pass}${elapsed}${chunk}`;
    }
    case 'model_progress':
    case 'model_download':
      return undefined;
    case 'preparing_pass':
      return payload.detail || `Preparing next pass${pass}`;
    case 'validated':
      return validationLog({
        codec: payload.codec,
        peak_error: payload.peak_error,
        rms_error: payload.rms_error,
        encoder: payload.encoder,
        bitrate: payload.bitrate,
        files: payload.files,
      });
    case 'encoding': {
      if (payload.index !== 1) { return undefined; }
      return payload.total === undefined ? 'Encoding Opus stems' : `Encoding ${payload.total} Opus stems`;
    }
    case 'finalizing':
      return payload.detail || 'Finalizing stems.';
    case 'installing':
      return 'Installing outputs.';
    case 'completed':
      return `Processing completed${payload.destination ? `: ${payload.destination}` : '.'}`
        + (payload.backup ? `\nOriginal backup: ${payload.backup}` : '');
    case 'skipped':
      return `Skipped: ${skipReason(payload.reason)}`;
    case 'recovered':
      return `Recovery completed${payload.result ? `: ${payload.result}` : '.'}`;
    case 'recovery_required':
    case 'error':
    case 'runtime_error':
    case 'models_error':
      return `${payload.event.replace('_', ' ')}: ${payload.message || 'No explanation was provided.'}`;
    case 'cancelled':
      return 'Processing cancelled.';
    default:
      return undefined;
  }
}

export function updateSong(song: Song, event: Exclude<BatchEvent, { kind: 'batch_finished' }>, now = performance.now()): Song {
  if (event.kind === 'started') {
    return { ...song, status: 'running', detail: 'Starting…', progress: 0, message: '', log: appendLog(song.log, 'Processing attempt started.'), chunkSize: undefined, canRetry: false, startedAt: now, finishedAt: now, inferenceAt: now };
  }
  if (event.kind === 'finished') {
    const message = event.message || (event.status === 'skipped' ? song.message : '');
    return { ...song, status: event.status, message, finishedAt: now,
      canRetry: ['cancelled', 'error', 'recovered'].includes(event.status),
      log: event.status === 'completed' ? '' : song.log,
      detail: event.status === 'completed'
        ? (song.destination ? 'Stems exported to destination' : 'Stems installed · original backed up')
        : message || event.status,
      progress: event.status === 'completed' ? 100 : song.progress };
  }
  if (event.kind === 'diagnostic') {
    return { ...song, log: appendLog(song.log, event.message) };
  }
  const payload = event.payload;
  const logMessage = workerLogMessage(payload);
  const loggedSong = logMessage ? { ...song, log: appendLog(song.log, logMessage) } : song;
  const chunkSize = payload.chunk_size ?? song.chunkSize;
  if (payload.event === 'model_download') {
    return { ...loggedSong, detail: `Downloading ${payload.asset} · ${Math.round(100 * payload.received / payload.total)}%` };
  }
  if (payload.event === 'model_started' || payload.event === 'model_progress' || payload.event === 'model_finished') {
    const { target } = payload;
    const fraction = payload.event === 'model_finished' ? 1 : payload.event === 'model_progress' ? payload.fraction : 0;
    if (target === '6stem') {
      const isTwoPass = payload.total_passes === 2;
      return {
        ...loggedSong,
        progress: isTwoPass ? 95 * 0.5 * fraction : 95 * fraction,
        inferenceAt: payload.event === 'model_started' ? now : song.inferenceAt,
        detail: payload.detail || song.detail || 'Separating 6 stems',
        chunkSize,
      };
    }
    if (target === 'drumsep') {
      return {
        ...loggedSong,
        progress: 95 * (0.5 + 0.5 * fraction),
        detail: payload.detail || 'Separating drums',
        chunkSize,
      };
    }
    return loggedSong;
  }
  if (payload.event === 'encoding') {
    const index = payload.index;
    const total = payload.total;
    const detail = index && total ? `Encoding Opus (${index}/${total})…` : 'Encoding Opus…';
    const progress = index && total ? Math.max(song.progress, 95 + 4 * (index / total)) : Math.max(song.progress, 95);
    return { ...loggedSong, detail, progress };
  }
  if (payload.event === 'installing') {
    return { ...loggedSong, detail: 'Saving…', progress: Math.max(song.progress, 99.5) };
  }
  if (payload.event === 'finalizing') {
    return { ...loggedSong, detail: payload.detail || 'Finalizing stems…', progress: Math.max(song.progress, 95) };
  }
  if (payload.event === 'preparing_pass') {
    return { ...loggedSong, detail: payload.detail || 'Preparing next pass…' };
  }
  if (payload.event === 'skipped') {
    const message = skipReason(payload.reason);
    return { ...loggedSong, message, detail: message };
  }
  const descriptions: Record<string, string> = {
    runtime_setup: 'Preparing audio engine…',
    checking_models: 'Checking models…',
    models_ready: 'Preparing…',
    decoded: 'Preparing…',
    validated: 'Checking…',
    recovered: 'Recovery completed',
    recovery_required: 'Interrupted · will restart clean',
    cancelled: 'Cancelled · rollback finished',
    completed: 'Finishing…',
  };
  const destination = payload.destination ?? song.destination;
  return { ...loggedSong, destination, detail: payload.message || descriptions[payload.event] || song.detail, chunkSize };
}

export function mergePreviews(songs: Song[], previews: Preview[]): Song[] {
  const existing = new Set(songs.map(song => song.folder));
  const added = previews.filter(preview => !existing.has(preview.folder)).map(preview => {
    existing.add(preview.folder);
    return { ...preview, id: crypto.randomUUID(), detail: preview.message, progress: 0, log: '', canRetry: false, startedAt: 0, finishedAt: 0, inferenceAt: 0 };
  });
  return [...songs, ...added];
}

export function refreshSong(song: Song, preview: Preview): Song {
  return { ...song, source: preview.source, backup: preview.backup,
    status: preview.status === 'recovery_required' ? 'recovery_required' : song.status,
    detail: preview.status === 'recovery_required' ? preview.message : song.detail,
    canRetry: preview.status === 'ready' };
}

export function duration(milliseconds: number): string {
  const seconds = Math.floor(milliseconds / 1000);
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor(seconds / 60) % 60;
  return `${hours > 0 ? `${hours}:${String(minutes).padStart(2, '0')}` : minutes}:${String(seconds % 60).padStart(2, '0')}`;
}

export function timing(song: Song, now: number): string {
  const elapsed = (song.status === 'running' ? now : song.finishedAt) - song.startedAt;
  const label = duration(elapsed);
  if (song.status !== 'running') { return label; }
  if (song.progress === 0) { return `${label} / —`; }
  if (song.progress >= 95) {
    const remainingSeconds = Math.max(1, Math.round((100 - song.progress) * 3));
    const formatted = remainingSeconds >= 60 ? duration(remainingSeconds * 1000) : `${remainingSeconds}s`;
    return `${label} · ~${formatted} left`;
  }
  const remaining = (now - song.inferenceAt) * (95 - song.progress) / song.progress;
  return `${label} / ${duration(elapsed + remaining)}`;
}

export const ROW_HEIGHT = 52;

export function rowWindow(total: number, scrollTop: number, height: number) {
  const start = Math.min(total, Math.max(0, Math.floor(scrollTop / ROW_HEIGHT) - 5));
  const end = Math.min(total, start + Math.ceil(height / ROW_HEIGHT) + 10);
  return { start, end, top: start * ROW_HEIGHT, bottom: (total - end) * ROW_HEIGHT };
}

export interface ScrollbarGeometry {
  isScrollable: boolean;
  thumbHeight: number;
  thumbTop: number;
}

export function scrollbarGeometry(totalRows: number, scrollTop: number, viewportHeight: number): ScrollbarGeometry {
  const totalHeight = totalRows * ROW_HEIGHT;
  if (viewportHeight <= 0 || totalHeight <= viewportHeight) {
    return { isScrollable: false, thumbHeight: 0, thumbTop: 0 };
  }
  const maxScroll = totalHeight - viewportHeight;
  const thumbHeight = Math.max(28, (viewportHeight / totalHeight) * viewportHeight);
  const maxThumbTop = viewportHeight - thumbHeight;
  const clampedScroll = Math.min(maxScroll, Math.max(0, scrollTop));
  const thumbTop = maxScroll > 0 ? (clampedScroll / maxScroll) * maxThumbTop : 0;
  return { isScrollable: true, thumbHeight, thumbTop };
}

export function scrollPositionFromThumb(thumbTop: number, totalRows: number, viewportHeight: number, thumbHeight: number): number {
  const totalHeight = totalRows * ROW_HEIGHT;
  const maxScroll = totalHeight - viewportHeight;
  const maxThumbTop = viewportHeight - thumbHeight;
  if (maxThumbTop <= 0 || maxScroll <= 0) {
    return 0;
  }
  const clampedThumb = Math.min(maxThumbTop, Math.max(0, thumbTop));
  return (clampedThumb / maxThumbTop) * maxScroll;
}

export interface QueueProgress {
  completed: number;
  remaining: number;
  failed: number;
  skipped: number;
  total: number;
  percent: number;
}

export function queueProgress(songs: Song[]): QueueProgress {
  let completed = 0;
  let remaining = 0;
  let failed = 0;
  let skipped = 0;
  let activeFraction = 0;

  for (const song of songs) {
    if (song.status === 'completed') {
      completed++;
    } else if (song.status === 'running') {
      remaining++;
      activeFraction = song.progress / 100;
    } else if (['ready', 'recovery_required', 'queued'].includes(song.status)) {
      remaining++;
    } else if (['error', 'cancelled'].includes(song.status)) {
      failed++;
    } else if (song.status === 'skipped') {
      skipped++;
    }
  }

  const target = completed + remaining + failed;
  const percent = target > 0 ? Math.min(100, Math.max(0, ((completed + activeFraction) / target) * 100)) : 0;

  return { completed, remaining, failed, skipped, total: songs.length, percent };
}

function splitPathParts(p: string): { isAbsolute: boolean; parts: string[] } {
  const normalized = p.replace(/\\/g, '/');
  const isAbsolute = normalized.startsWith('/');
  const parts = normalized.split('/').filter(Boolean);
  return { isAbsolute, parts };
}

function findCommonDirectory(paths: string[]): string | null {
  if (paths.length === 0) {
    return null;
  }
  const split = paths.map(splitPathParts);
  const first = split[0];
  let commonLen = 0;
  for (let i = 0; i < first.parts.length; i++) {
    const part = first.parts[i];
    if (split.every(s => s.parts[i] === part)) {
      commonLen++;
    } else {
      break;
    }
  }
  if (commonLen === 0) {
    return null;
  }
  const commonParts = first.parts.slice(0, commonLen);
  const sep = paths[0].includes('\\') ? '\\' : '/';
  const prefix = first.isAbsolute ? (paths[0].startsWith('\\\\') ? '\\\\' : '/') : '';
  return prefix + commonParts.join(sep);
}

export function resolveExportTargets(folders: string[], exportRoot: string): Map<string, string> {
  const result = new Map<string, string>();
  if (folders.length === 0) {
    return result;
  }
  const sep = exportRoot.includes('\\') || folders[0].includes('\\') ? '\\' : '/';
  const cleanExportRoot = exportRoot.replace(/[/\\]+$/, '');

  if (folders.length === 1) {
    const folder = folders[0];
    const name = (folder.replace(/[/\\]+$/, '').split(/[/\\]/).pop() || 'song').replace(/\.(con|live|pirs|opus|ogg|mp3|wav|flac|aiff|m4a|aac)$/i, '');
    result.set(folder, `${cleanExportRoot}${sep}${name}`);
    return result;
  }

  const ancestor = findCommonDirectory(folders);
  const used = new Map<string, number>();

  for (const folder of folders) {
    const cleanFolder = folder.replace(/[/\\]+$/, '');
    let relative = '';
    if (ancestor && cleanFolder.length > ancestor.length && cleanFolder.startsWith(ancestor)) {
      relative = cleanFolder.slice(ancestor.length).replace(/^[/\\]+/, '');
    }
    if (!relative) {
      relative = cleanFolder.split(/[/\\]/).pop() || 'song';
    }
    relative = relative.replace(/\.(con|live|pirs|opus|ogg|mp3|wav|flac|aiff|m4a|aac)$/i, '');
    const relativeNormalized = relative.replace(/\//g, sep);
    let target = `${cleanExportRoot}${sep}${relativeNormalized}`;

    const count = used.get(target) ?? 0;
    used.set(target, count + 1);
    if (count > 0) {
      target = `${target} (${count})`;
    }
    result.set(folder, target);
  }
  return result;
}
