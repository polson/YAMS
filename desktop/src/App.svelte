<script lang="ts">
  import { onMount } from 'svelte';
  import { Channel, invoke, isTauri } from '@tauri-apps/api/core';
  import { BundleType, getBundleType } from '@tauri-apps/api/app';
  import { listen } from '@tauri-apps/api/event';
  import { getCurrentWebview } from '@tauri-apps/api/webview';
  import { getCurrentWindow } from '@tauri-apps/api/window';
  import { open } from '@tauri-apps/plugin-dialog';
  import { relaunch } from '@tauri-apps/plugin-process';
  import { check } from '@tauri-apps/plugin-updater';
  import { mergePreviews, queueProgress, refreshSong, resolveExportTargets, rowWindow, ROW_HEIGHT, timing, updateSong, scrollbarGeometry, scrollPositionFromThumb, type BatchEvent, type Preview, type Settings, type Song } from './queue';
  import logo from './assets/logo.png';

  const native = isTauri();
  let settings = $state<Settings>({ device: 'cpu', modelProfile: 'mega-8stem', exportFolder: '' });
  let songs = $state<Song[]>([]);
  let running = $state(false);
  let pausing = $state(false);
  let paused = $state(false);
  let confirmingClear = $state(false);
  let clearTimer: ReturnType<typeof setTimeout> | undefined;
  let inspecting = $state(false);
  let stopping = $state(false);
  let closing = $state(false);
  let dragging = $state(false);
  let initialized = $state(false);
  let engineReady = $state(false);
  let preparing = $state(false);
  let notice = $state('');
  let error = $state('');
  let scrollTop = $state(0);
  let viewportHeight = $state(0);
  let viewport = $state<HTMLDivElement>();
  let queuePanel = $state<HTMLElement>();
  let queueHeading = $state<HTMLElement>();
  let queueFooter = $state<HTMLElement>();
  let selectedId = $state('');
  let detailsHeight = $state(190);
  let resizingDetails = $state(false);
  let resizeStartY = 0;
  let resizeStartHeight = 190;
  let clock = $state(performance.now());
  const positions = $derived(new Map(songs.map((song, index) => [song.id, index])));
  const visible = $derived(rowWindow(songs.length, scrollTop, viewportHeight));
  const visibleSongs = $derived(songs.slice(visible.start, visible.end));
  const selectedSong = $derived(songs.find(song => song.id === selectedId));
  const activeIndex = $derived(songs.findIndex(song => song.status === 'running'));
  const scrollbar = $derived(scrollbarGeometry(songs.length, scrollTop, viewportHeight));
  let isDraggingThumb = $state(false);
  let dragStartY = 0;
  let dragStartScrollTop = 0;

  $effect(() => {
    const maxScroll = Math.max(0, songs.length * ROW_HEIGHT - viewportHeight);
    if (scrollTop > maxScroll && viewport) {
      viewport.scrollTop = maxScroll;
      scrollTop = maxScroll;
    }
  });

  const busy = $derived(running || inspecting || preparing);
  const ready = $derived(songs.filter(song => ['ready', 'recovery_required'].includes(song.status)).length);
  const queueStats = $derived(queueProgress(songs));
  const queueProgressPercent = $derived(
    queueStats.percent === 0 || queueStats.percent === 100
      ? `${Math.round(queueStats.percent)}%`
      : `${queueStats.percent.toFixed(1)}%`
  );

  onMount(() => {
    const timer = setInterval(() => { clock = performance.now(); }, 1000);
    let disposed = false;
    const cleanups: (() => void)[] = [];
    async function initialize() {
      if (!native) {
        initialized = true;
        notice = 'Browser preview. Launch the desktop app to select and process folders.';
        return;
      }
      try {
        try {
          if (await updateOnLaunch()) {
            return;
          }
        } catch (problem) {
          console.warn('Could not apply a YAMS update on launch:', problem);
          notice = '';
        }
        const defaults = await invoke<Settings>('default_settings');
        const availableDevices = defaults.availableDevices ?? ['cpu'];
        const availableModelProfiles = defaults.availableModelProfiles ?? ['mega-8stem'];
        const bestDevice = defaults.device || availableDevices[0] || 'cpu';
        settings = { ...defaults, device: bestDevice, availableDevices, availableModelProfiles, exportFolder: '' };
        const saved = localStorage.getItem('yams-settings') || localStorage.getItem('yass-settings');
        if (saved) {
          try {
            const value = JSON.parse(saved);
            const modelProfile = typeof value.modelProfile === 'string' && availableModelProfiles.includes(value.modelProfile)
              ? value.modelProfile
              : defaults.modelProfile ?? 'mega-8stem';
            const exportFolder = typeof value.exportFolder === 'string' ? value.exportFolder : '';
            settings = {
              device: bestDevice,
              availableDevices,
              modelProfile,
              availableModelProfiles,
              exportFolder,
            };
            saveSettings();
          } catch (problem) {
            error = String(problem);
          }
        }
        const drop = await getCurrentWebview().onDragDropEvent(event => {
          dragging = event.payload.type === 'over' || event.payload.type === 'enter';
          if (event.payload.type === 'drop') {
            dragging = false;
            void addPaths(event.payload.paths);
          }
        });
        if (disposed) { drop(); return; }
        cleanups.push(drop);
        const close = await listen('closing_requested', () => {
          closing = true;
          stopping = true;
        });
        if (disposed) { close(); return; }
        cleanups.push(close);
        await prepareEngine();
      } catch (problem) {
        error = String(problem);
      } finally {
        initialized = true;
      }
    }
    void initialize();
    return () => { clearInterval(timer); disposed = true; cleanups.forEach(cleanup => cleanup()); };
  });

  async function updateOnLaunch(): Promise<boolean> {
    if (await getBundleType() !== BundleType.AppImage) {
      return false;
    }
    const update = await check({ timeout: 5000 });
    if (!update) {
      return false;
    }
    let downloaded = 0;
    notice = `Downloading YAMS ${update.version}…`;
    await update.downloadAndInstall(event => {
      if (event.event === 'Progress') {
        downloaded += event.data.chunkLength;
        notice = `Downloading YAMS ${update.version}… ${Math.floor(downloaded / 1048576)} MiB`;
      }
    });
    await relaunch();
    return true;
  }

  function clearQueue() {
    if (busy) { return; }
    if (!confirmingClear) {
      confirmingClear = true;
      clearTimeout(clearTimer);
      clearTimer = setTimeout(() => { confirmingClear = false; }, 3000);
      return;
    }
    clearTimeout(clearTimer);
    confirmingClear = false;
    songs = [];
    scrollTop = 0;
    selectedId = '';
    copied = false;
    notice = '';
    paused = false;
  }

  function removeSong(id: string) {
    if (busy) { return; }
    songs = songs.filter(song => song.id !== id);
    if (selectedId === id) {
      selectedId = '';
      copied = false;
    }
  }

  async function prepareEngine() {
    preparing = true;
    if (initialized) { error = ''; }
    notice = 'Checking audio engine…';
    const events = new Channel<string>();
    events.onmessage = message => { notice = message; };
    try {
      await invoke('prepare_engine', { events });
      engineReady = true;
      notice = '';
    } catch (problem) {
      error = String(problem);
      notice = '';
    } finally { preparing = false; }
  }

  async function openFolder(folder: string) {
    try { await invoke('open_folder', { folder }); }
    catch (problem) { error = String(problem); }
  }

  async function controlWindow(action: 'minimize' | 'toggleMaximize' | 'close') {
    try { await getCurrentWindow()[action](); }
    catch (problem) { error = String(problem); }
  }

  async function chooseFolders() {
    try {
      const paths = await open({ directory: true, multiple: true, title: 'Add song folders or a song collection' });
      if (paths) { await addPaths(Array.isArray(paths) ? paths : [paths]); }
    } catch (problem) { error = String(problem); }
  }

  async function chooseFiles() {
    try {
      const paths = await open({
        directory: false,
        multiple: true,
        title: 'Add audio files or CON packages',
        filters: [
          { name: 'Audio files & CON packages', extensions: ['opus', 'ogg', 'mp3', 'wav', 'flac', 'aiff', 'm4a', 'aac', 'con', 'live', 'pirs'] },
          { name: 'All files', extensions: ['*'] }
        ]
      });
      if (paths) { await addPaths(Array.isArray(paths) ? paths : [paths]); }
    } catch (problem) { error = String(problem); }
  }

  async function addPaths(paths: string[]) {
    if (busy || !initialized || !engineReady) { notice = 'Wait for the audio engine before adding songs.'; return; }
    inspecting = true;
    error = '';
    notice = '';
    try {
      const previews = await invoke<Preview[]>('inspect_folders', { settings: $state.snapshot(settings), paths });
      songs = mergePreviews(songs, previews);
      if (previews.length === 0) {
        const target = paths.length === 1
          ? (paths[0].split(/[/\\]/).filter(Boolean).pop() || paths[0])
          : (paths.length > 1 ? `${paths.length} folders` : 'the selected location');
        notice = `⚠️ No songs found in ${target}.`;
      } else {
        notice = '';
      }
    } catch (problem) { error = String(problem); notice = ''; }
    finally { inspecting = false; }
  }

  let copied = $state(false);
  let copyTimer: ReturnType<typeof setTimeout> | undefined;

  function toggleDetails(id: string) {
    selectedId = selectedId === id ? '' : id;
    copied = false;
  }

  function hasSongInfo(song: Song): boolean {
    return Boolean(song.source || song.message || song.detail || song.log || song.destination || song.chunkSize !== undefined);
  }

  function onSongRowClick(event: MouseEvent, song: Song) {
    if ((event.target as Element).closest('button') || !hasSongInfo(song)) {
      return;
    }
    toggleDetails(song.id);
  }

  function onSongRowKeydown(event: KeyboardEvent, song: Song) {
    if (event.target !== event.currentTarget || (event.key !== 'Enter' && event.key !== ' ')) {
      return;
    }
    event.preventDefault();
    if (hasSongInfo(song)) {
      toggleDetails(song.id);
    }
  }

  function closeDetails() {
    selectedId = '';
    copied = false;
  }

  function maximumDetailsHeight(): number {
    return Math.max(100, queuePanel!.clientHeight - queueHeading!.clientHeight - queueFooter!.clientHeight - ROW_HEIGHT * 2);
  }

  function startDetailsResize(event: PointerEvent) {
    if (event.button !== 0) {
      return;
    }
    event.preventDefault();
    resizingDetails = true;
    resizeStartY = event.clientY;
    resizeStartHeight = detailsHeight;
    (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
  }

  function resizeDetails(event: PointerEvent) {
    if (!resizingDetails) {
      return;
    }
    const nextHeight = resizeStartHeight + resizeStartY - event.clientY;
    detailsHeight = Math.min(maximumDetailsHeight(), Math.max(100, nextHeight));
  }

  function stopDetailsResize() {
    resizingDetails = false;
  }

  function resizeDetailsWithKeyboard(event: KeyboardEvent) {
    if (event.key !== 'ArrowUp' && event.key !== 'ArrowDown') {
      return;
    }
    event.preventDefault();
    const delta = event.key === 'ArrowUp' ? 24 : -24;
    detailsHeight = Math.min(maximumDetailsHeight(), Math.max(100, detailsHeight + delta));
  }

  function formatChunkSize(samples: number): string {
    return `${samples.toLocaleString()} samples (${(samples / 44100).toFixed(1)} s at 44.1 kHz)`;
  }

  function getDetailsText(song: Song): string {
    const sep = song.folder.includes('\\') ? '\\' : '/';
    const audioPath = song.source ? `${song.folder}${sep}${song.source}` : '';
    const lines = [
      audioPath ? `File:    ${audioPath}` : `Folder:  ${song.folder}`,
      `Status:  ${song.status}`,
      `Device:  ${settings.device}`,
    ];
    if (song.chunkSize !== undefined) {
      lines.push(`Chunk size: ${formatChunkSize(song.chunkSize)}`);
    }
    const message = song.message || song.detail;
    if (message) {
      lines.push(`Detail:  ${message}`);
    }
    if (song.log) {
      lines.push('', '--- Worker Log ---', song.log.trim());
    }
    return lines.join('\n');
  }

  async function copyDetails() {
    if (!selectedSong) {
      return;
    }
    await navigator.clipboard.writeText(getDetailsText(selectedSong));
    copied = true;
    clearTimeout(copyTimer);
    copyTimer = setTimeout(() => {
      copied = false;
    }, 1800);
  }

  function saveSettings() {
    localStorage.setItem('yams-settings', JSON.stringify({
      device: settings.device,
      modelProfile: settings.modelProfile,
      exportFolder: settings.exportFolder,
    }));
  }

  async function chooseExportFolder() {
    if (!native || busy) { return; }
    try {
      const selected = await open({
        directory: true,
        multiple: false,
        title: 'Choose export folder',
        defaultPath: settings.exportFolder || undefined,
      });
      if (typeof selected === 'string' && selected.length > 0) {
        settings.exportFolder = selected;
        saveSettings();
      }
    } catch (problem) {
      error = String(problem);
    }
  }

  async function run(onlyId?: string) {
    const selected = songs.filter(song => ['ready', 'recovery_required'].includes(song.status) && (!onlyId || song.id === onlyId));
    if (busy || !engineReady || selected.length === 0) {
      return;
    }
    if (!settings.exportFolder) {
      error = 'Choose a destination folder before starting export.';
      return;
    }
    const ids = new Set(selected.map(song => song.id));
    const started = new Set<string>();
    const batchSettings = $state.snapshot(settings);
    const exportTargets = resolveExportTargets(selected.map(song => song.folder), settings.exportFolder);
    for (const song of selected) {
      const dest = exportTargets.get(song.folder);
      if (dest) {
        song.destination = dest;
      }
    }
    songs = songs.map(song => ids.has(song.id) ? { ...song, status: 'queued', detail: 'Queued' } : song);
    running = true;
    pausing = false;
    paused = false;
    stopping = false;
    error = '';
    notice = '';
    const events = new Channel<BatchEvent>();
    events.onmessage = event => {
      if (event.kind === 'batch_finished') {
        const wasPaused = paused;
        stopping = false;
        pausing = false;
        notice = '';
        if (closing) {
          running = false;
          void getCurrentWindow().close().catch(problem => { error = String(problem); });
          return;
        }
        if (wasPaused) {
          running = false;
          songs = songs.map(song => {
            if (song.status === 'cancelled') {
              return { ...song, status: 'ready', detail: '', message: '', progress: 0, canRetry: false };
            }
            return song;
          });
          return;
        }
        paused = false;
        void refreshFinished(started, batchSettings);
        return;
      }
      const now = performance.now();
      if (event.kind === 'started' || (event.kind === 'worker' && event.payload.event === 'model_started')) {
        clock = now;
      }
      if (event.kind === 'started') {
        started.add(event.id);
      }
      const index = positions.get(event.id)!;
      Object.assign(songs[index], updateSong(songs[index], event, now));
    };
    try {
      await invoke('start_batch', {
        settings: $state.snapshot(settings),
        jobs: selected.map(song => ({
          id: song.id,
          folder: song.folder,
          output_folder: exportTargets.get(song.folder) || null,
          recover: false,
        })),
        events,
      });
    } catch (problem) {
      songs = songs.map(song => ids.has(song.id) && song.status === 'queued' ? { ...song, status: 'ready', detail: '' } : song);
      running = false;
      pausing = false;
      paused = false;
      error = String(problem);
    }
  }

  async function pauseQueue() {
    if (!running || pausing) {
      return;
    }
    pausing = true;
    paused = true;
    try {
      await invoke('cancel_batch');
    } catch (problem) {
      pausing = false;
      error = String(problem);
    }
  }


  async function refreshFinished(started: Set<string>, batchSettings: Settings) {
    const candidates = songs.filter(song => started.has(song.id) && ['cancelled', 'error', 'recovered'].includes(song.status));
    inspecting = true;
    running = false;
    try {
      if (candidates.length > 0) {
        const previews = await invoke<Preview[]>('inspect_folders', { settings: batchSettings, paths: candidates.map(song => song.folder) });
        const byFolder = new Map(previews.map(preview => [preview.folder, preview]));
        for (const song of candidates) {
          const preview = byFolder.get(song.folder);
          if (preview) {
            Object.assign(song, refreshSong(song, preview));
          }
        }
      }
    } catch (problem) {
      error = String(problem);
    } finally {
      inspecting = false;
    }
  }

  async function retry(id: string) {
    inspecting = true;
    error = '';
    const song = songs[positions.get(id)!];
    try {
      const previews = await invoke<Preview[]>('inspect_folders', { settings: $state.snapshot(settings), paths: [song.folder] });
      const preview = previews[0];
      if (!preview || !['ready', 'recovery_required'].includes(preview.status)) {
        song.canRetry = false;
        song.detail = preview?.message || 'Folder is no longer ready';
        song.status = preview?.status || 'error';
        return;
      }
      Object.assign(song, preview, { canRetry: false });
      inspecting = false;
      void run(id);
    } catch (problem) {
      error = String(problem);
    } finally {
      inspecting = false;
    }
  }

  function startScrollDrag(event: PointerEvent) {
    if (event.button !== 0) {
      return;
    }
    event.stopPropagation();
    isDraggingThumb = true;
    dragStartY = event.clientY;
    dragStartScrollTop = scrollTop;
    (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
  }

  function onScrollDrag(event: PointerEvent) {
    if (!isDraggingThumb || !viewport) {
      return;
    }
    const deltaY = event.clientY - dragStartY;
    const maxScroll = songs.length * ROW_HEIGHT - viewportHeight;
    const maxThumbTop = viewportHeight - scrollbar.thumbHeight;
    if (maxThumbTop > 0 && maxScroll > 0) {
      const nextScrollTop = Math.min(maxScroll, Math.max(0, dragStartScrollTop + (deltaY / maxThumbTop) * maxScroll));
      viewport.scrollTop = nextScrollTop;
      scrollTop = nextScrollTop;
    }
  }

  function endScrollDrag(event: PointerEvent) {
    if (isDraggingThumb) {
      isDraggingThumb = false;
      try {
        (event.currentTarget as HTMLElement).releasePointerCapture(event.pointerId);
      } catch {}
    }
  }

  function onScrollTrackClick(event: PointerEvent) {
    if (event.button !== 0 || !viewport) {
      return;
    }
    const trackRect = (event.currentTarget as HTMLElement).getBoundingClientRect();
    const clickY = event.clientY - trackRect.top;
    const targetThumbTop = clickY - scrollbar.thumbHeight / 2;
    const nextScrollTop = scrollPositionFromThumb(targetThumbTop, songs.length, viewportHeight, scrollbar.thumbHeight);
    viewport.scrollTop = nextScrollTop;
    scrollTop = nextScrollTop;
  }
</script>

<svelte:head><title>YAMS · Music separator</title></svelte:head>

{#if native}
  <header class="titlebar" aria-label="Window controls">
    <div class="titlebar-drag" data-tauri-drag-region aria-hidden="true"></div>
    <button class="window-button" aria-label="Minimize window" title="Minimize" onclick={() => controlWindow('minimize')}>
      <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3 11h10" /></svg>
    </button>
    <button class="window-button" aria-label="Maximize or restore window" title="Maximize / restore" onclick={() => controlWindow('toggleMaximize')}>
      <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3 3h10v10H3Z" /></svg>
    </button>
    <button class="window-button window-close" aria-label="Close window" title="Close" onclick={() => controlWindow('close')} disabled={closing}>
      <svg viewBox="0 0 16 16" aria-hidden="true"><path d="m3 3 10 10M13 3 3 13" /></svg>
    </button>
  </header>
{/if}

<div class="shell" class:dragging class:native>
  {#if dragging}
    <div class="drag-overlay" aria-hidden="true">
      <div class="drag-cue">
        <strong>Drop songs, audio, or CON files anywhere</strong>
        <span>Release to add songs to the queue</span>
      </div>
    </div>
  {/if}
  <header class="app-header">
    <div class="brand">
      <img class="brand-logo" src={logo} alt="YAMS" title="Yet Another Music Separator" />
    </div>
    {#if songs.length > 0}
      <div class="global-progress-panel">
        <div class="global-progress-header">
          <div class="global-progress-counts">
            <span class="count-completed">{queueStats.completed.toLocaleString()} completed</span>
            <span class="meta-sep">·</span>
            <span class="count-remaining">{queueStats.remaining.toLocaleString()} remaining</span>
            {#if queueStats.failed > 0}
              <span class="meta-sep">·</span>
              <span class="count-failed">{queueStats.failed.toLocaleString()} failed</span>
            {/if}
            {#if queueStats.skipped > 0}
              <span class="meta-sep">·</span>
              <span class="count-skipped">{queueStats.skipped.toLocaleString()} skipped</span>
            {/if}
            {#if paused && ready > 0}
              <span class="meta-sep">·</span>
              <span class="count-paused" style="color: #ffc18d; font-weight: 500;">Queue paused</span>
            {:else if pausing}
              <span class="meta-sep">·</span>
              <span class="count-pausing" style="color: #d1b58a; font-weight: 500;">Pausing…</span>
            {/if}
          </div>
          <span class="global-progress-pct">{queueProgressPercent}</span>
        </div>
        <div
          class="global-progress-bar"
          role="progressbar"
          aria-label="Overall queue progress"
          aria-valuenow={Math.round(queueStats.percent)}
          aria-valuemin="0"
          aria-valuemax="100"
        >
          <div class="global-progress-fill" style:width={`${queueStats.percent}%`}></div>
        </div>
      </div>
    {/if}
  </header>
  <main>
    {#if error}<div class="message error" role="alert">{error}</div>{/if}
    {#if notice}<div class="message notice" role="status">{notice}</div>{/if}
    {#if native && initialized && !engineReady && !preparing}<div class="message"><button onclick={prepareEngine}>Retry setup</button></div>{/if}

    <section class="queue panel" aria-label="Queue" bind:this={queuePanel}>
      <div class="section-heading" bind:this={queueHeading}>
        <div>
          <h2>Queue</h2>
          {#if songs.length > 0}<span>{songs.length.toLocaleString()} {songs.length === 1 ? 'song' : 'songs'}</span>{/if}
        </div>
        <div class="actions">
          {#if songs.length > 0}
            <button class="quiet" onclick={chooseFolders} disabled={!native || busy || !initialized || !engineReady}>Add folder</button>
            <button class="quiet" onclick={chooseFiles} disabled={!native || busy || !initialized || !engineReady}>Add file</button>
          {/if}
          <button class={confirmingClear ? 'danger' : 'quiet'} onclick={clearQueue} disabled={busy || songs.length === 0}>{confirmingClear ? 'Confirm clear?' : 'Clear'}</button>
        </div>
      </div>
      {#if songs.length === 0}
        <div class="empty drop-zone">
          <h2>{inspecting ? 'Checking songs…' : 'Drop songs here'}</h2>
          {#if inspecting}
            <p>Reading chart and audio metadata…</p>
          {:else}
            <p class="supported-types">Supports unpacked INI songs, audio files (.opus, .ogg, .mp3, .wav, .flac), CON packages, and exCON folders</p>
          {/if}
          <div class="actions">
            <button class="primary" onclick={chooseFolders} disabled={!native || busy || !initialized || !engineReady}>Add folder</button>
            <button class="primary" onclick={chooseFiles} disabled={!native || busy || !initialized || !engineReady}>Add file</button>
          </div>
        </div>
      {:else}
        <div class="song-list-wrapper">
          <div class="song-list" class:has-scrollbar={scrollbar.isScrollable} bind:this={viewport} bind:clientHeight={viewportHeight} onscroll={() => { scrollTop = viewport!.scrollTop; }} aria-label="Songs" role="grid" aria-rowcount={songs.length} aria-colcount="4" tabindex="0">
            <div style:height={`${visible.top}px`} aria-hidden="true"></div>
            {#each visibleSongs as song, index (song.id)}
              <div class="song" class:selected={selectedId === song.id} role="row" aria-rowindex={visible.start + index + 1} style:height={`${ROW_HEIGHT}px`} tabindex="0" onclick={event => onSongRowClick(event, song)} onkeydown={event => onSongRowKeydown(event, song)}>
                <div class="song-name" role="gridcell"><h3 title={song.folder}>{song.folder.split(/[\\/]/).filter(Boolean).at(-1)}</h3></div>
                <div class="song-state" role="gridcell"><span class="badge" title={song.message || song.detail} data-status={song.status}>{song.status === 'running' ? song.detail : (song.status === 'recovery_required' ? 'interrupted' : song.status.replaceAll('_', ' '))}</span>
                  {#if song.status === 'running'}<div class="progress" role="progressbar" aria-label={`Progress for ${song.folder}`} aria-valuemin="0" aria-valuemax="100" aria-valuenow={Math.round(song.progress)}><div class="progress-fill" style:width={`${song.progress}%`}></div></div>{/if}
                </div>
                <span class="timing" role="gridcell">{#if song.status === 'running' || song.status === 'completed'}{timing(song, clock)}{/if}</span>
                <div class="row-actions" role="gridcell">
                  {#if song.canRetry || song.status === 'recovery_required'}<button class="quiet" onclick={() => retry(song.id)} disabled={busy || !native}>Retry</button>{/if}
                  <button class="quiet icon-button open-folder" aria-label="Open folder" title="Open folder" onclick={() => openFolder(song.status === 'completed' && song.destination ? song.destination : song.folder)} disabled={!native}>
                    <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 7V5a1 1 0 0 1 1-1h5l2 3h9a1 1 0 0 1 1 1v11a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V7Z" /></svg>
                  </button>
                  <button class="quiet icon-button remove" aria-label="Remove from list" title="Remove from list" onclick={() => removeSong(song.id)} disabled={busy}>
                    <svg viewBox="0 0 24 24" aria-hidden="true"><path d="m6 6 12 12M18 6 6 18" /></svg>
                  </button>
                </div>
              </div>
            {/each}
            <div style:height={`${visible.bottom}px`} aria-hidden="true"></div>
          </div>
          {#if scrollbar.isScrollable}
            <div
              class="custom-scrollbar"
              role="scrollbar"
              aria-controls="song-list"
              aria-orientation="vertical"
              aria-valuenow={Math.round(scrollTop)}
              aria-valuemin="0"
              aria-valuemax={Math.round(songs.length * ROW_HEIGHT - viewportHeight)}
              tabindex="-1"
              onpointerdown={onScrollTrackClick}
            >
              <div
                class="scrollbar-thumb"
                role="presentation"
                class:dragging={isDraggingThumb}
                style:height={`${scrollbar.thumbHeight}px`}
                style:top={`${scrollbar.thumbTop}px`}
                onpointerdown={startScrollDrag}
                onpointermove={onScrollDrag}
                onpointerup={endScrollDrag}
                onpointercancel={endScrollDrag}
              ></div>
            </div>
          {/if}
        </div>
      {/if}
      {#if selectedSong}
        <div class="song-details" class:resizing={resizingDetails} data-status={selectedSong.status} style:height={`${detailsHeight}px`}>
          <div
            class="details-resize-handle"
            role="slider"
            aria-label="Resize information panel"
            aria-orientation="vertical"
            aria-valuenow={detailsHeight}
            aria-valuemin="100"
            aria-valuemax={maximumDetailsHeight()}
            tabindex="0"
            onpointerdown={startDetailsResize}
            onpointermove={resizeDetails}
            onpointerup={stopDetailsResize}
            onpointercancel={stopDetailsResize}
            onkeydown={resizeDetailsWithKeyboard}
          ></div>
          <div class="song-details-header">
            <span class="terminal-title" title={selectedSong.folder}>
              <span class="terminal-dot"></span>
              <span class="terminal-folder">{selectedSong.folder.split(/[\\/]/).at(-1)}</span>
            </span>
            <div class="details-header-actions">
              <button class="copy-button" class:copied onclick={copyDetails} title={copied ? 'Copied to clipboard' : 'Copy details'}>
                {#if copied}
                  <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <polyline points="20 6 9 17 4 12"></polyline>
                  </svg>
                  <span>Copied</span>
                {:else}
                  <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                  </svg>
                  <span>Copy</span>
                {/if}
              </button>
              <button class="copy-button close-details-button" aria-label="Close information panel" title="Close" onclick={closeDetails}>
                <svg viewBox="0 0 24 24" aria-hidden="true"><path d="m6 6 12 12M18 6 6 18" /></svg>
              </button>
            </div>
          </div>
          <div class="terminal-body">
            <div class="terminal-meta">
              {#if selectedSong.source}
                <div class="meta-row">
                  <span class="meta-label">File:</span>
                  <span class="meta-val">{selectedSong.folder}{selectedSong.folder.includes('\\') ? '\\' : '/'}{selectedSong.source}</span>
                </div>
              {:else}
                <div class="meta-row">
                  <span class="meta-label">Folder:</span>
                  <span class="meta-val">{selectedSong.folder}</span>
                </div>
              {/if}
              <div class="meta-row">
                <span class="meta-label">Status:</span>
                <span class="meta-val">{selectedSong.status}</span>
                <span class="meta-sep">·</span>
                <span class="meta-label">Device:</span>
                <span class="meta-val">{settings.device}</span>
                {#if settings.modelProfile === 'mlx-6stem' || settings.modelProfile === 'torch-6stem' || settings.modelProfile === 'onnx-6stem'}
                  <span class="meta-sep">·</span>
                  <span class="meta-val">6-stem</span>
                {/if}
              </div>
              {#if selectedSong.chunkSize !== undefined}
                <div class="meta-row">
                  <span class="meta-label">Chunk size:</span>
                  <span class="meta-val">{formatChunkSize(selectedSong.chunkSize)}</span>
                </div>
              {/if}
              {#if selectedSong.destination}
                <div class="meta-row">
                  <span class="meta-label">Export:</span>
                  <span class="meta-val">{selectedSong.destination}</span>
                </div>
              {/if}
            </div>
            {#if selectedSong.message || selectedSong.detail}
              <div class="terminal-line">{selectedSong.message || selectedSong.detail}</div>
            {/if}
            {#if selectedSong.log}
              <pre class="terminal-log">{selectedSong.log}</pre>
            {/if}
          </div>
        </div>
      {/if}
      <div class="queue-footer" bind:this={queueFooter}>
        <div class="queue-options">
          {#if activeIndex >= 0}<button class="quiet show-active" onclick={() => viewport!.scrollTo({ top: activeIndex * ROW_HEIGHT })}>Show active</button>{/if}
          {#if (settings.availableModelProfiles?.length ?? 1) > 1}
            {@const sixStemProfile = settings.availableModelProfiles?.find(p => p !== 'mega-8stem') ?? 'mlx-6stem'}
            <div class="mode-select" role="group" aria-label="Model selection">
              <div class="mode-pill-wrapper">
                <button
                  type="button"
                  class="mode-pill"
                  class:active={settings.modelProfile === sixStemProfile}
                  disabled={busy}
                  onclick={() => {
                    settings.modelProfile = sixStemProfile;
                    saveSettings();
                  }}
                >6 stems</button>
                <div class="mode-tooltip" role="tooltip">
                  <strong>6 stems</strong>
                  <span>Separates vocals, drums, bass, guitar, piano, and other.</span>
                  <span>Faster separation with all drums on one track.</span>
                </div>
              </div>
              <div class="mode-pill-wrapper">
                <button
                  type="button"
                  class="mode-pill"
                  class:active={settings.modelProfile === 'mega-8stem'}
                  disabled={busy}
                  onclick={() => { settings.modelProfile = 'mega-8stem'; saveSettings(); }}
                >8 stems</button>
                <div class="mode-tooltip" role="tooltip">
                  <strong>8 stems</strong>
                  <span>Separates vocals, bass, guitar, piano, kick, snare, kit, and other.</span>
                  <span>Higher detail drum separation, but takes longer to finish.</span>
                </div>
              </div>
            </div>
          {/if}
          <div class="mode-select" role="group" aria-label="Export destination">
            <div class="mode-pill-wrapper">
              <span class="mode-pill active">Export</span>
              <div class="mode-tooltip" role="tooltip">
                <strong>Exports to separate folder</strong>
                <span>Writes complete separated songs to the destination folder.</span>
                <span>CON files are extracted to the specified folder.</span>
                <span>Source files remain untouched.</span>
              </div>
            </div>
            <svg class="mode-pill-arrow" viewBox="0 0 16 16" aria-hidden="true"><path d="M3 8h10M9 4l4 4-4 4" /></svg>
            <div class="mode-pill-wrapper">
              <button
                type="button"
                class="mode-folder-pill"
                disabled={busy}
                onclick={chooseExportFolder}
                aria-label="Choose export folder"
              >
                <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 7V5a1 1 0 0 1 1-1h5l2 3h9a1 1 0 0 1 1 1v11a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V7Z" /></svg>
                <span>{settings.exportFolder ? settings.exportFolder.split(/[/\\]/).filter(Boolean).pop() : 'Choose folder…'}</span>
              </button>
              <div class="mode-tooltip" role="tooltip">
                <strong>Output folder</strong>
                <span>Destination folder where exported songs are saved.</span>
                {#if settings.exportFolder}<span>{settings.exportFolder}</span>{/if}
              </div>
            </div>
          </div>
        </div>
        <div class="actions">
          {#if running}
            <button class="stop" onclick={pauseQueue} disabled={pausing}>{pausing ? 'Pausing…' : 'Pause'}</button>
          {:else if paused && ready > 0}
            <button
              class="primary"
              onclick={() => run()}
              disabled={!native || busy || !engineReady || !settings.exportFolder}
            >
              {!settings.exportFolder ? 'Choose destination' : `Resume (${ready} ${ready === 1 ? 'song' : 'songs'})`}
            </button>
          {:else}
            <button
              class="primary"
              onclick={() => run()}
              disabled={!native || busy || !engineReady || ready === 0 || !settings.exportFolder}
            >
              {!settings.exportFolder ? 'Choose destination' : `Separate ${ready > 0 ? ready : ''} songs`}
            </button>
          {/if}
        </div>
      </div>
    </section>
  </main>
</div>
