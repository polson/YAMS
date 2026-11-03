import test from 'node:test';
import assert from 'node:assert/strict';
import { SUPPORTED_LOCALES, translate, type SupportedLocale } from '../src/i18n/index.ts';
import { en } from '../src/i18n/locales/en.ts';
import { es } from '../src/i18n/locales/es.ts';
import { de } from '../src/i18n/locales/de.ts';
import { fr } from '../src/i18n/locales/fr.ts';
import { ja } from '../src/i18n/locales/ja.ts';
import { pt } from '../src/i18n/locales/pt.ts';
import { zh } from '../src/i18n/locales/zh.ts';
import { timing, updateSong, type Song } from '../src/queue.ts';

const LOCALES: Record<SupportedLocale, Record<string, string>> = {
  en,
  es,
  de,
  fr,
  ja,
  pt,
  zh,
};

test('all supported locales contain all translation keys with non-empty values', () => {
  const englishKeys = Object.keys(en);
  assert.equal(SUPPORTED_LOCALES.length, 7);

  for (const { code } of SUPPORTED_LOCALES) {
    const dictionary = LOCALES[code];
    assert.ok(dictionary, `Missing dictionary for locale ${code}`);
    for (const key of englishKeys) {
      const val = dictionary[key];
      assert.ok(typeof val === 'string' && val.trim().length > 0, `Missing or empty key '${key}' in locale '${code}'`);
    }
  }
});

test('parameter substitution interpolates correctly in all languages', () => {
  for (const { code } of SUPPORTED_LOCALES) {
    const formatted = translate(code, 'count_completed', { count: 42 });
    assert.ok(formatted.includes('42'), `Failed to interpolate count in locale ${code}: ${formatted}`);
  }

  assert.equal(translate('es', 'count_completed', { count: 5 }), '5 completadas');
  assert.equal(translate('de', 'count_completed', { count: 5 }), '5 abgeschlossen');
  assert.equal(translate('fr', 'count_completed', { count: 5 }), '5 terminés');
  assert.equal(translate('ja', 'count_completed', { count: 5 }), '5 件完了');
  assert.equal(translate('pt', 'count_completed', { count: 5 }), '5 concluídas');
  assert.equal(translate('zh', 'count_completed', { count: 5 }), '5 首已完成');
});

test('timing formats remaining duration in target locale', () => {
  const song: Song = { id: 'x', folder: '/songs/x', status: 'running', source: '', backup: '', message: '', detail: '', progress: 95, log: '', canRetry: false, startedAt: 0, finishedAt: 0, inferenceAt: 0 };
  assert.equal(timing(song, 10000, 'en'), '0:10 · ~15s left');
  assert.equal(timing(song, 10000, 'es'), '0:10 · ~15s restante');
  assert.equal(timing(song, 10000, 'de'), '0:10 · ~15s verbleibend');
  assert.equal(timing(song, 10000, 'fr'), '0:10 · ~15s restant');
  assert.equal(timing(song, 10000, 'ja'), '0:10 · 残り ~15s');
  assert.equal(timing(song, 10000, 'pt'), '0:10 · ~15s restante');
  assert.equal(timing(song, 10000, 'zh'), '0:10 · 约剩余 ~15s');
});

test('updateSong produces translated details for worker events across locales', () => {
  const song: Song = { id: 'x', folder: '/songs/x', status: 'ready', source: '', backup: '', message: '', detail: '', progress: 0, log: '', canRetry: false, startedAt: 0, finishedAt: 0, inferenceAt: 0 };
  const esStarted = updateSong(song, { kind: 'started', id: 'x' }, 0, 'es');
  assert.equal(esStarted.detail, 'Iniciando…');

  const de6Stem = updateSong(song, { kind: 'worker', id: 'x', payload: { event: 'model_progress', folder: song.folder, target: '6stem', pass_number: 1, fraction: 0.5 } }, 0, 'de');
  assert.equal(de6Stem.detail, '6 Stems werden getrennt');

  const jaDrum = updateSong(song, { kind: 'worker', id: 'x', payload: { event: 'model_progress', folder: song.folder, target: 'drumsep', pass_number: 2, fraction: 0.5 } }, 0, 'ja');
  assert.equal(jaDrum.detail, 'ドラムパートを分離中');

  const zhCompleted = updateSong(song, { kind: 'finished', id: 'x', status: 'completed', message: '' }, 0, 'zh');
  assert.equal(zhCompleted.detail, '已安装分轨 · 已备份原始文件');
});
