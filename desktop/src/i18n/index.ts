import type { LocaleInfo, SupportedLocale } from './types.ts';
import { en, type TranslationKey, type Translations } from './locales/en.ts';
import { es } from './locales/es.ts';
import { de } from './locales/de.ts';
import { fr } from './locales/fr.ts';
import { ja } from './locales/ja.ts';
import { pt } from './locales/pt.ts';
import { zh } from './locales/zh.ts';

export type { SupportedLocale, LocaleInfo } from './types.ts';
export type { TranslationKey, Translations } from './locales/en.ts';

export const SUPPORTED_LOCALES: LocaleInfo[] = [
  { code: 'en', label: 'English' },
  { code: 'es', label: 'Español' },
  { code: 'de', label: 'Deutsch' },
  { code: 'fr', label: 'Français' },
  { code: 'ja', label: '日本語' },
  { code: 'pt', label: 'Português' },
  { code: 'zh', label: '简体中文' },
];

const LOCALES: Record<SupportedLocale, Translations> = {
  en,
  es,
  de,
  fr,
  ja,
  pt,
  zh,
};

export function resolveInitialLocale(): SupportedLocale {
  for (const language of navigator.languages) {
    const code = language.toLowerCase().split('-')[0];
    const match = SUPPORTED_LOCALES.find(locale => locale.code === code);
    if (match) {
      return match.code;
    }
  }
  return 'en';
}

export function translate(
  locale: SupportedLocale,
  key: TranslationKey,
  params?: Record<string, string | number>
): string {
  const dictionary = LOCALES[locale] || LOCALES.en;
  let text: string = dictionary[key] ?? LOCALES.en[key] ?? key;
  if (!params) {
    return text;
  }
  for (const [name, value] of Object.entries(params)) {
    text = text.replaceAll(`{${name}}`, String(value));
  }
  return text;
}
