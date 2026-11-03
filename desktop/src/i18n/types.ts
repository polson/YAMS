export type SupportedLocale = 'en' | 'es' | 'de' | 'fr' | 'ja' | 'pt' | 'zh';

export interface LocaleInfo {
  code: SupportedLocale;
  label: string;
}
