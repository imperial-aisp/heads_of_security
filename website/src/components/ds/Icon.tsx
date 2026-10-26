import { BookOpen, Box, ChevronDown, ChevronUp, Copy, ExternalLink, FileText, Github, Play, Shuffle } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';

// Lucide icons used on the page (lucide-react pinned to 0.469.0, the version the prototype loads;
// its brand icons such as Github are marked deprecated but still ship).
const ICONS = {
  'book-open': BookOpen,
  box: Box,
  'chevron-down': ChevronDown,
  'chevron-up': ChevronUp,
  copy: Copy,
  'external-link': ExternalLink,
  'file-text': FileText,
  github: Github,
  play: Play,
  shuffle: Shuffle,
} satisfies Record<string, LucideIcon>;

export type IconName = keyof typeof ICONS;

interface IconProps {
  name: IconName;
  size?: number;
  strokeWidth?: number;
  label?: string;
}

export function Icon({ name, size = 16, strokeWidth = 2, label }: IconProps) {
  const Glyph = ICONS[name];
  return (
    <span
      role={label ? 'img' : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : true}
      style={{ display: 'inline-flex', width: size, height: size, flex: 'none' }}
    >
      <Glyph size={size} strokeWidth={strokeWidth} style={{ display: 'block' }} />
    </span>
  );
}
