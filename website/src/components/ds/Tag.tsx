import type { CSSProperties, ReactNode } from 'react';

export type TagTone = 'accent' | 'neutral' | 'outline' | 'trusted' | 'untrusted' | 'system' | 'caution' | 'slate';

export function Tag({ tone = 'accent', style, children }: { tone?: TagTone; style?: CSSProperties; children?: ReactNode }) {
  return <span className={'tag tag-' + tone} style={style}>{children}</span>;
}
