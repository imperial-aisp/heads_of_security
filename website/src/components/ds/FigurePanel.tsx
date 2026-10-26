import type { CSSProperties, ReactNode } from 'react';

interface FigurePanelProps {
  tone?: 'method' | 'utility' | 'security' | 'plain';
  title?: ReactNode;
  className?: string;
  style?: CSSProperties;
  children?: ReactNode;
}

export function FigurePanel({ tone = 'method', title, className = '', style, children }: FigurePanelProps) {
  const cls = ['fig', 'fig-panel', tone !== 'method' ? 'fig-panel-' + tone : '', className].filter(Boolean).join(' ');
  return (
    <section className={cls} style={style}>
      {title ? <h3 className="fig-panel-title">{title}</h3> : null}
      {children}
    </section>
  );
}
