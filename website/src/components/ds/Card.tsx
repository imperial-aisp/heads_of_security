import type { ReactNode } from 'react';

interface CardProps {
  kicker?: ReactNode;
  title?: ReactNode;
  tone?: 'trusted' | 'untrusted' | 'system' | 'accent';
  className?: string;
  children?: ReactNode;
}

export function Card({ kicker, title, tone, className = '', children }: CardProps) {
  const cls = ['card', tone ? 'card-' + tone : '', className].filter(Boolean).join(' ');
  return (
    <div className={cls}>
      {kicker ? <div className="card-kicker">{kicker}</div> : null}
      {title ? <div className="card-title">{title}</div> : null}
      {typeof children === 'string' ? <p className="card-body">{children}</p> : children}
    </div>
  );
}
