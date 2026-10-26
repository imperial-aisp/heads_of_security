import type { AnchorHTMLAttributes, ButtonHTMLAttributes, ReactNode } from 'react';

type Variant = 'primary' | 'secondary' | 'ghost';

interface BaseProps {
  variant?: Variant;
  size?: 'md' | 'sm';
  className?: string;
  children?: ReactNode;
}

type ButtonProps = BaseProps &
  ((ButtonHTMLAttributes<HTMLButtonElement> & { href?: undefined }) | (AnchorHTMLAttributes<HTMLAnchorElement> & { href: string }));

function buttonClass(variant: Variant = 'primary', size: 'md' | 'sm' = 'md', className = '') {
  return ['btn', 'btn-' + variant, size === 'sm' ? 'btn-sm' : '', className].filter(Boolean).join(' ');
}

/** Renders an <a> when given an href, otherwise a <button type="button">. */
export function Button({ variant = 'primary', size = 'md', className = '', children, ...rest }: ButtonProps) {
  const cls = buttonClass(variant, size, className);
  if (rest.href !== undefined) {
    return <a className={cls} {...(rest as AnchorHTMLAttributes<HTMLAnchorElement>)}>{children}</a>;
  }
  return <button type="button" className={cls} {...(rest as ButtonHTMLAttributes<HTMLButtonElement>)}>{children}</button>;
}
