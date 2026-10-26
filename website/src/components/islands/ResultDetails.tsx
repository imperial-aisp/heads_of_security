import { useId, useState } from 'react';
import type { ReactNode } from 'react';
import { Button } from '../ds/Button';
import { Icon } from '../ds/Icon';

interface ResultDetailsProps {
  id: string;
  title: string;
  /** One-line summary, always visible. */
  summary: string;
  /** The details, collapsed by default. */
  children?: ReactNode;
}

export default function ResultDetails({ id, title, summary, children }: ResultDetailsProps) {
  const [open, setOpen] = useState(false);
  const bodyId = useId();
  return (
    <div id={id} className="subsection">
      <div className="subsection-head">
        <h3>{title}</h3>
        <div className="nowrap">
          <Button variant="ghost" size="sm" onClick={() => setOpen(!open)} aria-expanded={open} aria-controls={bodyId}>
            {open ? 'Hide details' : 'Show details'}
            <Icon name={open ? 'chevron-up' : 'chevron-down'} size={14} />
          </Button>
        </div>
      </div>
      <p className="subsection-summary">{summary}</p>
      <div id={bodyId} className="subsection-body" hidden={!open}>
        {children}
      </div>
    </div>
  );
}
