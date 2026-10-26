import { useId, useState } from 'react';
import type { ReactNode } from 'react';
import { Button } from '../ds/Button';
import { Icon } from '../ds/Icon';

/** "Read the abstract" ghost toggle; the abstract (passed as children) is collapsed by default. */
export default function AbstractToggle({ defaultOpen = false, children }: { defaultOpen?: boolean; children?: ReactNode }) {
  const [open, setOpen] = useState(defaultOpen);
  const bodyId = useId();
  return (
    <div className="abstract-block">
      <div className="nowrap">
        <Button variant="ghost" size="sm" onClick={() => setOpen(!open)} aria-expanded={open} aria-controls={bodyId}>
          {open ? 'Hide the abstract' : 'Read the abstract'}
          <Icon name={open ? 'chevron-up' : 'chevron-down'} size={14} />
        </Button>
      </div>
      <div id={bodyId} className="prose abstract" hidden={!open}>
        {children}
      </div>
    </div>
  );
}
