import { Button } from './Button';
import { Icon } from './Icon';
import { useCopy } from './useCopy';

interface CodeBlockProps {
  code: string;
  /** Show the built-in "Copy" button in the top-right corner (the same button as the snippets'). */
  copyable?: boolean;
  copyLabel?: string;
}

export function CodeBlock({ code, copyable = true, copyLabel = 'Copy' }: CodeBlockProps) {
  const [copied, copy] = useCopy();
  return (
    <div style={{ position: 'relative' }}>
      <pre className="codeblock">{code}</pre>
      {copyable ? (
        <Button variant="ghost" size="sm" className="codeblock-copy" onClick={() => copy('code', code)}>
          <Icon name="copy" size={14} />
          {copied ? 'Copied' : copyLabel}
        </Button>
      ) : null}
    </div>
  );
}
