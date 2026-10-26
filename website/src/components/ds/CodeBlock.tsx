import { useCopy } from './useCopy';

interface CodeBlockProps {
  code: string;
  /** Show the built-in "Copy" button in the top-right corner. */
  copyable?: boolean;
  copyLabel?: string;
}

export function CodeBlock({ code, copyable = true, copyLabel = 'Copy' }: CodeBlockProps) {
  const [copied, copy] = useCopy();
  return (
    <div style={{ position: 'relative' }}>
      <pre className="codeblock">{code}</pre>
      {copyable ? (
        <button type="button" className="btn btn-ghost btn-sm codeblock-copy" onClick={() => copy('code', code)}>
          {copied ? 'Copied' : copyLabel}
        </button>
      ) : null}
    </div>
  );
}
