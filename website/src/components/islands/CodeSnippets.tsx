import { Button } from '../ds/Button';
import { CodeBlock } from '../ds/CodeBlock';
import { Icon } from '../ds/Icon';
import { useCopy } from '../ds/useCopy';

interface Snippet {
  key: string;
  label: string;
  code: string;
}

/** Labelled code blocks, each with a ghost Copy button that reads "Copied" for a moment. */
export default function CodeSnippets({ snippets }: { snippets: Snippet[] }) {
  const [copied, copy] = useCopy();
  return (
    <div className="code-stack">
      {snippets.map((s) => (
        <div key={s.key} className="snippet">
          <div className="snippet-head">
            <span className="field-label">{s.label}</span>
            <Button variant="ghost" size="sm" onClick={() => copy(s.key, s.code)}>
              <Icon name="copy" size={14} />
              {copied === s.key ? 'Copied' : 'Copy'}
            </Button>
          </div>
          <CodeBlock code={s.code} copyable={false} />
        </div>
      ))}
    </div>
  );
}
