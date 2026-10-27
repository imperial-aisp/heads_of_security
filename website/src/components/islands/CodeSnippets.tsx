import { CodeBlock } from '../ds/CodeBlock';

interface Snippet {
  key: string;
  label: string;
  code: string;
}

/** Labelled code blocks, each with the block's own Copy button in its corner (the same as the BibTeX block's). */
export default function CodeSnippets({ snippets }: { snippets: Snippet[] }) {
  return (
    <div className="code-stack">
      {snippets.map((s) => (
        <div key={s.key} className="snippet">
          <div className="snippet-head">
            <span className="field-label">{s.label}</span>
          </div>
          <CodeBlock code={s.code} />
        </div>
      ))}
    </div>
  );
}
