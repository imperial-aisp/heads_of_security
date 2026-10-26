import type { ImageMetadata } from 'astro';

// A figure from the paper, shown as its own SVG (exported from the paper's PDF) in an <img>, so each file
// keeps its own ids and fonts.

interface PaperFigureProps {
  src: ImageMetadata;
  alt: string;
  /** CSS width of the figure; defaults to 90% of the column. Narrower figures are centred. */
  width?: string;
}

export function PaperFigure({ src, alt, width = '90%' }: PaperFigureProps) {
  return (
    <img
      src={src.src}
      width={src.width}
      height={src.height}
      alt={alt}
      loading="lazy"
      decoding="async"
      style={{ display: 'block', width, height: 'auto', marginInline: 'auto' }}
    />
  );
}
