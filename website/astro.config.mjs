// @ts-check
import { defineConfig } from 'astro/config';
import react from '@astrojs/react';

// Deployed as a GitHub Pages project site. The Pages workflow passes SITE_URL / BASE_PATH from
// actions/configure-pages; BASE_PATH is "" on a custom domain, which serves the site from the root.
const basePath = process.env.BASE_PATH;

export default defineConfig({
  site: process.env.SITE_URL || 'https://imperial-aisp.github.io',
  base: basePath === undefined ? '/heads_of_security' : basePath || '/',
  integrations: [react()],
});
