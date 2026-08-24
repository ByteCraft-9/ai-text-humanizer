/**
 * Copy pdf.js's worker into /public.
 *
 * The worker must be served from our own origin: loading it from a CDN would
 * mean every uploaded PDF's parsing depends on a third party, which
 * contradicts the promise that file handling never leaves the browser.
 * Copied at build time rather than committed so it can never drift from the
 * installed pdfjs-dist version.
 */

import { copyFileSync, mkdirSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";

const require = createRequire(import.meta.url);
const pdfjsRoot = dirname(require.resolve("pdfjs-dist/package.json"));
const source = join(pdfjsRoot, "build", "pdf.worker.min.mjs");
const target = join(process.cwd(), "public", "pdf.worker.min.mjs");

mkdirSync(dirname(target), { recursive: true });
copyFileSync(source, target);
console.log(`pdf.worker.min.mjs -> ${target}`);
