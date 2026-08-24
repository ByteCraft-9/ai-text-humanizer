/**
 * Client-side file parsing (PRD 6.1, 15.2).
 *
 * All parsing happens in the browser: nothing large crosses Vercel's 4.5 MB
 * body limit, it costs zero server compute, and untrusted binaries never reach
 * the server. Scanned PDFs with no text layer fail loudly rather than
 * returning silent garbage (PRD 16.2).
 */

export const ACCEPTED_EXTENSIONS = [".txt", ".md", ".pdf", ".docx"] as const;
export const MAX_FILE_BYTES = 20 * 1024 * 1024;

export class ExtractionError extends Error {
  constructor(message: string, readonly kind: ExtractionErrorKind) {
    super(message);
    this.name = "ExtractionError";
  }
}

export type ExtractionErrorKind =
  | "unsupported"
  | "too_large"
  | "no_text_layer"
  | "encrypted"
  | "corrupt"
  | "empty";

export interface ExtractionResult {
  text: string;
  /** Pages for PDFs, undefined otherwise. */
  pages?: number;
  source: string;
}

export type ProgressFn = (fraction: number, label: string) => void;

function extensionOf(name: string): string {
  const dot = name.lastIndexOf(".");
  return dot === -1 ? "" : name.slice(dot).toLowerCase();
}

/** Collapse the ragged whitespace that PDF and DOCX extraction produce. */
export function normalizeText(raw: string): string {
  return raw
    .replace(/\r\n?/g, "\n")
    .replace(/\u00a0/g, " ")
    // De-hyphenate words broken across a line: "compre-\nhensive".
    .replace(/([a-z])-\n([a-z])/g, "$1$2")
    // A single newline inside a paragraph is a soft wrap, not a break.
    .replace(/([^\n])\n(?!\n)/g, "$1 ")
    .replace(/[ \t]{2,}/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

async function extractPdf(file: File, onProgress?: ProgressFn): Promise<ExtractionResult> {
  const pdfjs = await import("pdfjs-dist");
  // Worker is served from /public so it loads without a CDN.
  pdfjs.GlobalWorkerOptions.workerSrc = "/pdf.worker.min.mjs";

  const buffer = await file.arrayBuffer();
  let doc;
  try {
    doc = await pdfjs.getDocument({ data: buffer, isEvalSupported: false }).promise;
  } catch (err) {
    const message = String((err as Error)?.message ?? err);
    if (/password/i.test(message)) {
      throw new ExtractionError(
        "This PDF is password-protected. Remove the password and try again.",
        "encrypted",
      );
    }
    throw new ExtractionError("This PDF could not be opened — it may be corrupt.", "corrupt");
  }

  const parts: string[] = [];
  for (let page = 1; page <= doc.numPages; page++) {
    onProgress?.(page / doc.numPages, `Reading page ${page} of ${doc.numPages}`);
    const content = await (await doc.getPage(page)).getTextContent();
    const text = content.items
      .map((item) => ("str" in item ? item.str : ""))
      .join(" ");
    parts.push(text);
  }

  const text = normalizeText(parts.join("\n\n"));
  if (text.replace(/\s/g, "").length < 40) {
    throw new ExtractionError(
      "This PDF has no selectable text layer — it looks like a scan or images. " +
        "Run OCR on it first, or paste the text directly.",
      "no_text_layer",
    );
  }
  return { text, pages: doc.numPages, source: file.name };
}

async function extractDocx(file: File): Promise<ExtractionResult> {
  const mammoth = await import("mammoth");
  const buffer = await file.arrayBuffer();
  try {
    const { value } = await mammoth.extractRawText({ arrayBuffer: buffer });
    return { text: normalizeText(value), source: file.name };
  } catch {
    throw new ExtractionError(
      "This .docx could not be read. If it is an older .doc, re-save it as .docx.",
      "corrupt",
    );
  }
}

async function extractPlain(file: File): Promise<ExtractionResult> {
  const text = await file.text();
  return { text: normalizeText(text), source: file.name };
}

export async function extractFile(
  file: File,
  onProgress?: ProgressFn,
): Promise<ExtractionResult> {
  if (file.size > MAX_FILE_BYTES) {
    throw new ExtractionError(
      `That file is ${(file.size / 1024 / 1024).toFixed(1)} MB. The limit is ` +
        `${MAX_FILE_BYTES / 1024 / 1024} MB.`,
      "too_large",
    );
  }

  const ext = extensionOf(file.name);
  onProgress?.(0, "Opening file");

  let result: ExtractionResult;
  switch (ext) {
    case ".pdf":
      result = await extractPdf(file, onProgress);
      break;
    case ".docx":
      result = await extractDocx(file);
      break;
    case ".txt":
    case ".md":
      result = await extractPlain(file);
      break;
    default:
      throw new ExtractionError(
        `${ext || "That file type"} is not supported. Accepted: ` +
          ACCEPTED_EXTENSIONS.join(", "),
        "unsupported",
      );
  }

  onProgress?.(1, "Done");
  if (!result.text.trim()) {
    throw new ExtractionError("That file contains no readable text.", "empty");
  }
  return result;
}
