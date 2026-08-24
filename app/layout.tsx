import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "AI Text Detector & Humanizer",
  description:
    "Detect AI-generated text with per-sentence evidence, then rewrite the " +
    "flagged parts while preserving meaning. Two honest scores, never a fake 0%.",
  robots: { index: true, follow: true },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen font-sans antialiased">
        <a
          href="#workspace"
          className="sr-only focus:not-sr-only focus:absolute focus:left-3 focus:top-3 focus:z-50 focus:rounded focus:bg-white focus:px-3 focus:py-2 focus:text-black"
        >
          Skip to the workspace
        </a>
        {children}
      </body>
    </html>
  );
}
