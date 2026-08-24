import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./lib/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Heatmap bands. Paired with numeric labels + patterned underlines
        // so nothing is encoded by colour alone (WCAG 2.1 AA, PRD 15.3).
        band: {
          human: "#0f766e",
          low: "#65a30d",
          mid: "#ca8a04",
          high: "#ea580c",
          ai: "#be123c",
        },
      },
      fontFamily: {
        sans: ["ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "sans-serif"],
        mono: ["ui-monospace", "SFMono-Regular", "Consolas", "monospace"],
      },
    },
  },
  plugins: [],
};
export default config;
