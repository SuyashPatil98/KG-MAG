/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        mono: ["var(--font-mono)", "IBM Plex Mono", "Courier New", "monospace"],
      },
      colors: {
        "kg-bg": "#0a0a0f",
        "kg-surface": "#12121f",
        "kg-border": "#1e1e2e",
        "kg-accent": "#7c3aed",
        "kg-accent-light": "#a78bfa",
        "kg-text": "#e8e6f0",
        "kg-muted": "#6b7280",
      },
      typography: {
        invert: {
          css: {
            "--tw-prose-body": "#d1d5db",
            "--tw-prose-headings": "#c4b5fd",
            "--tw-prose-links": "#818cf8",
            "--tw-prose-code": "#a78bfa",
          },
        },
      },
    },
  },
  plugins: [require("@tailwindcss/typography")],
};
