/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        paper: "#F3F5F4",
        panel: "#FBFCFB",
        ink: "#17211F",
        muted: "#5B6763",
        line: "#C9D1CE",
        mask: { DEFAULT: "#1F6B5B", soft: "#DCEBE6", deep: "#154A3F" },
        amber: { DEFAULT: "#B7791F", soft: "#F4E8D3" },
        red: { DEFAULT: "#A33A2E", soft: "#F3DCD8" },
      },
      fontFamily: { sans: ["'IBM Plex Sans'", "system-ui", "sans-serif"] },
    },
  },
  plugins: [],
};
