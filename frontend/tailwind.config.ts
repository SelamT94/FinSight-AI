import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          50: "#f0f9ff",
          600: "#0a84ff",
          700: "#0066cc",
          900: "#0c4a6e",
        },
        glass: {
          accent: "#64d2ff",
          violet: "#bf5af2",
        },
      },
      borderRadius: {
        glass: "1.75rem",
      },
      animation: {
        "glass-shimmer": "glass-shimmer 3s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};

export default config;
