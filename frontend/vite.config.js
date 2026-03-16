import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/summarise":          "http://localhost:8000",
      "/health":             "http://localhost:8000",
      "/check-cache":        "http://localhost:8000",
      "/regenerate-summary": "http://localhost:8000",
      "/manual-shows":        "http://localhost:8000",
      "/relationship-types":  "http://localhost:8000",
    },
  },
});
