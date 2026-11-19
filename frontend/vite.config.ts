import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The API has no CORS middleware, so in dev the frontend talks to it through
// this proxy: /api/get_models -> http://localhost:8000/get_models.
// Override the target with API_PROXY_TARGET, or skip the proxy entirely in a
// deployed build by setting VITE_API_BASE to the API's URL.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": {
        target: process.env.API_PROXY_TARGET ?? "http://localhost:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
