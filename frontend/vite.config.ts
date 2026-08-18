import { defineConfig } from "vite";

// Local dev only: the backend serves API + built frontend from the same origin
// in production, so app.js deliberately uses relative paths on localhost (see
// API_BASE_URL). Running the frontend on its own Vite port against a
// separately-running backend needs this proxy to bridge that gap — it
// doesn't affect the production build, which never goes through vite dev.
const BACKEND_URL = process.env.VITE_DEV_BACKEND_URL || "http://localhost:8521";

export default defineConfig({
  server: {
    port: process.env.PORT ? Number(process.env.PORT) : 5173,
    strictPort: !!process.env.PORT,
    proxy: {
      "/ask-stream": BACKEND_URL,
      "/auth": BACKEND_URL,
      "/admin": BACKEND_URL,
      "/conversation": BACKEND_URL,
      "/docs": BACKEND_URL,
      "/upload": BACKEND_URL,
      "/videos": BACKEND_URL,
      "/webpages": BACKEND_URL,
      "/tunnel-url": BACKEND_URL,
      "/health": BACKEND_URL,
    },
  },
  build: {
    outDir: "dist",
  },
});
