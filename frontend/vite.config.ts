/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "path";

// Where /api is forwarded. The dev server normally runs on the same machine as
// uvicorn (the collection host the captures are shipped to), so localhost is
// the default; point API_TARGET elsewhere to drive a backend on another box.
const API_TARGET = process.env.API_TARGET ?? "http://localhost:8000";

// Vite refuses requests whose Host header it does not recognise (DNS-rebinding
// protection, on since 5.4.12), which rejects every LAN name the collection
// host is reached by -- http://lg:5173 included. `true` accepts any host, which
// is what a lab network wants; VITE_ALLOWED_HOSTS narrows it to a list when the
// server sits somewhere less trusted. A leading dot matches subdomains.
//
//   VITE_ALLOWED_HOSTS=lg,lg.local,.example.com
const ALLOWED_HOSTS = process.env.VITE_ALLOWED_HOSTS
  ? process.env.VITE_ALLOWED_HOSTS.split(",").map((h) => h.trim()).filter(Boolean)
  : true;

// host: true binds 0.0.0.0 rather than localhost, so the machine is reachable
// by name from another laptop. strictPort makes a taken port fail loudly
// instead of silently moving to 5174 -- a moved port looks exactly like a dead
// server from across the network.
const server = {
  host: true,
  port: 5173,
  strictPort: true,
  allowedHosts: ALLOWED_HOSTS,
  proxy: {
    // The frontend only ever fetches relative /api URLs, so proxying that one
    // prefix is the whole contract: same origin for the page and the API, no
    // CORS and no port to configure in the client.
    "/api": {
      target: API_TARGET,
      changeOrigin: true,
    },
  },
} as const;

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server,
  // `vite preview` serves the built bundle and needs the same proxy, otherwise
  // checking a production build over the network 404s on every /api call.
  preview: { ...server, port: 4173 },
  test: {
    environment: "node",
  },
});
