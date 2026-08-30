import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: true, // allows opening the dev server from your phone on the same wifi
    port: 5173,
  },
});
