import { defineConfig } from "vite";
import path from "path";
import react from "@vitejs/plugin-react";
import { svelte } from "@sveltejs/vite-plugin-svelte";

export default defineConfig({
  plugins: [
    svelte({
      // Annotorious's editor host still instantiates registered shape editors
      // with the Svelte 4 class API. See point-editor/register-point-tools.ts.
      compilerOptions: {
        compatibility: {
          componentApi: 4,
        },
      },
    }),
    react(),
  ],
  publicDir: "public",
  build: {
    outDir: path.resolve(__dirname, "../api/static"),
    commonjsOptions: {
      transformMixedEsModules: true, // ✅ fixes CJS/ESM interop
    },
    rolldownOptions: {
      output: {
        codeSplitting: {
          groups: [
            {
              name: "react",
              test: /node_modules\/react\//,
            },
            {
              name: "plotly",
              test: /plotly/,
            },
          ],
        },
      },
    },
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src"), // use @ as alias for src/
      stream: "stream-browserify",
    },
  },
  define: {
    global: "globalThis", // polyfill Node's global
    "process.env": {},
  },
  optimizeDeps: {
    include: ["plotly.js-dist-min"],
  },
});
