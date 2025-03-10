import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";
import { visualizer } from "rollup-plugin-visualizer";
import compression from "vite-plugin-compression";

import { TanStackRouterVite } from "@tanstack/router-plugin/vite";

export default defineConfig({
  plugins: [
    react(),
    compression(), // Enables Gzip/Brotli compression
    TanStackRouterVite({ autoCodeSplitting: true }),
    visualizer(), // Generates bundle analysis report
  ],

  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
      "@components": path.resolve(__dirname, "./src/components"),
      "@hooks": path.resolve(__dirname, "./src/hooks"),
      "@pages": path.resolve(__dirname, "./src/pages"),
    },
  },

  build: {
    target: "esnext",
    outDir: "dist",
    cssCodeSplit: true,
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ["react", "react-dom"],
        },
        // Optimize chunk size
        chunkFileNames: "assets/js/[name]-[hash].js",
        entryFileNames: "assets/js/[name]-[hash].js",
        assetFileNames: "assets/[ext]/[name]-[hash].[ext]",
      },
    },
    // Enable minification optimizations
    minify: "terser",
    terserOptions: {
      compress: {
        drop_console: false,
        drop_debugger: false,
      },
    },
  },

  server: {
    port: 3000,
    strictPort: true,
    host: true,
    // Enable HMR with overlay
    hmr: {
      overlay: true,
    },
  },

  preview: {
    port: 4173,
    strictPort: true,
    host: true,
  },

  optimizeDeps: {
    include: ["react", "react-dom"],
    exclude: ["@testing-library/react"],
  },

  // Enable source maps for production debugging
  css: {
    devSourcemap: true,
  },

  // Enable detailed build logging
  logLevel: "info",

  // Cache configuration
  cacheDir: ".vite",
});
