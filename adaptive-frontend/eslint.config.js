// eslint.config.js
import js from "@eslint/js";
import globals from "globals";
import tseslint from "typescript-eslint";
import { FlatCompat } from "@eslint/eslintrc";

const compat = new FlatCompat({
  baseDirectory: import.meta.dirname,
});

export default tseslint.config(
  {
    ignores: [".next/**", "dist/**", "node_modules/**"],
  },
  {
    files: ["**/*.{js,jsx,ts,tsx}"],
    languageOptions: {
      ecmaVersion: "latest",
      sourceType: "module",
      globals: {
        ...globals.browser,
        // ...globals.node, // Uncomment if you have server-side ESLint needs
      },
      parser: tseslint.parser,
      parserOptions: {
        project: ["./tsconfig.json"],
        // Optionally, specify `ecmaFeatures.jsx: true` if not covered by extended configs
      },
    },
    extends: [
      js.configs.recommended,
      ...tseslint.configs.recommended,
      // Extends Next.js core rules (includes React, React Hooks, Next.js specific rules)
      ...compat.config({ extends: ["next"] }).map((config) => ({
        ...config,
        files: ["**/*.{js,jsx,ts,tsx}"], // Ensure Next.js rules apply to your files
      })),
      // Optional: Strict rules for Core Web Vitals
      // ...compat.config({ extends: ['next/core-web-vitals'] }).map(config => ({
      //   ...config,
      //   files: ['**/*.{js,jsx,ts,tsx}'],
      // })),
      // Prettier integration (MUST BE LAST)
      ...compat.config({ extends: ["prettier"] }).map((config) => ({
        ...config,
        files: ["**/*.{js,jsx,ts,tsx}"],
      })),
    ],
    rules: {
      // Custom rule overrides
      // 'react/no-unescaped-entities': 'off',
      // '@next/next/no-img-element': 'off',
      // '@typescript-eslint/no-explicit-any': 'warn',
    },
    settings: {
      react: {
        version: "detect",
      },
      // next: {
      //   rootDir: 'packages/my-app/', // For monorepos
      // },
    },
  },
);
