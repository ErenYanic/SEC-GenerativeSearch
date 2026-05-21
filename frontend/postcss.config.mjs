// Tailwind v4 uses its own PostCSS plugin; no `tailwindcss` / `autoprefixer`
// entries needed — the @tailwindcss/postcss plugin handles everything.
const config = {
  plugins: {
    "@tailwindcss/postcss": {},
  },
};

export default config;
