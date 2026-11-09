/** Static export: the whole site (UI + pre-processed model data) is served as plain files,
 *  e.g. from GitHub Pages. NEXT_PUBLIC_BASE_PATH is set by the deploy workflow ("/<repo>"). */
const basePath = process.env.NEXT_PUBLIC_BASE_PATH || '';
module.exports = {
  output: 'export',
  basePath,
  trailingSlash: true,
  images: { unoptimized: true },
  reactStrictMode: true,
};
