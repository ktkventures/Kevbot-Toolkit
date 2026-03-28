/** @type {import('next').NextConfig} */
const nextConfig = {
  // Ignore ESLint and TypeScript errors during build
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
  // Use standalone output for Docker deployments
  output: 'standalone',
  // Disable SWC minification — use Terser instead.
  // SWC's minifier reorders const declarations causing TDZ errors
  // ("Cannot access 'X' before initialization") in production builds.
  swcMinify: false,
};

export default nextConfig;
