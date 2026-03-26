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
};

export default nextConfig;
