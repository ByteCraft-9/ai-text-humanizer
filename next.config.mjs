/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,

  /**
   * In production Vercel routes /api/py/* to the Python functions via
   * vercel.json. `next dev` cannot, so development proxies to the local
   * Python server instead — `npm run dev` starts both.
   */
  async rewrites() {
    if (process.env.NODE_ENV !== "development") return [];
    const target = process.env.PY_DEV_URL || "http://127.0.0.1:8000";
    return [{ source: "/api/py/:path*", destination: `${target}/api/py/:path*` }];
  },

  // pdfjs-dist ships a worker that must not be bundled server-side.
  webpack: (config, { isServer }) => {
    if (!isServer) {
      config.resolve.fallback = { ...config.resolve.fallback, canvas: false, fs: false };
    }
    return config;
  },
};
export default nextConfig;
