/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",  // Required for Docker deployment
  images: {
    remotePatterns: [
      { protocol: "http", hostname: "localhost" },
      { protocol: "https", hostname: "**" },
    ],
  },
  async rewrites() {
    // Proxy /api calls to backend in development
    return process.env.NODE_ENV === "development"
      ? [
          {
            source: "/api/:path*",
            destination: `${process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"}/api/:path*`,
          },
        ]
      : [];
  },
};

module.exports = nextConfig;
