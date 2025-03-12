import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/backend-api/:path*",
        destination: "http://localhost:5000/data/:path*",
      },
    ];
  },
};

export default nextConfig;
