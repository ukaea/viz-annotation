import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/backend-api/:path*",
        destination: "http://data_app:8002/data/:path*",
      },
    ];
  },
};

export default nextConfig;
