import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/backend-api/:path*",
        destination: "http://data_app:8002/data/:path*",
      },
      {
        source: "/db-api/shots/:path*",
        destination: "http://db_app:8000/shots/:path*",
      },
    ];
  },
};

export default nextConfig;
