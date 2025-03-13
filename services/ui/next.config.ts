import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/db-api/:path*",
        destination: "http://event_app:8000/:path*",
      },
      {
        source: "/model-api/:path*",
        destination: "http://model_app:8001/:path*",
      },
      {
        source: "/backend-api/:path*",
        destination: "http://data_app:8002/:path*",
      },
    ];
  },
};

export default nextConfig;
