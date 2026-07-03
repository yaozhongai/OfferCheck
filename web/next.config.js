/** @type {import('next').NextConfig} */
const NEXA_API = process.env.NEXA_API_BASE || "http://127.0.0.1:8000";

const nextConfig = {
  // 开发期把 /api/v0/* 代理到后端，避免 CORS，同时前端代码用相对路径
  async rewrites() {
    return [
      { source: "/api/v0/:path*", destination: `${NEXA_API}/api/v0/:path*` },
    ];
  },
};

module.exports = nextConfig;
