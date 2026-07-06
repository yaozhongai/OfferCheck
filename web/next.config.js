/** @type {import('next').NextConfig} */
const NEXA_API = process.env.NEXA_API_BASE || "http://127.0.0.1:8000";

// 生产（Docker 构建时设 NEXT_OUTPUT_EXPORT=true）：静态导出到 out/，由 FastAPI
// 同源托管，前端直连 /api/v0/*，无需代理 / CORS。
// 本地开发（不设该变量）：保留 rewrites 代理，/api/v0/* 转发到后端 :8000。
const isExport = process.env.NEXT_OUTPUT_EXPORT === "true";

const nextConfig = isExport
  ? { output: "export" }
  : {
      async rewrites() {
        return [
          { source: "/api/v0/:path*", destination: `${NEXA_API}/api/v0/:path*` },
        ];
      },
    };

module.exports = nextConfig;
