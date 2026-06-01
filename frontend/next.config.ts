import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  // Permite que dispositivos na rede local (celular) carreguem os recursos
  // de dev do Next.js. Sem isto, o JS do cliente é bloqueado por CORS e a
  // página renderiza mas NUNCA hidrata — botões e inputs ficam "mortos".
  allowedDevOrigins: ["192.168.1.15", "192.168.1.6"],
};

export default nextConfig;
