import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // NÃO usar output: "standalone" no Vercel — ele espera o build padrão (.next)
  // e, com standalone, procura uma pasta "public" inexistente e falha o deploy.
  // standalone só é necessário para self-hosting em Docker.

  // Permite que dispositivos na rede local (celular) carreguem os recursos
  // de dev do Next.js. Sem isto, o JS do cliente é bloqueado por CORS e a
  // página renderiza mas NUNCA hidrata — botões e inputs ficam "mortos".
  // (usado apenas em `next dev`; ignorado em produção)
  allowedDevOrigins: ["192.168.1.15", "192.168.1.6"],

  // Fixa a raiz do Turbopack neste diretório (frontend/) para evitar o aviso
  // de "multiple lockfiles" quando há um package-lock.json na raiz do repo.
  turbopack: { root: __dirname },
};

export default nextConfig;
