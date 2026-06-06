import "./globals.css";
import { Outfit, JetBrains_Mono } from "next/font/google";

const outfit = Outfit({
  subsets: ["latin"],
  variable: "--outfit",
  display: "swap",
});

const mono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--jetbrains",
  weight: ["400", "500"],
  display: "swap",
});

export const metadata = {
  title: "Memory Chat",
  description: "Chat com memória persistente",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="pt-BR" className={`${outfit.variable} ${mono.variable}`}>
      <body>{children}</body>
    </html>
  );
}
