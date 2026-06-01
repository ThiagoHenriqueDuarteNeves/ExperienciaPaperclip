import "./globals.css";

export const metadata = {
  title: "Memory Chat",
  description: "Chat with persistent memory across conversations",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="pt-BR">
      <body>{children}</body>
    </html>
  );
}
