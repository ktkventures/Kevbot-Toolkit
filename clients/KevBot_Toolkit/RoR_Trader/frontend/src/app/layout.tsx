import type { Metadata } from "next";
import "./globals.css";
import Sidebar from "@/components/Sidebar";

export const metadata: Metadata = {
  title: "RoR Trader",
  description: "Rate of Return Trading System",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">
        <Sidebar />
        <main
          className="min-h-screen"
          style={{
            marginLeft: 'var(--sidebar-width)',
            padding: '24px 32px',
          }}
        >
          {children}
        </main>
      </body>
    </html>
  );
}
