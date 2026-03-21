import type { Metadata } from "next";
import "./globals.css";
import Sidebar from "@/components/Sidebar";
import ThemeLoader from "@/components/ThemeLoader";
import AnimatedBackground from "@/components/AnimatedBackground";
import CRTOverlay from "@/components/CRTOverlay";

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
        <ThemeLoader />
        <AnimatedBackground />
        <CRTOverlay />
        <Sidebar />
        <main
          className="min-h-screen"
          style={{
            marginLeft: 'var(--sidebar-width)',
            padding: '24px 32px',
            position: 'relative',
            zIndex: 1,
          }}
        >
          {children}
        </main>
      </body>
    </html>
  );
}
