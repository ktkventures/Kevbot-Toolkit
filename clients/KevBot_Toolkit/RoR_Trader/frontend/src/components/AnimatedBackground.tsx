'use client';

import { useEffect, useState } from 'react';

interface Orb {
  id: number;
  size: number;
  x: number;
  y: number;
  color: string;
  duration: number;
  delay: number;
  path: string;
}

const themeOrbs: Record<string, Orb[]> = {
  dark: [
    { id: 1, size: 600, x: 70, y: 25, color: 'rgba(33, 150, 243, 0.08)', duration: 40, delay: 0, path: 'float1' },
    { id: 2, size: 400, x: 30, y: 70, color: 'rgba(33, 150, 243, 0.05)', duration: 35, delay: 5, path: 'float2' },
  ],
  'cyberpunk-lion': [
    { id: 1, size: 800, x: 15, y: 75, color: 'rgba(255, 140, 0, 0.14)', duration: 20, delay: 0, path: 'float1' },
    { id: 2, size: 600, x: 80, y: 15, color: 'rgba(224, 64, 251, 0.10)', duration: 25, delay: 2, path: 'float3' },
    { id: 3, size: 500, x: 50, y: 50, color: 'rgba(255, 60, 0, 0.08)', duration: 18, delay: 4, path: 'float4' },
    { id: 4, size: 400, x: 35, y: 25, color: 'rgba(255, 200, 0, 0.06)', duration: 22, delay: 7, path: 'float2' },
  ],
  crystal: [
    { id: 1, size: 800, x: 25, y: 20, color: 'rgba(124, 157, 245, 0.15)', duration: 22, delay: 0, path: 'float1' },
    { id: 2, size: 700, x: 70, y: 60, color: 'rgba(179, 136, 245, 0.12)', duration: 28, delay: 2, path: 'float3' },
    { id: 3, size: 600, x: 50, y: 80, color: 'rgba(104, 216, 232, 0.10)', duration: 32, delay: 4, path: 'float4' },
    { id: 4, size: 500, x: 85, y: 25, color: 'rgba(160, 180, 255, 0.08)', duration: 26, delay: 6, path: 'float2' },
    { id: 5, size: 400, x: 10, y: 50, color: 'rgba(200, 160, 255, 0.06)', duration: 30, delay: 9, path: 'float1' },
  ],
  fortress: [
    { id: 1, size: 900, x: 50, y: 30, color: 'rgba(60, 160, 255, 0.18)', duration: 30, delay: 0, path: 'float1' },
    { id: 2, size: 700, x: 20, y: 70, color: 'rgba(40, 200, 240, 0.15)', duration: 25, delay: 3, path: 'float3' },
    { id: 3, size: 600, x: 80, y: 50, color: 'rgba(100, 120, 255, 0.12)', duration: 35, delay: 5, path: 'float4' },
    { id: 4, size: 500, x: 60, y: 80, color: 'rgba(60, 180, 255, 0.10)', duration: 28, delay: 8, path: 'float2' },
    { id: 5, size: 800, x: 40, y: 20, color: 'rgba(180, 220, 255, 0.14)', duration: 40, delay: 0, path: 'float2' },
    { id: 6, size: 400, x: 75, y: 15, color: 'rgba(100, 200, 255, 0.12)', duration: 22, delay: 12, path: 'float1' },
  ],
  pipboy: [
    { id: 1, size: 500, x: 50, y: 50, color: 'rgba(48, 232, 104, 0.04)', duration: 30, delay: 0, path: 'float2' },
    { id: 2, size: 300, x: 25, y: 30, color: 'rgba(48, 232, 104, 0.03)', duration: 25, delay: 5, path: 'float1' },
  ],
};

export default function AnimatedBackground() {
  const [theme, setTheme] = useState('dark');

  useEffect(() => {
    const checkTheme = () => {
      const t = document.documentElement.getAttribute('data-theme') || 'dark';
      setTheme(t);
    };
    checkTheme();
    const observer = new MutationObserver(checkTheme);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    return () => observer.disconnect();
  }, []);

  const orbs = themeOrbs[theme] || themeOrbs.dark;

  return (
    <>
      {/* Fixed background container — sits between html bg and content */}
      <div
        style={{
          position: 'fixed',
          inset: 0,
          zIndex: 0,
          overflow: 'hidden',
          pointerEvents: 'none',
        }}
      >
        {orbs.map((orb) => (
          <div
            key={`${theme}-${orb.id}`}
            style={{
              position: 'absolute',
              width: orb.size,
              height: orb.size,
              borderRadius: '50%',
              background: `radial-gradient(circle, ${orb.color} 0%, transparent 70%)`,
              left: `${orb.x}%`,
              top: `${orb.y}%`,
              transform: 'translate(-50%, -50%)',
              animation: `${orb.path} ${orb.duration}s ease-in-out ${orb.delay}s infinite alternate`,
              filter: 'blur(60px)',
              willChange: 'transform',
            }}
          />
        ))}
      </div>

      <style>{`
        @keyframes float1 {
          0% { transform: translate(-50%, -50%) translate(0px, 0px); }
          25% { transform: translate(-50%, -50%) translate(80px, -40px); }
          50% { transform: translate(-50%, -50%) translate(-60px, 60px); }
          75% { transform: translate(-50%, -50%) translate(100px, 30px); }
          100% { transform: translate(-50%, -50%) translate(-40px, -80px); }
        }

        @keyframes float2 {
          0% { transform: translate(-50%, -50%) translate(0px, 0px); }
          33% { transform: translate(-50%, -50%) translate(-100px, 50px); }
          66% { transform: translate(-50%, -50%) translate(60px, -70px); }
          100% { transform: translate(-50%, -50%) translate(-30px, 90px); }
        }

        @keyframes float3 {
          0% { transform: translate(-50%, -50%) translate(0px, 0px) scale(1); }
          50% { transform: translate(-50%, -50%) translate(-80px, -50px) scale(1.15); }
          100% { transform: translate(-50%, -50%) translate(70px, 40px) scale(0.9); }
        }

        @keyframes float4 {
          0% { transform: translate(-50%, -50%) translate(0px, 0px) scale(1); opacity: 0.7; }
          33% { transform: translate(-50%, -50%) translate(90px, -30px) scale(1.1); opacity: 1; }
          66% { transform: translate(-50%, -50%) translate(-70px, 60px) scale(0.95); opacity: 0.8; }
          100% { transform: translate(-50%, -50%) translate(40px, -50px) scale(1.05); opacity: 0.6; }
        }
      `}</style>
    </>
  );
}
