'use client';

import { useEffect, useState } from 'react';

export default function CRTOverlay() {
  const [theme, setTheme] = useState('dark');

  useEffect(() => {
    const check = () => setTheme(document.documentElement.getAttribute('data-theme') || 'dark');
    check();
    const observer = new MutationObserver(check);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    return () => observer.disconnect();
  }, []);

  if (theme !== 'pipboy') return null;

  return (
    <>
      {/* Scanlines */}
      <div
        style={{
          position: 'fixed',
          inset: 0,
          zIndex: 99999,
          pointerEvents: 'none',
          background: `repeating-linear-gradient(
            0deg,
            rgba(0, 0, 0, 0.15) 0px,
            rgba(0, 0, 0, 0.15) 1px,
            transparent 1px,
            transparent 4px
          )`,
        }}
      />

      {/* CRT vignette — dark edges like an old monitor */}
      <div
        style={{
          position: 'fixed',
          inset: 0,
          zIndex: 99998,
          pointerEvents: 'none',
          background: 'radial-gradient(ellipse at center, transparent 50%, rgba(0, 0, 0, 0.5) 100%)',
          animation: 'crtFlicker 3s ease-in-out infinite',
        }}
      />

      {/* Green ambient screen glow */}
      <div
        style={{
          position: 'fixed',
          inset: 0,
          zIndex: 99997,
          pointerEvents: 'none',
          background: 'radial-gradient(ellipse at center, rgba(48, 232, 104, 0.03) 0%, transparent 70%)',
        }}
      />

      <style>{`
        @keyframes crtFlicker {
          0%, 92%, 100% { opacity: 1; }
          93% { opacity: 0.7; }
          94% { opacity: 0.9; }
          95% { opacity: 0.75; }
          96% { opacity: 0.95; }
        }
      `}</style>
    </>
  );
}
