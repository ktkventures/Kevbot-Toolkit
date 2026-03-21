'use client';

import { useEffect, useState, useCallback } from 'react';

export default function CRTOverlay() {
  const [theme, setTheme] = useState('dark');
  const [flash, setFlash] = useState(false);

  useEffect(() => {
    const check = () => setTheme(document.documentElement.getAttribute('data-theme') || 'dark');
    check();
    const observer = new MutationObserver(check);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    return () => observer.disconnect();
  }, []);

  // Lightning flash effect
  useEffect(() => {
    if (theme !== 'lightning') return;
    const scheduleFlash = () => {
      const delay = 5000 + Math.random() * 12000; // 5-17 seconds
      return setTimeout(() => {
        setFlash(true);
        // Double strike effect
        setTimeout(() => setFlash(false), 80);
        setTimeout(() => setFlash(true), 150);
        setTimeout(() => setFlash(false), 200);
        // Occasional triple strike
        if (Math.random() > 0.6) {
          setTimeout(() => setFlash(true), 350);
          setTimeout(() => setFlash(false), 400);
        }
        scheduleFlash();
      }, delay);
    };
    const timer = scheduleFlash();
    return () => clearTimeout(timer);
  }, [theme]);

  return (
    <>
      {/* ===== PIP-BOY: Scanlines + CRT ===== */}
      {theme === 'pipboy' && (
        <>
          <div style={{
            position: 'fixed', inset: 0, zIndex: 99999, pointerEvents: 'none',
            background: `repeating-linear-gradient(0deg, rgba(0,0,0,0.15) 0px, rgba(0,0,0,0.15) 1px, transparent 1px, transparent 4px)`,
          }} />
          <div style={{
            position: 'fixed', inset: 0, zIndex: 99998, pointerEvents: 'none',
            background: 'radial-gradient(ellipse at center, transparent 50%, rgba(0,0,0,0.5) 100%)',
            animation: 'crtFlicker 3s ease-in-out infinite',
          }} />
        </>
      )}

      {/* ===== NEON TOKYO: Rain + Neon glow ===== */}
      {theme === 'neon-tokyo' && (
        <>
          {/* Rain streaks — bold neon rain */}
          <div style={{
            position: 'fixed', inset: 0, zIndex: 99997, pointerEvents: 'none',
            overflow: 'hidden',
          }}>
            {Array.from({ length: 80 }).map((_, i) => {
              const left = (i * 1.25 + (i % 3) * 7.5) % 100;
              const height = 30 + (i % 5) * 15;
              const speed = 0.6 + (i % 4) * 0.2;
              const delay = (i * 0.15) % 2.5;
              const opacity = 0.4 + (i % 3) * 0.2;
              const pink = i % 3 === 0;
              return (
                <div
                  key={i}
                  style={{
                    position: 'absolute',
                    left: `${left}%`,
                    top: '-5%',
                    width: pink ? '2px' : '1px',
                    height: `${height}px`,
                    background: pink
                      ? `linear-gradient(to bottom, transparent, rgba(255, 48, 144, 0.5), rgba(255, 48, 144, 0.2), transparent)`
                      : `linear-gradient(to bottom, transparent, rgba(48, 192, 255, 0.3), rgba(48, 192, 255, 0.1), transparent)`,
                    animation: `rainFall ${speed}s linear ${delay}s infinite`,
                    opacity,
                  }}
                />
              );
            })}
          </div>
          {/* Neon sign pulse on sidebar edge */}
          <div style={{
            position: 'fixed', left: 236, top: 0, bottom: 0, width: 4, zIndex: 99998, pointerEvents: 'none',
            background: 'rgba(255, 48, 144, 0.4)',
            boxShadow: '0 0 15px rgba(255, 48, 144, 0.3), 0 0 30px rgba(255, 48, 144, 0.15)',
            animation: 'neonPulse 2s ease-in-out infinite',
          }} />
        </>
      )}

      {/* ===== LIGHTNING STORM: Stars + Clouds + Flash ===== */}
      {theme === 'lightning' && (
        <>
          {/* Star field */}
          <div style={{
            position: 'fixed', inset: 0, zIndex: 99995, pointerEvents: 'none',
          }}>
            {Array.from({ length: 60 }).map((_, i) => (
              <div
                key={`star-${i}`}
                style={{
                  position: 'absolute',
                  left: `${(i * 1.7 + (i % 7) * 13) % 100}%`,
                  top: `${(i * 2.3 + (i % 5) * 17) % 100}%`,
                  width: i % 5 === 0 ? 2 : 1,
                  height: i % 5 === 0 ? 2 : 1,
                  borderRadius: '50%',
                  background: 'rgba(200, 210, 255, 0.7)',
                  animation: `sparkle ${3 + (i % 4) * 1.5}s ease-in-out ${(i * 0.3) % 5}s infinite`,
                }}
              />
            ))}
          </div>
          {/* Cloud layer — dark drifting shapes */}
          <div style={{
            position: 'fixed', inset: 0, zIndex: 99996, pointerEvents: 'none', overflow: 'hidden',
          }}>
            {Array.from({ length: 5 }).map((_, i) => (
              <div
                key={`cloud-${i}`}
                style={{
                  position: 'absolute',
                  left: `${i * 20}%`,
                  top: `${5 + i * 8}%`,
                  width: 600 + i * 100,
                  height: 150 + i * 30,
                  borderRadius: '50%',
                  background: `radial-gradient(ellipse, rgba(15, 12, 35, ${0.5 + i * 0.05}) 0%, transparent 70%)`,
                  filter: 'blur(30px)',
                  animation: `float${(i % 4) + 1} ${40 + i * 8}s ease-in-out ${i * 3}s infinite alternate`,
                }}
              />
            ))}
          </div>
          {/* Lightning flash overlay */}
          <div style={{
            position: 'fixed', inset: 0, zIndex: 99999, pointerEvents: 'none',
            background: flash ? 'rgba(200, 210, 255, 0.3)' : 'transparent',
            transition: flash ? 'none' : 'background 0.3s ease-out',
          }} />
          {/* Storm vignette */}
          <div style={{
            position: 'fixed', inset: 0, zIndex: 99997, pointerEvents: 'none',
            background: 'radial-gradient(ellipse at center, transparent 30%, rgba(6, 5, 15, 0.5) 100%)',
          }} />
          {/* Lightning bolt (appears during flash) */}
          {flash && (
            <svg style={{
              position: 'fixed', zIndex: 99998, pointerEvents: 'none',
              left: `${20 + Math.random() * 60}%`, top: 0, width: 100, height: '50%',
              opacity: 0.6,
            }}>
              <path
                d={`M50,0 L${45 + Math.random() * 10},${80 + Math.random() * 40} L${55 + Math.random() * 10},${100 + Math.random() * 20} L${40 + Math.random() * 20},${200 + Math.random() * 100}`}
                stroke="rgba(180, 200, 255, 0.8)"
                strokeWidth="2"
                fill="none"
                filter="url(#lightningGlow)"
              />
              <defs>
                <filter id="lightningGlow">
                  <feGaussianBlur stdDeviation="4" result="blur" />
                  <feMerge>
                    <feMergeNode in="blur" />
                    <feMergeNode in="SourceGraphic" />
                  </feMerge>
                </filter>
              </defs>
            </svg>
          )}
        </>
      )}

      {/* ===== KAWAII: Floating bubbles + sparkles ===== */}
      {theme === 'kawaii' && (
        <div style={{
          position: 'fixed', inset: 0, zIndex: 99997, pointerEvents: 'none', overflow: 'hidden',
        }}>
          {/* Bubbles */}
          {Array.from({ length: 15 }).map((_, i) => {
            const colors = ['rgba(255,150,200,0.2)', 'rgba(200,150,255,0.15)', 'rgba(150,220,200,0.15)', 'rgba(255,200,150,0.15)'];
            const size = 20 + Math.random() * 40;
            return (
              <div
                key={i}
                style={{
                  position: 'absolute',
                  left: `${Math.random() * 100}%`,
                  bottom: `-${size}px`,
                  width: size,
                  height: size,
                  borderRadius: '50%',
                  background: colors[i % colors.length],
                  border: `1px solid ${colors[i % colors.length].replace('0.2', '0.3').replace('0.15', '0.25')}`,
                  animation: `bubbleFloat ${8 + Math.random() * 12}s ease-in-out ${Math.random() * 10}s infinite`,
                }}
              />
            );
          })}
          {/* Sparkles */}
          {Array.from({ length: 20 }).map((_, i) => (
            <div
              key={`sparkle-${i}`}
              style={{
                position: 'absolute',
                left: `${Math.random() * 100}%`,
                top: `${Math.random() * 100}%`,
                width: 3,
                height: 3,
                borderRadius: '50%',
                background: 'rgba(255, 200, 240, 0.6)',
                boxShadow: '0 0 6px rgba(255, 150, 220, 0.4)',
                animation: `sparkle ${2 + Math.random() * 3}s ease-in-out ${Math.random() * 5}s infinite`,
              }}
            />
          ))}
        </div>
      )}

      {/* ===== TRON: Grid floor + scan sweep ===== */}
      {theme === 'tron' && (
        <>
          {/* Perspective grid */}
          <div style={{
            position: 'fixed', left: 240, right: 0, bottom: 0, height: '40%',
            zIndex: 0, pointerEvents: 'none',
            background: `
              linear-gradient(rgba(0, 220, 255, 0.06) 1px, transparent 1px),
              linear-gradient(90deg, rgba(0, 220, 255, 0.06) 1px, transparent 1px)
            `,
            backgroundSize: '60px 40px',
            transform: 'perspective(500px) rotateX(60deg)',
            transformOrigin: 'bottom center',
            maskImage: 'linear-gradient(to top, rgba(0,0,0,0.6) 0%, transparent 100%)',
            WebkitMaskImage: 'linear-gradient(to top, rgba(0,0,0,0.6) 0%, transparent 100%)',
            animation: 'gridScroll 4s linear infinite',
          }} />
          {/* Horizontal scan sweep */}
          <div style={{
            position: 'fixed', inset: 0, zIndex: 99997, pointerEvents: 'none', overflow: 'hidden',
          }}>
            <div style={{
              position: 'absolute', left: 0, right: 0, height: 2,
              background: 'linear-gradient(90deg, transparent, rgba(0, 220, 255, 0.4), rgba(0, 220, 255, 0.6), rgba(0, 220, 255, 0.4), transparent)',
              boxShadow: '0 0 20px rgba(0, 220, 255, 0.3), 0 0 40px rgba(0, 220, 255, 0.15)',
              animation: 'scanSweep 6s linear infinite',
            }} />
          </div>
        </>
      )}

      <style>{`
        @keyframes crtFlicker {
          0%, 92%, 100% { opacity: 1; }
          93% { opacity: 0.7; }
          94% { opacity: 0.9; }
          95% { opacity: 0.75; }
          96% { opacity: 0.95; }
        }

        @keyframes rainFall {
          0% { transform: translateY(-100vh); }
          100% { transform: translateY(120vh); }
        }

        @keyframes neonPulse {
          0%, 100% { opacity: 0.6; }
          50% { opacity: 1; }
        }

        @keyframes bubbleFloat {
          0% { transform: translateY(0) translateX(0) scale(1); opacity: 0; }
          10% { opacity: 1; }
          90% { opacity: 1; }
          100% { transform: translateY(-110vh) translateX(30px) scale(0.5); opacity: 0; }
        }

        @keyframes sparkle {
          0%, 100% { opacity: 0; transform: scale(0.5); }
          50% { opacity: 1; transform: scale(1.5); }
        }

        @keyframes gridScroll {
          0% { background-position: 0 0; }
          100% { background-position: 0 40px; }
        }

        @keyframes scanSweep {
          0% { top: -10px; }
          100% { top: 100vh; }
        }
      `}</style>
    </>
  );
}
