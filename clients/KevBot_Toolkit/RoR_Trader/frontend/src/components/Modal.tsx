'use client';

import { useEffect, useState } from 'react';
import { createPortal } from 'react-dom';

interface ModalProps {
  title: string;
  isOpen: boolean;
  onClose: () => void;
  width?: string;
  children: React.ReactNode;
}

export default function Modal({ title, isOpen, onClose, width = '640px', children }: ModalProps) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => { document.body.style.overflow = ''; };
  }, [isOpen]);

  if (!isOpen || !mounted) return null;

  return createPortal(
    <div
      className="fixed inset-0 z-50 flex items-center justify-center"
      style={{ background: 'rgba(0, 0, 0, 0.6)', backdropFilter: 'blur(4px)' }}
      onClick={onClose}
    >
      <div
        className="rounded-xl border overflow-hidden flex flex-col"
        style={{
          background: 'var(--bg-secondary)',
          borderColor: 'var(--border)',
          width,
          maxWidth: '90vw',
          maxHeight: '85vh',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div
          className="flex items-center justify-between px-6 py-4 border-b"
          style={{ borderColor: 'var(--border)' }}
        >
          <h2 className="text-lg font-semibold">{title}</h2>
          <button
            onClick={onClose}
            className="w-8 h-8 rounded-lg flex items-center justify-center text-lg transition-colors"
            style={{ color: 'var(--text-muted)', background: 'transparent' }}
            onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--bg-card)')}
            onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
          >
            x
          </button>
        </div>

        {/* Body */}
        <div className="px-6 py-5 overflow-y-auto flex-1">
          {children}
        </div>
      </div>
    </div>,
    document.body,
  );
}
