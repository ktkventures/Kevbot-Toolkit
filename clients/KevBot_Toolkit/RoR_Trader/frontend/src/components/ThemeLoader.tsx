'use client';

import { useEffect } from 'react';

export default function ThemeLoader() {
  useEffect(() => {
    const saved = localStorage.getItem('ror-theme');
    if (saved) {
      document.documentElement.setAttribute('data-theme', saved);
    }
  }, []);

  return null;
}
