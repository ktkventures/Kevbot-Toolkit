'use client';

import { createContext, useContext, useEffect, useState, useCallback, ReactNode } from 'react';
import { Session, User } from '@supabase/supabase-js';
import { supabase } from '@/lib/supabase';

interface AuthContextType {
  user: User | null;
  session: Session | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType>({
  user: null,
  session: null,
  loading: true,
  login: async () => {},
  logout: async () => {},
});

export function useAuth() {
  return useContext(AuthContext);
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState(true);

  // Sync tokens to localStorage for apiFetch
  const syncTokens = useCallback((sess: Session | null) => {
    if (sess) {
      localStorage.setItem('ror_access_token', sess.access_token);
      localStorage.setItem('ror_refresh_token', sess.refresh_token);
    } else {
      localStorage.removeItem('ror_access_token');
      localStorage.removeItem('ror_refresh_token');
    }
  }, []);

  useEffect(() => {
    // Skip Supabase init if not configured (local dev without auth)
    const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
    if (!supabaseUrl) {
      setLoading(false);
      return;
    }

    // Check for existing session on mount
    supabase.auth.getSession().then(({ data: { session: existingSession } }) => {
      setSession(existingSession);
      setUser(existingSession?.user ?? null);
      syncTokens(existingSession);
      setLoading(false);
    }).catch(() => {
      setLoading(false);
    });

    // Listen for auth state changes (token refresh, logout, etc.)
    let subscription: { unsubscribe: () => void } | null = null;
    try {
      const { data } = supabase.auth.onAuthStateChange(
        (_event, newSession) => {
          setSession(newSession);
          setUser(newSession?.user ?? null);
          syncTokens(newSession);
        }
      );
      subscription = data.subscription;
    } catch {
      // Supabase not available
    }

    return () => {
      subscription?.unsubscribe();
    };
  }, [syncTokens]);

  const login = useCallback(async (email: string, password: string) => {
    if (!process.env.NEXT_PUBLIC_SUPABASE_URL) {
      throw new Error('Supabase not configured. Set NEXT_PUBLIC_SUPABASE_URL in .env.local');
    }
    const { data, error } = await supabase.auth.signInWithPassword({ email, password });
    if (error) throw error;
    setSession(data.session);
    setUser(data.user);
    syncTokens(data.session);
  }, [syncTokens]);

  const logout = useCallback(async () => {
    await supabase.auth.signOut();
    setSession(null);
    setUser(null);
    syncTokens(null);
  }, [syncTokens]);

  return (
    <AuthContext.Provider value={{ user, session, loading, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}
