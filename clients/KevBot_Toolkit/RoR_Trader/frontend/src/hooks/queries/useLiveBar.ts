/**
 * useLiveBar — subscribes to Supabase Realtime broadcast channel for live bar
 * updates published by Ralph (see src/live_bar_publisher.py).
 *
 * Channel: `live_bars:{user_id}` (per-user broadcast)
 * Event:   `bar_update`
 * Payload: { symbol, tf_seconds, bar: { open, high, low, close, volume, timestamp }, is_forming }
 *
 * The hook returns the latest bar matching (symbol, tf_seconds), plus a
 * `receivedAt` timestamp used by the UI to render a "Live · / Not live"
 * status indicator.
 *
 * The hook is a no-op when the user is not authenticated yet — so callers
 * can mount it unconditionally without guarding.
 */

import { useEffect, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { supabase } from '@/lib/supabase';
import { apiFetch } from '@/lib/api/client';

export interface LiveBarPayload {
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  timestamp: string; // ISO
}

export interface LiveBarState {
  bar: LiveBarPayload;
  isForming: boolean;
  receivedAt: number; // epoch ms
}

interface MeResponse {
  id: string;
  email: string;
}

/**
 * Subscribe to live-bar broadcasts for a given (symbol, tfSeconds) pair.
 * Returns the most recent matching bar, or null if none received yet.
 */
export function useLiveBar(
  symbol: string | null | undefined,
  tfSeconds: number | null | undefined,
): LiveBarState | null {
  const [state, setState] = useState<LiveBarState | null>(null);

  // Resolve user_id via /api/auth/me (works under DEV_BYPASS_AUTH and real JWT)
  const { data: me } = useQuery<MeResponse>({
    queryKey: ['auth', 'me'],
    queryFn: () => apiFetch<MeResponse>('/api/auth/me'),
    staleTime: 60_000,
    retry: false,
  });
  const userId = me?.id;

  // Latest (symbol, tf) pair as a ref so the subscription handler
  // always uses the current filter without resubscribing on every change.
  const filterRef = useRef<{ symbol: string | null; tf: number | null }>({
    symbol: null,
    tf: null,
  });
  useEffect(() => {
    filterRef.current = {
      symbol: symbol ?? null,
      tf: tfSeconds ?? null,
    };
    // Clear stale state when the caller switches strategy/tf
    setState(null);
  }, [symbol, tfSeconds]);

  useEffect(() => {
    if (!userId) return;

    const channelName = `live_bars:${userId}`;
    let channel: ReturnType<typeof supabase.channel> | null = null;

    try {
      channel = supabase.channel(channelName);
      channel.on('broadcast', { event: 'bar_update' }, (msg: any) => {
        const payload = msg?.payload;
        if (!payload) return;
        const filter = filterRef.current;
        if (filter.symbol && payload.symbol !== filter.symbol) return;
        if (filter.tf != null && payload.tf_seconds !== filter.tf) return;
        setState({
          bar: payload.bar,
          isForming: Boolean(payload.is_forming),
          receivedAt: Date.now(),
        });
      });
      channel.subscribe();
    } catch (e) {
      // Supabase not configured — silently no-op
      // (logout/login or local dev without Realtime enabled)
    }

    return () => {
      if (channel) {
        try {
          supabase.removeChannel(channel);
        } catch {
          // ignore
        }
      }
    };
  }, [userId]);

  return state;
}
