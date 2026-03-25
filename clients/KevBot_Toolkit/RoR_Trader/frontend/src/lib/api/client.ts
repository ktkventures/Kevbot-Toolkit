/**
 * API fetch wrapper with JWT authentication.
 *
 * Attaches the Supabase access token as a Bearer header on every request.
 * On 401, attempts a token refresh before retrying once.
 */

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export class ApiError extends Error {
  status: number;
  body: string;

  constructor(status: number, body: string) {
    super(`API error ${status}: ${body}`);
    this.status = status;
    this.body = body;
  }
}

/**
 * Get the stored access token from localStorage.
 */
function getToken(): string | null {
  if (typeof window === 'undefined') return null;
  return localStorage.getItem('ror_access_token');
}

/**
 * Attempt to refresh the session using the Supabase client.
 * Returns true if refresh succeeded and a new token is stored.
 */
async function refreshToken(): Promise<boolean> {
  try {
    const { supabase } = await import('@/lib/supabase');
    const { data, error } = await supabase.auth.refreshSession();
    if (error || !data.session) return false;

    localStorage.setItem('ror_access_token', data.session.access_token);
    localStorage.setItem('ror_refresh_token', data.session.refresh_token);
    return true;
  } catch {
    return false;
  }
}

/**
 * Typed fetch wrapper for API calls.
 *
 * Usage:
 *   const strategies = await apiFetch<Strategy[]>('/api/strategies');
 *   await apiFetch('/api/strategies', { method: 'POST', body: JSON.stringify(data) });
 */
export async function apiFetch<T = unknown>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(options.headers as Record<string, string> || {}),
  };

  const token = getToken();
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const res = await fetch(`${API_URL}${path}`, { ...options, headers });

  // 401 — try refresh once
  if (res.status === 401) {
    const refreshed = await refreshToken();
    if (refreshed) {
      const newToken = getToken();
      if (newToken) {
        headers['Authorization'] = `Bearer ${newToken}`;
      }
      const retry = await fetch(`${API_URL}${path}`, { ...options, headers });
      if (!retry.ok) {
        throw new ApiError(retry.status, await retry.text());
      }
      return retry.json() as Promise<T>;
    }

    // Refresh failed — redirect to login
    if (typeof window !== 'undefined') {
      window.location.href = '/login';
    }
    throw new ApiError(401, 'Session expired');
  }

  if (!res.ok) {
    throw new ApiError(res.status, await res.text());
  }

  // Handle 204 No Content
  if (res.status === 204) {
    return undefined as T;
  }

  return res.json() as Promise<T>;
}
