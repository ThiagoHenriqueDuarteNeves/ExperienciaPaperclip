// Multi-user auth helpers (client-side).
//
// Profile LIST lives in localStorage (per-device). The PIN is validated by the
// backend, which returns a signed session token. Memory is shared by user_id
// across devices — typing the same name on phone and PC reaches the same memory.

export interface Profile {
  userId: string;
  displayName: string;
}

export interface Session {
  userId: string;
  displayName: string;
  token: string;
  expiresAt: number; // epoch ms
}

const PROFILES_KEY = "memory-profiles";
const SESSION_KEY = "memory-session";

// Token TTL on the backend is 720h; mirror a slightly shorter local expiry so
// we re-auth before the server rejects us.
const SESSION_TTL_MS = 715 * 60 * 60 * 1000;

/**
 * Base URL of the memory-api. In production NEXT_PUBLIC_CHAT_API_URL points at
 * ".../api/chat"; strip that to get the API root for /auth/* calls. In local
 * dev it is unset, so go straight to the backend (CORS is wildcard-open there).
 */
export function apiBase(): string {
  const chat = process.env.NEXT_PUBLIC_CHAT_API_URL;
  if (chat) return chat.replace(/\/api\/chat\/?$/, "");
  return process.env.NEXT_PUBLIC_MEMORY_API_URL || "http://localhost:8001";
}

const AUTH_HEADERS: Record<string, string> = {
  "Content-Type": "application/json",
  skip_zrok_interstitial: "true",
};

/** Derive a stable user_id from a display name (lowercase, no spaces/accents). */
export function slugify(name: string): string {
  return name
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "")
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9_-]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 64);
}

// -- localStorage: profiles list --

export function listProfiles(): Profile[] {
  try {
    const raw = window.localStorage.getItem(PROFILES_KEY);
    return raw ? (JSON.parse(raw) as Profile[]) : [];
  } catch {
    return [];
  }
}

export function addProfile(profile: Profile): void {
  try {
    const profiles = listProfiles().filter((p) => p.userId !== profile.userId);
    profiles.push(profile);
    window.localStorage.setItem(PROFILES_KEY, JSON.stringify(profiles));
  } catch { /* ignore */ }
}

// -- localStorage: session --

export function getSession(): Session | null {
  try {
    const raw = window.localStorage.getItem(SESSION_KEY);
    if (!raw) return null;
    const s = JSON.parse(raw) as Session;
    if (!s.token || Date.now() > s.expiresAt) {
      window.localStorage.removeItem(SESSION_KEY);
      return null;
    }
    return s;
  } catch {
    return null;
  }
}

export function setSession(s: Session): void {
  try { window.localStorage.setItem(SESSION_KEY, JSON.stringify(s)); } catch { /* ignore */ }
}

export function clearSession(): void {
  try { window.localStorage.removeItem(SESSION_KEY); } catch { /* ignore */ }
}

// -- backend calls --

interface AuthResponse {
  token: string;
  user_id: string;
  display_name: string;
}

async function authCall(path: string, payload: Record<string, unknown>): Promise<Session> {
  const res = await fetch(`${apiBase()}${path}`, {
    method: "POST",
    headers: AUTH_HEADERS,
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const data = await res.json();
      if (data?.detail) detail = String(data.detail);
    } catch { /* keep status */ }
    throw new Error(detail);
  }
  const data = (await res.json()) as AuthResponse;
  const session: Session = {
    userId: data.user_id,
    displayName: data.display_name || data.user_id,
    token: data.token,
    expiresAt: Date.now() + SESSION_TTL_MS,
  };
  setSession(session);
  addProfile({ userId: session.userId, displayName: session.displayName });
  return session;
}

/** Create a new profile (name → slug user_id) with a PIN. Throws on 409/error. */
export async function register(name: string, pin: string): Promise<Session> {
  const userId = slugify(name);
  if (!userId) throw new Error("Nome inválido");
  return authCall("/auth/register", { user_id: userId, pin, display_name: name.trim() });
}

/** Log into an existing profile by user_id + PIN. Throws on 401/error. */
export async function login(userId: string, pin: string): Promise<Session> {
  return authCall("/auth/login", { user_id: userId, pin });
}
