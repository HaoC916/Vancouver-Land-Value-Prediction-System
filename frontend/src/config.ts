// Single source of truth for the backend API base URL.
// Set VITE_API_BASE per environment (see .env.development for local dev, and the
// Vercel project's Environment Variables for production). Trailing slashes are
// trimmed so callers can safely write `${API_BASE}/predict`.
const raw = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

export const API_BASE = raw.replace(/\/+$/, "");
