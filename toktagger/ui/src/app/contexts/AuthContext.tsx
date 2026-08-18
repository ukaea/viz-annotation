"use client";
import React, {
  createContext,
  useContext,
  useState,
  useEffect,
  ReactNode,
} from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { BACKEND_API_URL } from "@/app/core";
import type { CurrentUser } from "@/types";

const TOKEN_KEY = "tt_access_token";

interface AuthContextType {
  user: CurrentUser | null;
  token: string | null;
  isLoading: boolean;
  login: (username: string, password: string) => Promise<void>;
  logout: () => void;
  refreshUser: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<CurrentUser | null>(null);
  const [token, setToken] = useState<string | null>(() =>
    localStorage.getItem(TOKEN_KEY),
  );
  const [isLoading, setIsLoading] = useState(true);
  const navigate = useNavigate();
  const location = useLocation();

  const refreshUser = async () => {
    const stored = localStorage.getItem(TOKEN_KEY);
    if (!stored) return;
    const res = await fetch(`${BACKEND_API_URL}/auth/me`, {
      headers: { Authorization: `Bearer ${stored}` },
    });
    if (res.ok) {
      setUser((await res.json()) as CurrentUser);
    }
  };

  // Force a password change before anything else - an admin knows the password they
  // just set on a new account, so the new owner must replace it on first login.
  useEffect(() => {
    if (user?.must_change_password && location.pathname !== "/ui/profile") {
      navigate("/ui/profile", { replace: true });
    }
  }, [user, location.pathname, navigate]);

  // Validate stored token on mount. Always calls /auth/me since auth is required.
  useEffect(() => {
    const validate = async () => {
      const stored = localStorage.getItem(TOKEN_KEY);
      const headers: HeadersInit = stored
        ? { Authorization: `Bearer ${stored}` }
        : {};
      try {
        const res = await fetch(`${BACKEND_API_URL}/auth/me`, { headers });
        if (res.ok) {
          const data = await res.json();
          setUser(data as CurrentUser);
          setToken(stored);
        } else {
          if (stored) localStorage.removeItem(TOKEN_KEY);
          setToken(null);
          setUser(null);
        }
      } catch {
        if (stored) localStorage.removeItem(TOKEN_KEY);
        setToken(null);
        setUser(null);
      } finally {
        setIsLoading(false);
      }
    };
    validate();
  }, []);

  const login = async (username: string, password: string) => {
    const body = new URLSearchParams({ username, password });
    const res = await fetch(`${BACKEND_API_URL}/auth/token`, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: body.toString(),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data?.detail ?? "Login failed");
    }
    const { access_token } = await res.json();
    localStorage.setItem(TOKEN_KEY, access_token);
    setToken(access_token);

    // Fetch user profile
    const meRes = await fetch(`${BACKEND_API_URL}/auth/me`, {
      headers: { Authorization: `Bearer ${access_token}` },
    });
    if (!meRes.ok) {
      localStorage.removeItem(TOKEN_KEY);
      setToken(null);
      throw new Error("Login failed: could not load user profile");
    }
    setUser((await meRes.json()) as CurrentUser);
  };

  const logout = () => {
    localStorage.removeItem(TOKEN_KEY);
    setToken(null);
    setUser(null);
    navigate("/ui/login");
  };

  return (
    <AuthContext.Provider
      value={{ user, token, isLoading, login, logout, refreshUser }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
