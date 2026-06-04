"use client";

import { useState } from "react";
import {
  listProfiles,
  login,
  register,
  type Profile,
  type Session,
} from "../lib/auth";

type Mode = { kind: "list" } | { kind: "pin"; profile: Profile } | { kind: "new" };

export default function ProfileGate({
  onAuthenticated,
}: {
  onAuthenticated: (s: Session) => void;
}) {
  const [profiles] = useState<Profile[]>(() => listProfiles());
  const [mode, setMode] = useState<Mode>(() =>
    listProfiles().length > 0 ? { kind: "list" } : { kind: "new" },
  );
  const [pin, setPin] = useState("");
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function doLogin(profile: Profile) {
    setBusy(true);
    setError("");
    try {
      const s = await login(profile.userId, pin);
      onAuthenticated(s);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Falha ao entrar");
    } finally {
      setBusy(false);
    }
  }

  async function doRegister() {
    setBusy(true);
    setError("");
    try {
      const s = await register(name, pin);
      onAuthenticated(s);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Falha ao criar";
      setError(msg.includes("taken") ? "Esse nome já existe — entre com o PIN." : msg);
    } finally {
      setBusy(false);
    }
  }

  const card: React.CSSProperties = {
    width: "100%", maxWidth: 360, display: "flex", flexDirection: "column",
    gap: 14, padding: 24, borderRadius: 16, background: "var(--surface)",
    border: "1px solid var(--border)", boxShadow: "var(--shadow-md)",
  };
  const input: React.CSSProperties = {
    width: "100%", padding: "12px 14px", borderRadius: 10, fontSize: 16,
    border: "1.5px solid var(--input-border)", background: "var(--input-bg)",
    color: "var(--text)", outline: "none",
  };
  const primaryBtn: React.CSSProperties = {
    width: "100%", padding: "12px", borderRadius: 10, border: "none",
    background: "var(--btn-primary)", color: "#fff", fontSize: 15, fontWeight: 600,
    cursor: "pointer", opacity: busy ? 0.6 : 1,
  };
  const linkBtn: React.CSSProperties = {
    background: "none", border: "none", color: "var(--btn-primary)",
    fontSize: 13, cursor: "pointer", padding: 4,
  };

  return (
    <div style={{
      display: "flex", flexDirection: "column", alignItems: "center",
      justifyContent: "center", height: "100dvh", padding: 20,
      background: "var(--surface-alt)",
    }}>
      <div style={card}>
        <div style={{ textAlign: "center" }}>
          <div style={{ fontSize: 40 }}>🧠</div>
          <h1 style={{ margin: "8px 0 2px", fontSize: 18, color: "var(--text)" }}>Memory Chat</h1>
          <p style={{ margin: 0, fontSize: 12, color: "var(--text-muted)" }}>
            {mode.kind === "new" ? "Criar um perfil" : mode.kind === "pin" ? `Entrar como ${mode.profile.displayName}` : "Quem é você?"}
          </p>
        </div>

        {mode.kind === "list" && (
          <>
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {profiles.map((p) => (
                <button
                  key={p.userId}
                  type="button"
                  onClick={() => { setPin(""); setError(""); setMode({ kind: "pin", profile: p }); }}
                  style={{
                    display: "flex", alignItems: "center", gap: 10, padding: "12px 14px",
                    borderRadius: 10, border: "1px solid var(--border)", background: "var(--surface-alt)",
                    color: "var(--text)", fontSize: 15, cursor: "pointer", textAlign: "left",
                  }}
                >
                  <span style={{ fontSize: 20 }}>👤</span> {p.displayName}
                </button>
              ))}
            </div>
            <button type="button" style={linkBtn} onClick={() => { setName(""); setPin(""); setError(""); setMode({ kind: "new" }); }}>
              + Criar novo perfil
            </button>
          </>
        )}

        {mode.kind === "pin" && (
          <>
            <input
              style={input} type="password" inputMode="numeric" autoFocus
              placeholder="PIN" value={pin}
              onChange={(e) => setPin(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter" && pin.length >= 4 && !busy) doLogin(mode.profile); }}
            />
            <button type="button" style={primaryBtn} disabled={busy || pin.length < 4} onClick={() => doLogin(mode.profile)}>
              {busy ? "Entrando…" : "Entrar"}
            </button>
            <button type="button" style={linkBtn} onClick={() => { setError(""); setMode({ kind: "list" }); }}>
              ← Voltar
            </button>
          </>
        )}

        {mode.kind === "new" && (
          <>
            <input
              style={input} type="text" autoFocus placeholder="Seu nome"
              value={name} onChange={(e) => setName(e.target.value)}
            />
            <input
              style={input} type="password" inputMode="numeric"
              placeholder="Crie um PIN (mín. 4 dígitos)" value={pin}
              onChange={(e) => setPin(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter" && name.trim() && pin.length >= 4 && !busy) doRegister(); }}
            />
            <button type="button" style={primaryBtn} disabled={busy || !name.trim() || pin.length < 4} onClick={doRegister}>
              {busy ? "Criando…" : "Criar perfil"}
            </button>
            {profiles.length > 0 && (
              <button type="button" style={linkBtn} onClick={() => { setError(""); setMode({ kind: "list" }); }}>
                ← Já tenho um perfil
              </button>
            )}
          </>
        )}

        {error && (
          <p style={{ margin: 0, fontSize: 13, color: "var(--btn-stop)", textAlign: "center" }}>{error}</p>
        )}
      </div>
    </div>
  );
}
