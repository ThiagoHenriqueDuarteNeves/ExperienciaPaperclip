"use client";

import { useState } from "react";
import {
  listProfiles, login, register, slugify,
  type Profile, type Session,
} from "../lib/auth";

type Mode = { kind: "list" } | { kind: "pin"; profile: Profile } | { kind: "new" };

const GLASS_CARD: React.CSSProperties = {
  background: "rgba(8, 6, 28, 0.68)",
  backdropFilter: "blur(28px)",
  WebkitBackdropFilter: "blur(28px)",
  border: "1px solid rgba(255, 255, 255, 0.1)",
  boxShadow: "0 24px 60px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.08)",
};

const inputStyle: React.CSSProperties = {
  width: "100%", padding: "12px 14px", borderRadius: 10,
  border: "1px solid rgba(255, 255, 255, 0.1)",
  background: "rgba(255, 255, 255, 0.05)",
  backdropFilter: "blur(8px)", WebkitBackdropFilter: "blur(8px)",
  color: "#dde0ff", fontSize: 16, outline: "none",
  transition: "border-color 0.15s",
};

const linkBtnStyle: React.CSSProperties = {
  background: "none", border: "none",
  color: "rgba(140, 140, 200, 0.6)", fontSize: 13,
  cursor: "pointer", padding: "4px", textAlign: "center",
  letterSpacing: "0.01em",
};

export default function ProfileGate({ onAuthenticated }: { onAuthenticated: (s: Session) => void }) {
  const [profiles] = useState<Profile[]>(() => listProfiles());
  const [mode, setMode] = useState<Mode>(() =>
    listProfiles().length > 0 ? { kind: "list" } : { kind: "new" });
  const [pin, setPin] = useState("");
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function doLogin(profile: Profile) {
    setBusy(true); setError("");
    try { onAuthenticated(await login(profile.userId, pin)); }
    catch (e) { setError(e instanceof Error ? e.message : "Falha ao entrar"); }
    finally { setBusy(false); }
  }

  async function doRegister() {
    setBusy(true); setError("");
    try { onAuthenticated(await register(name, pin)); }
    catch (e) {
      const msg = e instanceof Error ? e.message : "Falha ao criar";
      if (msg.includes("taken") || msg.includes("409")) {
        try { onAuthenticated(await login(slugify(name), pin)); return; }
        catch { setError("Esse nome já existe. Se é seu, confira o PIN."); }
      } else { setError(msg); }
    }
    finally { setBusy(false); }
  }

  const subtitle =
    mode.kind === "new" ? "Criar perfil"
    : mode.kind === "pin" ? `Entrar como ${mode.profile.displayName}`
    : "Quem é você?";

  const canPin = !busy && pin.length >= 4;
  const canNew = !busy && name.trim().length > 0 && pin.length >= 4;

  function primaryBtn(enabled: boolean): React.CSSProperties {
    return {
      width: "100%", padding: "12px", borderRadius: 10,
      border: enabled ? "1px solid rgba(129,140,248,0.42)" : "1px solid rgba(255,255,255,0.08)",
      background: enabled ? "rgba(99,102,241,0.28)" : "rgba(255,255,255,0.04)",
      backdropFilter: "blur(8px)", WebkitBackdropFilter: "blur(8px)",
      color: enabled ? "#c0c8ff" : "rgba(140,140,200,0.35)",
      fontSize: 15, fontWeight: 600, cursor: enabled ? "pointer" : "not-allowed",
      transition: "all 0.15s", letterSpacing: "-0.01em",
    };
  }

  return (
    <div className="bg-root">
      <div className="bg-blob bg-blob-1" />
      <div className="bg-blob bg-blob-2" />
      <div className="bg-blob bg-blob-3" />

      <div style={{
        position: "absolute", inset: 0, zIndex: 1,
        display: "flex", alignItems: "center", justifyContent: "center", padding: 20,
      }}>
        <div style={{
          width: "100%", maxWidth: 360,
          padding: "28px 24px", borderRadius: 20,
          display: "flex", flexDirection: "column", gap: 14,
          animation: "fadeSlide 0.3s ease-out",
          ...GLASS_CARD,
        } as React.CSSProperties}>

          {/* Brand */}
          <div style={{ textAlign: "center", marginBottom: 2 }}>
            <div style={{
              width: 54, height: 54, borderRadius: 16, fontSize: 26,
              background: "rgba(129, 140, 248, 0.14)",
              backdropFilter: "blur(12px)", WebkitBackdropFilter: "blur(12px)",
              border: "1px solid rgba(129, 140, 248, 0.28)",
              boxShadow: "0 8px 32px rgba(99,102,241,0.2)",
              display: "flex", alignItems: "center", justifyContent: "center",
              margin: "0 auto 12px",
            } as React.CSSProperties}>🧠</div>
            <h1 style={{ margin: "0 0 3px", fontSize: 20, fontWeight: 700, color: "#dde0ff", letterSpacing: "-0.03em" }}>
              Memory Chat
            </h1>
            <p style={{ margin: 0, fontSize: 12.5, color: "rgba(130,130,200,0.55)" }}>
              {subtitle}
            </p>
          </div>

          {/* Profile list */}
          {mode.kind === "list" && (
            <>
              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                {profiles.map((p) => (
                  <button key={p.userId} type="button" className="glass-profile-btn"
                    onClick={() => { setPin(""); setError(""); setMode({ kind: "pin", profile: p }); }}
                    style={{
                      display: "flex", alignItems: "center", gap: 12,
                      padding: "11px 14px", borderRadius: 10, width: "100%", textAlign: "left",
                      border: "1px solid rgba(255,255,255,0.08)",
                      background: "rgba(255,255,255,0.04)",
                      backdropFilter: "blur(8px)", WebkitBackdropFilter: "blur(8px)",
                      color: "#dde0ff", fontSize: 14.5, cursor: "pointer", transition: "all 0.15s",
                    } as React.CSSProperties}
                  >
                    <div style={{
                      width: 30, height: 30, borderRadius: "50%", fontSize: 13, fontWeight: 700,
                      background: "rgba(99,102,241,0.28)",
                      border: "1px solid rgba(129,140,248,0.4)",
                      color: "#c0c8ff", flexShrink: 0, userSelect: "none",
                      display: "flex", alignItems: "center", justifyContent: "center",
                    }}>
                      {p.displayName.charAt(0).toUpperCase()}
                    </div>
                    {p.displayName}
                  </button>
                ))}
              </div>
              <button type="button"
                onClick={() => { setName(""); setPin(""); setError(""); setMode({ kind: "new" }); }}
                style={{
                  background: "none",
                  border: "1px dashed rgba(255,255,255,0.12)",
                  borderRadius: 10, color: "rgba(140,140,200,0.55)",
                  fontSize: 13, padding: "10px", cursor: "pointer", transition: "all 0.12s",
                }}>
                + Criar novo perfil
              </button>
            </>
          )}

          {/* PIN */}
          {mode.kind === "pin" && (
            <>
              <input style={{ ...inputStyle, letterSpacing: "0.15em" }}
                type="password" inputMode="numeric" autoFocus placeholder="PIN"
                value={pin} onChange={(e) => setPin(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter" && canPin) doLogin(mode.profile); }}
              />
              <button type="button" style={primaryBtn(canPin)} disabled={!canPin}
                onClick={() => doLogin(mode.profile)}>
                {busy ? "Entrando…" : "Entrar"}
              </button>
              <button type="button" style={linkBtnStyle}
                onClick={() => { setError(""); setMode({ kind: "list" }); }}>
                ← Voltar
              </button>
            </>
          )}

          {/* New profile */}
          {mode.kind === "new" && (
            <>
              <input style={inputStyle} type="text" autoFocus placeholder="Seu nome"
                value={name} onChange={(e) => setName(e.target.value)} />
              <input style={{ ...inputStyle, letterSpacing: "0.08em" }}
                type="password" inputMode="numeric" placeholder="Crie um PIN (mín. 4 dígitos)"
                value={pin} onChange={(e) => setPin(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter" && canNew) doRegister(); }}
              />
              <button type="button" style={primaryBtn(canNew)} disabled={!canNew} onClick={doRegister}>
                {busy ? "Criando…" : "Criar perfil"}
              </button>
              {profiles.length > 0 && (
                <button type="button" style={linkBtnStyle}
                  onClick={() => { setError(""); setMode({ kind: "list" }); }}>
                  ← Já tenho um perfil
                </button>
              )}
            </>
          )}

          {/* Error */}
          {error && (
            <p style={{
              margin: 0, fontSize: 13, color: "#f87171", textAlign: "center",
              padding: "8px 12px", borderRadius: 8,
              background: "rgba(248,113,113,0.1)",
              border: "1px solid rgba(248,113,113,0.28)",
            }}>
              {error}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
