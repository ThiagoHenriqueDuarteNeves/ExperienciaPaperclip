"use client";

import { useState } from "react";
import {
  listProfiles,
  login,
  register,
  slugify,
  type Profile,
  type Session,
} from "../lib/auth";

type Mode = { kind: "list" } | { kind: "pin"; profile: Profile } | { kind: "new" };

const inputStyle: React.CSSProperties = {
  width: "100%",
  padding: "12px 14px",
  borderRadius: 10,
  border: "1.5px solid var(--border)",
  background: "var(--surface-alt)",
  color: "var(--text)",
  fontSize: 16,
  outline: "none",
  transition: "border-color 0.15s",
};

const linkBtnStyle: React.CSSProperties = {
  background: "none",
  border: "none",
  color: "var(--text-muted)",
  fontSize: 13,
  cursor: "pointer",
  padding: "4px",
  textAlign: "center",
  letterSpacing: "0.01em",
};

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
      if (msg.includes("taken") || msg.includes("409")) {
        try {
          const s = await login(slugify(name), pin);
          onAuthenticated(s);
          return;
        } catch {
          setError("Esse nome já existe. Se é seu, confira o PIN.");
        }
      } else {
        setError(msg);
      }
    } finally {
      setBusy(false);
    }
  }

  const subtitle =
    mode.kind === "new"
      ? "Criar perfil"
      : mode.kind === "pin"
      ? `Entrar como ${mode.profile.displayName}`
      : "Quem é você?";

  const canSubmitPin = !busy && pin.length >= 4;
  const canSubmitNew = !busy && name.trim().length > 0 && pin.length >= 4;

  function primaryBtnStyle(enabled: boolean): React.CSSProperties {
    return {
      width: "100%",
      padding: "12px",
      borderRadius: 10,
      border: "none",
      background: enabled ? "var(--accent)" : "var(--surface-alt)",
      color: enabled ? "var(--bg)" : "var(--text-muted)",
      fontSize: 15,
      fontWeight: 600,
      cursor: enabled ? "pointer" : "not-allowed",
      transition: "all 0.15s",
      letterSpacing: "-0.01em",
    };
  }

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        height: "100dvh",
        padding: 20,
        background: "var(--bg)",
      }}
    >
      <div
        style={{
          width: "100%",
          maxWidth: 360,
          display: "flex",
          flexDirection: "column",
          gap: 14,
          padding: "28px 24px",
          borderRadius: 18,
          background: "var(--surface)",
          border: "1px solid var(--border)",
          boxShadow: "var(--shadow-lg)",
          animation: "fadeSlide 0.25s ease-out",
        }}
      >
        {/* Brand */}
        <div style={{ textAlign: "center", marginBottom: 2 }}>
          <div
            style={{
              width: 50,
              height: 50,
              borderRadius: 14,
              background: "var(--accent-dim)",
              border: "1px solid var(--accent-border)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 24,
              margin: "0 auto 12px",
            }}
          >
            🧠
          </div>
          <h1
            style={{
              margin: "0 0 3px",
              fontSize: 20,
              fontWeight: 700,
              color: "var(--text)",
              letterSpacing: "-0.03em",
            }}
          >
            Memory Chat
          </h1>
          <p style={{ margin: 0, fontSize: 12.5, color: "var(--text-muted)" }}>
            {subtitle}
          </p>
        </div>

        {/* Profile list */}
        {mode.kind === "list" && (
          <>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {profiles.map((p) => (
                <button
                  key={p.userId}
                  type="button"
                  onClick={() => {
                    setPin("");
                    setError("");
                    setMode({ kind: "pin", profile: p });
                  }}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 12,
                    padding: "11px 14px",
                    borderRadius: 10,
                    border: "1px solid var(--border)",
                    background: "var(--surface-alt)",
                    color: "var(--text)",
                    fontSize: 14.5,
                    cursor: "pointer",
                    textAlign: "left",
                    transition: "background 0.12s, border-color 0.12s",
                  }}
                >
                  <div
                    style={{
                      width: 30,
                      height: 30,
                      borderRadius: "50%",
                      background: "var(--accent)",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontSize: 13,
                      fontWeight: 700,
                      color: "var(--bg)",
                      flexShrink: 0,
                      userSelect: "none",
                    }}
                  >
                    {p.displayName.charAt(0).toUpperCase()}
                  </div>
                  {p.displayName}
                </button>
              ))}
            </div>
            <button
              type="button"
              onClick={() => {
                setName("");
                setPin("");
                setError("");
                setMode({ kind: "new" });
              }}
              style={{
                background: "none",
                border: "1px dashed var(--border)",
                borderRadius: 10,
                color: "var(--text-muted)",
                fontSize: 13,
                padding: "10px",
                cursor: "pointer",
                transition: "all 0.12s",
              }}
            >
              + Criar novo perfil
            </button>
          </>
        )}

        {/* PIN entry */}
        {mode.kind === "pin" && (
          <>
            <input
              style={{ ...inputStyle, letterSpacing: "0.15em" }}
              type="password"
              inputMode="numeric"
              autoFocus
              placeholder="PIN"
              value={pin}
              onChange={(e) => setPin(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && canSubmitPin) doLogin(mode.profile);
              }}
            />
            <button
              type="button"
              style={primaryBtnStyle(canSubmitPin)}
              disabled={!canSubmitPin}
              onClick={() => doLogin(mode.profile)}
            >
              {busy ? "Entrando…" : "Entrar"}
            </button>
            <button
              type="button"
              style={linkBtnStyle}
              onClick={() => {
                setError("");
                setMode({ kind: "list" });
              }}
            >
              ← Voltar
            </button>
          </>
        )}

        {/* New profile */}
        {mode.kind === "new" && (
          <>
            <input
              style={inputStyle}
              type="text"
              autoFocus
              placeholder="Seu nome"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
            <input
              style={{ ...inputStyle, letterSpacing: "0.08em" }}
              type="password"
              inputMode="numeric"
              placeholder="Crie um PIN (mín. 4 dígitos)"
              value={pin}
              onChange={(e) => setPin(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && canSubmitNew) doRegister();
              }}
            />
            <button
              type="button"
              style={primaryBtnStyle(canSubmitNew)}
              disabled={!canSubmitNew}
              onClick={doRegister}
            >
              {busy ? "Criando…" : "Criar perfil"}
            </button>
            {profiles.length > 0 && (
              <button
                type="button"
                style={linkBtnStyle}
                onClick={() => {
                  setError("");
                  setMode({ kind: "list" });
                }}
              >
                ← Já tenho um perfil
              </button>
            )}
          </>
        )}

        {/* Error */}
        {error && (
          <p
            style={{
              margin: 0,
              fontSize: 13,
              color: "var(--danger)",
              textAlign: "center",
              padding: "8px 12px",
              borderRadius: 8,
              background: "var(--danger-dim)",
              border: "1px solid var(--danger-border)",
            }}
          >
            {error}
          </p>
        )}
      </div>
    </div>
  );
}
