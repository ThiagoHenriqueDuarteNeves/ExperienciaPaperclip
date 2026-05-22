#!/usr/bin/env python3
"""Interactive setup script for the Memory API.

Prompts for configuration values and writes them to .env.
Existing .env values are used as defaults when available.
"""

import getpass
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parent / ".env"


def load_env() -> dict[str, str]:
    """Parse existing .env into a dict, preserving comments."""
    if not ENV_PATH.exists():
        return {}
    env = {}
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip()
    return env


def prompt(key: str, label: str, current: str | None = None, secret: bool = False) -> str:
    """Prompt the user for a value, showing current default if available."""
    if current and current != "placeholder-key-not-configured":
        label = f"{label} [{current}]"

    if secret:
        value = getpass.getpass(f"  {label}: ").strip()
    else:
        value = input(f"  {label}: ").strip()

    return value if value else (current or "")


def write_env(values: dict[str, str]) -> None:
    """Write values to .env, preserving existing comments and adding new keys."""
    env = load_env()
    env.update(values)

    lines = []
    for key, value in env.items():
        lines.append(f"{key}={value}")

    ENV_PATH.write_text("\n".join(lines) + "\n")
    print(f"\n  Config written to {ENV_PATH}")


def main() -> None:
    print("Memory API Setup")
    print("=" * 40)

    env = load_env()

    # --- LLM Provider ---
    print("\n[1/4] LLM Provider Selection")
    print("  Choose your LLM provider:")
    print("  1) Anthropic (Claude models)")
    print("  2) Deepseek (v4 with Anthropic-compatible API)")
    provider_choice = input("  Select (1 or 2, default: 1): ").strip() or "1"
    
    if provider_choice == "2":
        provider = "deepseek"
        print("  Selected: Deepseek (https://api.deepseek.com/anthropic)")
        api_key = prompt(
            "ANTHROPIC_API_KEY",
            "Deepseek API Key",
            current=env.get("ANTHROPIC_API_KEY"),
            secret=True,
        )
    else:
        provider = "anthropic"
        print("  Selected: Anthropic (https://api.anthropic.com)")
        api_key = prompt(
            "ANTHROPIC_API_KEY",
            "Anthropic API Key",
            current=env.get("ANTHROPIC_API_KEY"),
            secret=True,
        )

    # --- Letta / MemGPT ---
    print("\n[2/4] Letta (MemGPT)")
    print("  Letta provides the procedural memory tier.")
    letta_url = prompt(
        "LETTA_BASE_URL",
        "LETTA_BASE_URL",
        current=env.get("LETTA_BASE_URL"),
    )
    letta_key = prompt(
        "LETTA_API_KEY",
        "LETTA_API_KEY",
        current=env.get("LETTA_API_KEY"),
        secret=True,
    )

    # --- Write ---
    write_env({
        "LLM_PROVIDER": provider,
        "ANTHROPIC_API_KEY": api_key,
        "LETTA_BASE_URL": letta_url,
        "LETTA_API_KEY": letta_key,
    })

    print("\nSetup complete. Run `docker compose up --build -d` to start.\n")


if __name__ == "__main__":
    main()
