#!/usr/bin/env python3
import json
from pathlib import Path

DEFAULTS = {
    "indexing": {
        "supported_languages": {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".go": "go",
            ".rs": "rust",
            ".md": "markdown",
        },
        "ignored_dirs": [
            ".git",
            "node_modules",
            "__pycache__",
            "venv",
            ".venv",
            "static",
            ".cudavenv",
            "build",
        ],
        "ignored_files": [
            "package-lock.json",
            "yarn.lock",
            "pnpm-lock.yaml",
            "bun.lockb",
            "Gemfile.lock",
            "Cargo.lock",
        ],
    },
    "openrouter": {
        "api_base": "https://openrouter.ai/api/v1",
        "chat_model": "openai/gpt-oss-120b:exacto",
        "embedding_model": "openai/text-embedding-3-small",
        "timeout_seconds": 15,
        "chat_provider_whitelist": ["groq"],
        "embedding_provider_whitelist": ["openai"],
    },
    "ollama": {
        "summarization_model": "gemma3:1b",
        "embedding_model": "qwen3-embedding:0.6b",
    },
}


def ask(prompt, default, skip_interactive=False):
    """Returns default if skip_interactive is True, else prompts user."""
    if skip_interactive:
        return default

    if isinstance(default, bool):
        choice = input(f"{prompt} ({'Y/n' if default else 'y/N'}): ").strip().lower()
        return default if not choice else choice in ["y", "yes"]

    res = input(f"{prompt} [{default}]: ").strip()
    return res if res else default


def get_list(prompt, defaults, skip_interactive):
    if skip_interactive or not ask(f"Customize {prompt}?", False):
        return defaults
    print(f"Enter items for {prompt} (one per line, empty to finish):")
    items = []
    while True:
        val = input("> ").strip()
        if not val:
            break
        items.append(val)
    return items


def main():
    print("=== Scythe Context Engine Configuration ===\n")

    provider = ""
    while provider not in ["openrouter", "ollama"]:
        provider = (
            input("Select provider (openrouter/ollama) [openrouter]: ").strip().lower()
            or "openrouter"
        )

    skip = ask("Use default settings for the rest?", True)

    config = {
        "cache": {"ttl_seconds": 86400},
        "provider": provider,
        "indexing": {
            "supported_languages": DEFAULTS["indexing"]["supported_languages"],
            "ignored_dirs": get_list(
                "ignored directories", DEFAULTS["indexing"]["ignored_dirs"], skip
            ),
            "ignored_files": get_list(
                "ignored files", DEFAULTS["indexing"]["ignored_files"], skip
            ),
        },
    }

    if provider == "openrouter":
        key = ""
        while not key:
            key = input("Enter OpenRouter API key: ").strip()

        p_def = DEFAULTS["openrouter"]
        config["openrouter"] = {
            "api_key": key,
            "api_base": ask("API Base", p_def["api_base"], skip),
            "chat_model": ask("Chat Model", p_def["chat_model"], skip),
            "embedding_model": ask("Embedding Model", p_def["embedding_model"], skip),
            "timeout_seconds": int(ask("Timeout", p_def["timeout_seconds"], skip)),
            "chat_provider_whitelist": p_def["chat_provider_whitelist"]
            if skip or ask("Use chat whitelist?", True)
            else [],
            "embedding_provider_whitelist": p_def["embedding_provider_whitelist"]
            if skip or ask("Use embedding whitelist?", True)
            else [],
        }
    else:
        p_def = DEFAULTS["ollama"]
        config["ollama"] = {
            "summarization_model": ask(
                "Summarization Model", p_def["summarization_model"], skip
            ),
            "embedding_model": ask("Embedding Model", p_def["embedding_model"], skip),
        }

    Path("config/config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))
    print("\n✅ config/config.json created successfully!")


if __name__ == "__main__":
    main()
