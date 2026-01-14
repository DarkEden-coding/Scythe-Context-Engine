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
    "groq": {
        "chat_model": "openai/gpt-oss-120b",
        "batch_completion_window": "24h",
        "use_batch_for_indexing": True,
        "poll_interval_seconds": 30,
        "timeout_seconds": 60,
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
    while provider not in ["openrouter", "ollama", "groq"]:
        provider = (
            input("Select provider (openrouter/ollama/groq) [openrouter]: ")
            .strip()
            .lower()
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
    elif provider == "ollama":
        p_def = DEFAULTS["ollama"]
        config["ollama"] = {
            "summarization_model": ask(
                "Summarization Model", p_def["summarization_model"], skip
            ),
            "embedding_model": ask("Embedding Model", p_def["embedding_model"], skip),
        }
    # provider == "groq" will be handled below

    # Add Groq configuration option (for primary provider or hybrid mode)
    if ask("Configure Groq?", provider == "groq"):
        key = ""
        while not key:
            key = input("Enter Groq API key: ").strip()

        g_def = DEFAULTS["groq"]
        config["groq"] = {
            "api_key": key,
            "chat_model": ask("Chat Model", g_def["chat_model"], skip),
            "batch_completion_window": ask(
                "Batch Completion Window (24h/48h/72h/7d)",
                g_def["batch_completion_window"],
                skip,
            ),
            "use_batch_for_indexing": ask(
                "Use Batch for Indexing", g_def["use_batch_for_indexing"], skip
            ),
            "poll_interval_seconds": int(
                ask("Poll Interval (seconds)", g_def["poll_interval_seconds"], skip)
            ),
            "timeout_seconds": int(
                ask("Timeout (seconds)", g_def["timeout_seconds"], skip)
            ),
        }

        # If Groq is not the primary provider, set it as batch_provider
        if provider != "groq":
            config["batch_provider"] = "groq"
            print("\nNote: Groq configured for batch operations only.")
            print(f"Primary provider: {provider}")
            print("Batch provider: groq")
        else:
            # Groq is primary, need fallback for embeddings
            print("\nNote: Groq doesn't support embeddings yet.")
            print("You'll need to configure a fallback provider for embeddings.")

            # Ask for fallback provider for embeddings
            fallback = ""
            while fallback not in ["openrouter", "ollama"]:
                fallback = (
                    input(
                        "Select fallback provider for embeddings (openrouter/ollama) [openrouter]: "
                    )
                    .strip()
                    .lower()
                    or "openrouter"
                )

            config["embedding_provider"] = fallback

            if fallback == "openrouter":
                f_key = ""
                while not f_key:
                    f_key = input("Enter OpenRouter API key for embeddings: ").strip()
                # Don't overwrite existing openrouter config if it exists
                if "openrouter" not in config:
                    config["openrouter"] = {}
                config["openrouter"]["api_key"] = f_key
                config["openrouter"]["embedding_model"] = ask(
                    "Embedding Model", DEFAULTS["openrouter"]["embedding_model"], skip
                )
            else:  # ollama
                # Don't overwrite existing ollama config if it exists
                if "ollama" not in config:
                    config["ollama"] = {}
                config["ollama"]["embedding_model"] = ask(
                    "Embedding Model", DEFAULTS["ollama"]["embedding_model"], skip
                )

    Path("config/config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False)
    )
    print("\nconfig/config.json created successfully!")


if __name__ == "__main__":
    main()
