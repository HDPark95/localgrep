#!/bin/bash
# localgrep - One-click installation script
# Usage: curl -sSL https://raw.githubusercontent.com/HDPark95/localgrep/main/setup.sh | bash

set -e

echo "==================================="
echo "  localgrep installer"
echo "  Local Semantic Code Search"
echo "==================================="
echo ""

# ------------------------------------------
# 1. Check Python 3.11+
# ------------------------------------------
echo "[1/4] Checking Python..."

if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found. Please install Python 3.11 or later."
    echo "  macOS: brew install python@3.13"
    echo "  Linux: sudo apt install python3"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 11 ]); then
    echo "ERROR: Python 3.11+ required (found $PYTHON_VERSION)"
    exit 1
fi

echo "  Found Python $PYTHON_VERSION"

# ------------------------------------------
# 2. Check Ollama
# ------------------------------------------
echo "[2/4] Checking Ollama..."

if ! command -v ollama &> /dev/null; then
    echo "ERROR: Ollama not found. Please install it first:"
    echo "  macOS: brew install ollama"
    echo "  Linux: curl -fsSL https://ollama.ai/install.sh | sh"
    echo "  Or visit: https://ollama.ai"
    exit 1
fi

echo "  Found Ollama $(ollama --version 2>/dev/null || echo '(version unknown)')"

# ------------------------------------------
# 3. Install localgrep
# ------------------------------------------
echo "[3/4] Installing localgrep..."

pip install localgrep

echo "  localgrep installed successfully"

# ------------------------------------------
# 4. Pull embedding model
# ------------------------------------------
echo "[4/4] Pulling embedding model (nomic-embed-text)..."

ollama pull nomic-embed-text

echo "  Model ready"

# ------------------------------------------
# Optional: Claude Code integration
# ------------------------------------------
if [ -d "$HOME/.claude" ]; then
    echo ""
    echo "Detected Claude Code installation."
    read -p "Configure Claude Code integration? [Y/n] " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
        localgrep install-claude
        echo "  Claude Code integration configured!"
    fi
fi

# ------------------------------------------
# Done
# ------------------------------------------
echo ""
echo "==================================="
echo "  Installation complete!"
echo "==================================="
echo ""
echo "Get started:"
echo "  cd /your/project"
echo "  localgrep index ."
echo "  localgrep search \"your query\""
echo ""
