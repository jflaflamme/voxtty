#!/bin/bash
# voxtty installer - https://github.com/jflaflamme/voxtty
# Usage: curl -fsSL https://raw.githubusercontent.com/jflaflamme/voxtty/main/install.sh | bash
set -euo pipefail

REPO="jflaflamme/voxtty"
BINARY="voxtty"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/bin}"
CLEANUP_DIR=""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYAN}==> ${NC}$1"; }
ok()    { echo -e "${GREEN}==> ${NC}$1"; }
warn()  { echo -e "${YELLOW}==> ${NC}$1"; }
fail()  { echo -e "${RED}==> ${NC}$1"; exit 1; }

# Detect platform
detect_platform() {
    local os arch

    os="$(uname -s)"
    arch="$(uname -m)"

    case "$os" in
        Linux)  os="linux" ;;
        *)      fail "Unsupported OS: $os (voxtty currently supports Linux only)" ;;
    esac

    case "$arch" in
        x86_64|amd64)   arch="x86_64" ;;
        aarch64|arm64)  arch="aarch64" ;;
        *)              fail "Unsupported architecture: $arch" ;;
    esac

    echo "${os}-${arch}"
}

# Get latest release tag from GitHub
get_latest_version() {
    local url="https://api.github.com/repos/${REPO}/releases/latest"
    if command -v curl &>/dev/null; then
        curl -fsSL "$url" | grep '"tag_name"' | head -1 | sed 's/.*"tag_name": *"//;s/".*//'
    elif command -v wget &>/dev/null; then
        wget -qO- "$url" | grep '"tag_name"' | head -1 | sed 's/.*"tag_name": *"//;s/".*//'
    else
        fail "Neither curl nor wget found"
    fi
}

# Download and install
download_and_install() {
    local platform="$1"
    local version="$2"
    local asset="voxtty-${version}-${platform}.tar.gz"
    local url="https://github.com/${REPO}/releases/download/${version}/${asset}"
    CLEANUP_DIR="$(mktemp -d)"
    trap 'rm -rf "$CLEANUP_DIR"' EXIT
    local tmpdir="$CLEANUP_DIR"

    info "Downloading ${asset}..."
    if command -v curl &>/dev/null; then
        curl -fsSL "$url" -o "${tmpdir}/${asset}" || fail "Download failed. Is there a release for ${version}?"
    else
        wget -q "$url" -O "${tmpdir}/${asset}" || fail "Download failed. Is there a release for ${version}?"
    fi

    info "Extracting..."
    tar xzf "${tmpdir}/${asset}" -C "$tmpdir"

    mkdir -p "$INSTALL_DIR"
    install -m 755 "${tmpdir}/voxtty" "${INSTALL_DIR}/voxtty"

    ok "Installed voxtty ${version} to ${INSTALL_DIR}/voxtty"
}

# Build from source as fallback
build_from_source() {
    info "No pre-built binary available. Building from source..."

    if ! command -v cargo &>/dev/null; then
        fail "Rust toolchain not found. Install it first: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    fi

    # Check build dependencies
    check_build_deps

    CLEANUP_DIR="$(mktemp -d)"
    trap 'rm -rf "$CLEANUP_DIR"' EXIT
    local tmpdir="$CLEANUP_DIR"

    info "Cloning repository..."
    git clone --depth 1 "https://github.com/${REPO}.git" "${tmpdir}/voxtty"

    info "Building (this may take a few minutes)..."
    cd "${tmpdir}/voxtty"
    cargo build --release

    mkdir -p "$INSTALL_DIR"
    install -m 755 "target/release/voxtty" "${INSTALL_DIR}/voxtty"

    ok "Built and installed voxtty to ${INSTALL_DIR}/voxtty"
}

# Check system dependencies
check_deps() {
    local missing=()

    # Runtime dependencies
    if ! command -v ydotool &>/dev/null; then
        missing+=("ydotool")
    fi

    if [ ${#missing[@]} -gt 0 ]; then
        warn "Missing runtime dependencies: ${missing[*]}"
        echo "    Install with: sudo apt install ${missing[*]}"
        echo "    voxtty needs ydotool for typing text into applications"
    fi
}

# Check build dependencies (for source builds)
check_build_deps() {
    local missing=()

    # ALSA dev headers needed by cpal
    if ! pkg-config --exists alsa 2>/dev/null; then
        missing+=("libasound2-dev")
    fi

    # pkg-config itself
    if ! command -v pkg-config &>/dev/null; then
        missing+=("pkg-config")
    fi

    if [ ${#missing[@]} -gt 0 ]; then
        fail "Missing build dependencies: ${missing[*]}\n    Install with: sudo apt install ${missing[*]}"
    fi
}

# Check PATH
check_path() {
    case ":$PATH:" in
        *":${INSTALL_DIR}:"*) ;;
        *)
            warn "${INSTALL_DIR} is not in your PATH"
            echo "    Add it with: export PATH=\"${INSTALL_DIR}:\$PATH\""
            echo "    Or add to your shell profile (~/.bashrc, ~/.zshrc, etc.)"
            ;;
    esac
}

main() {
    echo ""
    echo -e "${CYAN}voxtty installer${NC}"
    echo -e "${CYAN}The power of whisper — your voice commands${NC}"
    echo ""

    local platform
    platform="$(detect_platform)"
    info "Detected platform: ${platform}"

    # Try pre-built binary first
    local version
    version="$(get_latest_version 2>/dev/null || echo "")"

    if [ -n "$version" ]; then
        info "Latest release: ${version}"
        download_and_install "$platform" "$version" 2>/dev/null || build_from_source
    else
        warn "No releases found, building from source..."
        build_from_source
    fi

    check_deps
    check_path

    echo ""
    ok "Installation complete!"
    echo ""
    echo "  Quick start:"
    echo "    voxtty --echo-test          # Test your microphone"
    echo "    voxtty --speaches --tui     # Run with Speaches backend + TUI"
    echo ""
    echo "  Docs: https://github.com/${REPO}"
    echo ""
}

main "$@"
