#!/bin/bash
set -e

echo "Building voxtty Debian Package..."

# Build the Rust application
cargo build --release

# Create debian package
dpkg-buildpackage -rfakeroot -us -uc

echo "Package built successfully!"
echo "Install with: sudo dpkg -i ../voxtty_*.deb"
