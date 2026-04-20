#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" == "Darwin" ]]; then
  if ! command -v brew >/dev/null 2>&1; then
    echo "Homebrew is required to install Grafana and Prometheus automatically on macOS."
    exit 1
  fi

  export HOMEBREW_NO_AUTO_UPDATE="${HOMEBREW_NO_AUTO_UPDATE:-1}"
  brew install prometheus grafana
  exit 0
fi

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y apt-transport-https wget gnupg

mkdir -p /etc/apt/keyrings
wget -O /etc/apt/keyrings/grafana.asc https://apt.grafana.com/gpg-full.key
chmod 644 /etc/apt/keyrings/grafana.asc

cat >/etc/apt/sources.list.d/grafana.list <<'EOF'
deb [signed-by=/etc/apt/keyrings/grafana.asc] https://apt.grafana.com stable main
EOF

apt-get update
apt-get install -y prometheus grafana
