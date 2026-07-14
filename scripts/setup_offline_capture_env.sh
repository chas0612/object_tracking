#!/usr/bin/env bash
# Create/repair the two conda environments for the offline capture pipeline.
#
# This intentionally does not clone, modify, or patch private MV-GoTrack.
# Supply an already checked-out and patched MV-GoTrack directory with
# --gotrack-dir. The script never runs git reset/checkout/apply.
set -euo pipefail

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/autodex-matplotlib}"
mkdir -p "$MPLCONFIGDIR"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GOTRACK_DIR="$REPO_ROOT/autodex/perception/thirdparty/MV-GoTrack"
GOTRACK_ENV="gotrack"
SAM3_ENV="sam3"
INSTALL_GOTRACK=1
INSTALL_SAM3=1
VERIFY=1

usage() {
    cat <<EOF
usage: $0 [options]

Creates the public-repo side of the offline capture environments. MV-GoTrack
must already be checked out at --gotrack-dir, with the approved private patch
already applied. No git command is run by this script.

options:
  --gotrack-dir DIR       Existing private MV-GoTrack checkout.
  --gotrack-env NAME      Conda environment name (default: gotrack).
  --sam3-env NAME         Conda environment name (default: sam3).
  --skip-gotrack          Install only the SAM3 environment.
  --skip-sam3             Install only the GoTrack environment.
  --no-verify             Do not run the post-install preflight checks.
  -h, --help              Show this message.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gotrack-dir) GOTRACK_DIR="$2"; shift 2 ;;
        --gotrack-env) GOTRACK_ENV="$2"; shift 2 ;;
        --sam3-env) SAM3_ENV="$2"; shift 2 ;;
        --skip-gotrack) INSTALL_GOTRACK=0; shift ;;
        --skip-sam3) INSTALL_SAM3=0; shift ;;
        --no-verify) VERIFY=0; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

command -v conda >/dev/null || { echo "conda is required" >&2; exit 1; }
GOTRACK_DIR="$(cd "$GOTRACK_DIR" 2>/dev/null && pwd || true)"
if [[ "$INSTALL_GOTRACK" -eq 1 && ! -f "$GOTRACK_DIR/scripts/onboard_custom_mesh_for_foundpose.py" ]]; then
    echo "MV-GoTrack checkout is missing or invalid: $GOTRACK_DIR" >&2
    echo "Clone the approved private repository and apply its patch first." >&2
    exit 1
fi

ensure_env() {
    local name="$1" python_version="$2"
    if ! conda env list | awk '{print $1}' | grep -Fxq "$name"; then
        conda create -y -n "$name" "python=$python_version"
    fi
}

run_pip() {
    local env_name="$1"; shift
    conda run --no-capture-output -n "$env_name" python -m pip "$@"
}

if [[ "$INSTALL_SAM3" -eq 1 ]]; then
    ensure_env "$SAM3_ENV" "3.12"
    run_pip "$SAM3_ENV" install --upgrade pip
    run_pip "$SAM3_ENV" install torch==2.10.0 torchvision==0.25.0 \
        --index-url https://download.pytorch.org/whl/cu128
    run_pip "$SAM3_ENV" install -r "$REPO_ROOT/requirements/offline_capture/sam3-cu128.txt"
    run_pip "$SAM3_ENV" install -e "$REPO_ROOT/autodex/perception/thirdparty/sam3"
fi

if [[ "$INSTALL_GOTRACK" -eq 1 ]]; then
    ensure_env "$GOTRACK_ENV" "3.10"
    run_pip "$GOTRACK_ENV" install --upgrade "pip<26" "setuptools<82" wheel
    run_pip "$GOTRACK_ENV" install torch==2.11.0 torchvision==0.26.0 xformers==0.0.35 \
        --index-url https://download.pytorch.org/whl/cu128
    run_pip "$GOTRACK_ENV" install -r "$REPO_ROOT/requirements/offline_capture/gotrack-cu128.txt"
    run_pip "$GOTRACK_ENV" install -e "$GOTRACK_DIR/external/bop_toolkit"
    run_pip "$GOTRACK_ENV" install "$GOTRACK_DIR/external/dinov2"
    run_pip "$GOTRACK_ENV" install \
        "git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae" \
        --no-build-isolation
fi

if [[ "$VERIFY" -eq 1 ]]; then
    if [[ "$INSTALL_SAM3" -eq 1 ]]; then
        conda run --no-capture-output -n "$SAM3_ENV" python \
            "$REPO_ROOT/scripts/check_offline_capture_setup.py" --component sam3 --require-repo-sam3
    fi
    if [[ "$INSTALL_GOTRACK" -eq 1 ]]; then
        conda run --no-capture-output -n "$GOTRACK_ENV" python \
            "$REPO_ROOT/scripts/check_offline_capture_setup.py" --component gotrack \
            --gotrack-dir "$GOTRACK_DIR"
    fi
fi

echo "[done] offline capture environments are ready"
