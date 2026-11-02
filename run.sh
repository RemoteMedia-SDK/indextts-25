#!/usr/bin/env bash
set -euo pipefail
PLUGIN_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ -z "${REMOTEMEDIA_PYTHON_SRC:-}" && -d "$PLUGIN_DIR/../../clients/python/remotemedia" ]]; then
    export REMOTEMEDIA_PYTHON_SRC="$(cd "$PLUGIN_DIR/../../clients/python" && pwd)"
fi
if [[ -n "${REMOTEMEDIA_PYTHON_SRC:-}" ]]; then
    CLIENT_ROOT="$(cd "$REMOTEMEDIA_PYTHON_SRC" && pwd)"
    if [[ ! -d "$CLIENT_ROOT/remotemedia" ]]; then
        echo 'REMOTEMEDIA_PYTHON_SRC must point to the SDK clients/python directory' >&2
        exit 1
    fi
    SOURCE_PATHS="$CLIENT_ROOT"
    REPOSITORY_ROOT="$(cd "$CLIENT_ROOT/../.." && pwd)"
    if [[ -d "$REPOSITORY_ROOT/remotemedia/protos" ]]; then
        SOURCE_PATHS="$SOURCE_PATHS:$REPOSITORY_ROOT"
    fi
    export PYTHONPATH="$SOURCE_PATHS${PYTHONPATH:+:$PYTHONPATH}"
fi
export REMOTEMEDIA_NODE_TIMEOUT_MS="${REMOTEMEDIA_NODE_TIMEOUT_MS:-600000}"
exec python3 "$PLUGIN_DIR/consume.py" "$@"
