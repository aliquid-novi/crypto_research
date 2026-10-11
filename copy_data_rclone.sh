#!/usr/bin/env bash
set -euo pipefail

# Incremental Google Drive -> local data sync for Git Bash.
#
# Put this file beside your ./data folder, e.g.
#   ~/Documents/copy_data_rclone.sh
#   ~/Documents/data/
#
# It compares the newest DATE DIRECTORY on Google Drive with the newest
# date already present locally, then only asks rclone to consider a recent
# date window instead of rechecking the full archive.
#
# Default: recheck the latest local date plus 2 days of overlap before it,
# which protects against late/rebuilt compact files.
#
# Override if desired:
#   LOOKBACK_DAYS=7 ./copy_data_rclone.sh
#   LOCAL_ROOT=/c/Users/markp/Documents/data ./copy_data_rclone.sh

REMOTE="${REMOTE:-gdrive_ral:crypto_market_data/compact}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_ROOT="${LOCAL_ROOT:-$SCRIPT_DIR/data}"

LOOKBACK_DAYS="${LOOKBACK_DAYS:-2}"
TRANSFERS="${TRANSFERS:-8}"
CHECKERS="${CHECKERS:-16}"

mkdir -p "$LOCAL_ROOT"

echo "============================================================"
echo "Incremental market-data sync"
echo "============================================================"
echo "Remote:       $REMOTE"
echo "Local:        $LOCAL_ROOT"
echo "Lookback:     $LOOKBACK_DAYS day(s)"
echo

# Find newest YYYY-MM-DD directory on the remote.
# This scans directory names only, not every parquet file.
echo "Checking newest remote date..."
REMOTE_LATEST_DATE="$(
    rclone lsf "$REMOTE" --dirs-only --recursive 2>/dev/null \
    | grep -oE '20[0-9]{2}-[0-9]{2}-[0-9]{2}' \
    | sort -u \
    | tail -1
)"

if [[ -z "${REMOTE_LATEST_DATE:-}" ]]; then
    echo "ERROR: Could not find a YYYY-MM-DD directory on:"
    echo "       $REMOTE"
    exit 1
fi

# Find newest YYYY-MM-DD directory already present locally.
echo "Checking newest local date..."
LOCAL_LATEST_DATE="$(
    find "$LOCAL_ROOT" -type d 2>/dev/null \
    | grep -oE '20[0-9]{2}-[0-9]{2}-[0-9]{2}$' \
    | sort -u \
    | tail -1 || true
)"

echo
echo "Newest remote date: $REMOTE_LATEST_DATE"

if [[ -n "${LOCAL_LATEST_DATE:-}" ]]; then
    echo "Newest local date:  $LOCAL_LATEST_DATE"
else
    echo "Newest local date:  NONE"
fi
echo

# If this is the first sync, there is no sensible recent boundary.
if [[ -z "${LOCAL_LATEST_DATE:-}" ]]; then
    echo "No existing local data found."
    echo "Running initial full copy..."
    echo

    rclone copy \
        "$REMOTE" \
        "$LOCAL_ROOT" \
        --progress \
        --transfers "$TRANSFERS" \
        --checkers "$CHECKERS"

    echo
    echo "Initial copy complete."
    exit 0
fi

# Recheck a small overlap window in case the compactor produced/rebuilt
# files for a recent older date after the previous local sync.
START_DATE="$(date -u -d "$LOCAL_LATEST_DATE - $LOOKBACK_DAYS days" +%F)"

# Don't attempt dates beyond what currently exists remotely.
if [[ "$START_DATE" > "$REMOTE_LATEST_DATE" ]]; then
    START_DATE="$REMOTE_LATEST_DATE"
fi

echo "Syncing date window:"
echo "  $START_DATE -> $REMOTE_LATEST_DATE"
echo

# Build include filters for only the relevant date directories.
FILTER_ARGS=()
CURRENT="$START_DATE"

while [[ "$CURRENT" < "$REMOTE_LATEST_DATE" || "$CURRENT" == "$REMOTE_LATEST_DATE" ]]; do
    FILTER_ARGS+=(--include "**/$CURRENT/**")
    CURRENT="$(date -u -d "$CURRENT + 1 day" +%F)"
done

# rclone automatically excludes files that do not match the include filters.
# Existing unchanged files inside this recent window are skipped.
rclone copy \
    "$REMOTE" \
    "$LOCAL_ROOT" \
    "${FILTER_ARGS[@]}" \
    --progress \
    --transfers "$TRANSFERS" \
    --checkers "$CHECKERS"

echo
echo "============================================================"
echo "Sync complete"
echo "============================================================"

NEW_LOCAL_LATEST="$(
    find "$LOCAL_ROOT" -type d 2>/dev/null \
    | grep -oE '20[0-9]{2}-[0-9]{2}-[0-9]{2}$' \
    | sort -u \
    | tail -1 || true
)"

echo "Remote latest date: $REMOTE_LATEST_DATE"
echo "Local latest date:  ${NEW_LOCAL_LATEST:-NONE}"

if [[ "${NEW_LOCAL_LATEST:-}" == "$REMOTE_LATEST_DATE" ]]; then
    echo "Status: local has reached the newest remote date."
else
    echo "Status: local date still differs from remote; inspect rclone output."
fi
