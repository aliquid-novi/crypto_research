#!/usr/bin/env bash
# fetch_real_data.sh — pull REAL Binance USD-M futures data from Binance Vision.
#
# Binance Vision is a free public archive. No account, no API key, no auth.
# It was unreachable from the sandbox this repo was built in (egress policy),
# which is why the shipped CSVs are synthetic. Run this anywhere with normal
# internet and you get the real thing in the same schema.
#
#   ./fetch_real_data.sh BTCUSDT 2026-09-18
#   ./fetch_real_data.sh ETHUSDT 2026-09-18 real_data
#
# Then, unchanged:
#   python3 replay.py --datadir real_data/BTCUSDT-2026-09-18
#
set -euo pipefail

SYMBOL="${1:-BTCUSDT}"
DATE="${2:-2026-09-18}"
OUTROOT="${3:-real_data}"
BASE="https://data.binance.vision/data/futures/um/daily"

OUT="${OUTROOT}/${SYMBOL}-${DATE}"
mkdir -p "$OUT"

# Binance omits the header row on some archive files. These are the real
# column orders; we prepend them only if the first line looks like data.
BOOKTICKER_HDR="update_id,best_bid_price,best_bid_qty,best_ask_price,best_ask_qty,transaction_time,event_time"
AGGTRADES_HDR="agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker"

fetch () {                      # fetch <dataset> <local_name> <header>
  local ds="$1" local_name="$2" hdr="$3"
  local url="${BASE}/${ds}/${SYMBOL}/${SYMBOL}-${ds}-${DATE}.zip"
  local zip="${OUT}/${ds}.zip"

  echo ">> ${ds}"
  echo "   ${url}"
  if ! curl -fsSL --retry 3 -o "$zip" "$url"; then
    echo "   FAILED. Common causes:"
    echo "     - the date is in the future, or today (files land next day UTC)"
    echo "     - that symbol has no data for that date"
    echo "     - the date is a weekend for a non-24/7 product"
    echo "   Browse what exists: https://data.binance.vision/?prefix=data/futures/um/daily/${ds}/${SYMBOL}/"
    return 1
  fi

  unzip -o -q "$zip" -d "$OUT"
  rm -f "$zip"
  local csv
  csv="$(find "$OUT" -maxdepth 1 -name "${SYMBOL}-${ds}-${DATE}.csv" | head -1)"
  [ -n "$csv" ] || { echo "   no csv inside the zip"; return 1; }

  # prepend the header if the file starts with data
  if head -1 "$csv" | grep -qE '^[0-9]'; then
    echo "   (no header in archive — prepending)"
    { echo "$hdr"; cat "$csv"; } > "${csv}.tmp" && mv "${csv}.tmp" "$csv"
  fi
  mv "$csv" "${OUT}/${local_name}"
  echo "   -> ${OUT}/${local_name}  ($(wc -l < "${OUT}/${local_name}") lines)"
}

fetch bookTicker l1_book.csv "$BOOKTICKER_HDR"
fetch aggTrades  trades.csv  "$AGGTRADES_HDR"

# replay.py reads the tick from params.json; it is not in the archive.
# BTCUSDT and ETHUSDT perps tick at 0.10. CHECK YOUR SYMBOL against
# https://fapi.binance.com/fapi/v1/exchangeInfo  (PRICE_FILTER.tickSize)
cat > "${OUT}/params.json" <<JSON
{
  "source": "binance-vision",
  "symbol": "${SYMBOL}",
  "date": "${DATE}",
  "tick": 0.10,
  "_warning": "tick is assumed; verify against exchangeInfo PRICE_FILTER"
}
JSON

echo
echo "Done: ${OUT}"
echo
echo "NOTE: real data has no episodes.csv and no ground_truth.csv."
echo "  - episodes: define them from your own flatten decisions, or synthesise"
echo "    them on a schedule using build_episodes() in gen_market.py"
echo "  - ground truth: does not exist. That is why you validate the estimator"
echo "    on synthetic data first, then run the validated code here."
echo
echo "Other datasets in the same archive worth a look:"
echo "  bookDepth  - periodic depth snapshots beyond the touch"
echo "  metrics    - open interest, long/short ratios"
echo "  trades     - untruncated individual trades (larger than aggTrades)"
echo "Spot data lives under .../data/spot/daily/ with slightly different columns."
