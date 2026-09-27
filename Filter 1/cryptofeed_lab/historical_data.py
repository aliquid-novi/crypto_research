
"""
historical_data.py

Small async historical market-data layer with a common interface across:

- Hyperliquid
- Binance (spot / USD-M / COIN-M)
- Independent Reserve
- Deribit
- OKX
- Crypto.com Exchange

The goal is a *research* abstraction, not to pretend every venue has identical
historical coverage. Methods raise DataAvailabilityError where the venue does
not expose complete public history.

Dependencies:
    pip install httpx pandas pyarrow

Example:
    from historical_data import HistoricalData

    async with HistoricalData("binance", "BTC-USDT", market="spot") as h:
        candles = await h.candles(
            start="2026-09-20",
            end="2026-09-21",
            interval="1m",
        )

        trades = await h.trades(
            start="2026-09-20",
            end="2026-09-20 01:00:00",
        )
"""

from __future__ import annotations

import asyncio
import math
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import httpx
import pandas as pd


class HistoricalDataError(RuntimeError):
    pass


class DataAvailabilityError(HistoricalDataError):
    pass


EXCHANGE_ALIASES = {
    "hyperliquid": "hyperliquid",
    "hl": "hyperliquid",
    "binance": "binance",
    "independent reserve": "independent_reserve",
    "independent_reserve": "independent_reserve",
    "independent-reserve": "independent_reserve",
    "ir": "independent_reserve",
    "deribit": "deribit",
    "okx": "okx",
    "crypto.com": "cryptocom",
    "crypto_com": "cryptocom",
    "crypto-com": "cryptocom",
    "cryptocom": "cryptocom",
    "cdc": "cryptocom",
}

CAPABILITIES = {
    "hyperliquid": {
        "candles": "Public API; only the most recent 5,000 candles are available.",
        "trades": (
            "No general market-wide historical trade REST endpoint. "
            "Official bulk trade/fill archives are on S3."
        ),
        "order_book": (
            "Official S3 archive contains historical L2 snapshots, but it is a "
            "separate requester-pays bulk-data workflow."
        ),
    },
    "binance": {
        "candles": "Public REST klines; paginated by this module.",
        "trades": (
            "Uses public aggregate trades (aggTrades), not raw individual fills. "
            "For very large backfills use Binance's official bulk data archives."
        ),
        "order_book": "No long historical L2 reconstruction through normal REST.",
    },
    "independent_reserve": {
        "candles": "Only hourly summary history, maximum 240 hours back.",
        "trades": "Only the latest 50 public trades are exposed.",
        "order_book": "Current order book only via public REST.",
    },
    "deribit": {
        "candles": "Public TradingView chart endpoint.",
        "trades": "Public historical trades by instrument and time.",
        "order_book": "Current order book through standard public API; not full historical L2 here.",
    },
    "okx": {
        "candles": "Public historical candlesticks.",
        "trades": "Public history-trades endpoint; currently limited to recent history by OKX.",
        "order_book": (
            "OKX also has a separate historical-market-data facility including "
            "order-book modules; not wrapped in this first version."
        ),
    },
    "cryptocom": {
        "candles": "Public get-candlestick with start/end timestamps.",
        "trades": "Public get-trades with start/end timestamps.",
        "order_book": "Current book via public API; not historical L2 here.",
    },
}


def _to_ms(value: Any | None) -> int | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return int(ts.timestamp() * 1000)


def _utc_now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _ms_to_datetime(ms: int | float) -> pd.Timestamp:
    return pd.to_datetime(ms, unit="ms", utc=True)


def _interval_ms(interval: str) -> int:
    """
    Convert fixed-width intervals such as 1m, 4h, 1d, 1w into milliseconds.
    Calendar months are intentionally not supported for pagination.
    """
    m = re.fullmatch(r"(\d+)([mhdw])", interval)
    if not m:
        raise ValueError(
            f"Unsupported fixed interval {interval!r}. "
            "Use values such as 1m, 5m, 1h, 4h, 1d, 1w."
        )

    n = int(m.group(1))
    unit = m.group(2)

    multiplier = {
        "m": 60_000,
        "h": 3_600_000,
        "d": 86_400_000,
        "w": 604_800_000,
    }[unit]

    return n * multiplier


def _normalise_range(start: Any | None, end: Any | None, default_days: int = 1) -> tuple[int, int]:
    end_ms = _to_ms(end) if end is not None else _utc_now_ms()
    start_ms = _to_ms(start) if start is not None else end_ms - default_days * 86_400_000

    if start_ms >= end_ms:
        raise ValueError("start must be earlier than end")

    return start_ms, end_ms


def _finalise(df: pd.DataFrame, *, dedupe: list[str] | None = None) -> pd.DataFrame:
    if df.empty:
        return df

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").astype("Int64")
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    if dedupe:
        present = [c for c in dedupe if c in df.columns]
        if present:
            df = df.drop_duplicates(subset=present, keep="last")

    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")

    return df.reset_index(drop=True)


class HistoricalData:
    """
    Common historical-data client.

    Parameters
    ----------
    exchange:
        hyperliquid, binance, independent_reserve, deribit, okx, cryptocom
    symbol:
        A convenient symbol. Examples:
          Binance: BTC-USDT
          Hyperliquid perp: BTC or BTC-USDC
          Independent Reserve: BTC-AUD
          Deribit: BTC-PERPETUAL
          OKX: BTC-USDT or BTC-USDT-SWAP
          Crypto.com spot: BTC-USDT  -> converted to BTC_USDT
          Crypto.com perp: BTC-USD-PERP -> converted to BTCUSD-PERP
    market:
        Binance only: spot, usdm, coinm.
    """

    def __init__(
        self,
        exchange: str,
        symbol: str,
        *,
        market: str = "spot",
        timeout: float = 30.0,
        max_retries: int = 4,
    ):
        exchange_key = exchange.strip().lower()
        if exchange_key not in EXCHANGE_ALIASES:
            raise ValueError(
                f"Unsupported exchange {exchange!r}. "
                f"Supported: {sorted(set(EXCHANGE_ALIASES.values()))}"
            )

        self.exchange = EXCHANGE_ALIASES[exchange_key]
        self.symbol = symbol.strip().upper()
        self.market = market.strip().lower()
        self.timeout = timeout
        self.max_retries = max_retries

        if self.exchange == "binance" and self.market not in {"spot", "usdm", "coinm"}:
            raise ValueError("For Binance, market must be 'spot', 'usdm', or 'coinm'.")

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            headers={"User-Agent": "crypto-research-historical-data/0.1"},
        )

    async def __aenter__(self) -> "HistoricalData":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    def capabilities(self) -> dict[str, str]:
        return CAPABILITIES[self.exchange].copy()

    async def _request_json(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> Any:
        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = await self._client.request(
                    method,
                    url,
                    params=params,
                    json=json,
                )

                if response.status_code == 429:
                    wait = min(2 ** attempt, 8)
                    await asyncio.sleep(wait)
                    continue

                response.raise_for_status()
                return response.json()

            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
                if attempt == self.max_retries - 1:
                    break
                await asyncio.sleep(min(2 ** attempt, 8))

        raise HistoricalDataError(
            f"Request failed after {self.max_retries} attempts: "
            f"{method} {url}: {last_error}"
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def candles(
        self,
        *,
        start: Any | None = None,
        end: Any | None = None,
        interval: str = "1m",
    ) -> pd.DataFrame:
        start_ms, end_ms = _normalise_range(start, end)

        if self.exchange == "binance":
            return await self._binance_candles(start_ms, end_ms, interval)

        if self.exchange == "hyperliquid":
            return await self._hyperliquid_candles(start_ms, end_ms, interval)

        if self.exchange == "independent_reserve":
            return await self._independent_reserve_candles(start_ms, end_ms, interval)

        if self.exchange == "deribit":
            return await self._deribit_candles(start_ms, end_ms, interval)

        if self.exchange == "okx":
            return await self._okx_candles(start_ms, end_ms, interval)

        if self.exchange == "cryptocom":
            return await self._cryptocom_candles(start_ms, end_ms, interval)

        raise AssertionError("unreachable")

    async def trades(
        self,
        *,
        start: Any | None = None,
        end: Any | None = None,
        max_rows: int | None = None,
        strict: bool = True,
    ) -> pd.DataFrame:
        """
        Retrieve public trades where the venue permits it.

        strict=True prevents silently presenting a partial public history as
        complete on venues such as Independent Reserve.
        """
        if self.exchange == "independent_reserve" and start is None and end is None:
            start_ms = None
            end_ms = None
        else:
            start_ms, end_ms = _normalise_range(start, end)

        if self.exchange == "binance":
            return await self._binance_trades(start_ms, end_ms, max_rows)

        if self.exchange == "hyperliquid":
            raise DataAvailabilityError(
                "Hyperliquid does not expose a general market-wide historical "
                "trade REST endpoint. The official historical-data docs point "
                "to S3 node fill/trade archives. Use candles() for API history, "
                "or add an S3 bulk-data loader for trade-level backfills."
            )

        if self.exchange == "independent_reserve":
            return await self._independent_reserve_trades(
                start_ms, end_ms, max_rows=max_rows, strict=strict
            )

        if self.exchange == "deribit":
            return await self._deribit_trades(start_ms, end_ms, max_rows)

        if self.exchange == "okx":
            return await self._okx_trades(start_ms, end_ms, max_rows)

        if self.exchange == "cryptocom":
            return await self._cryptocom_trades(start_ms, end_ms, max_rows)

        raise AssertionError("unreachable")

    def save(
        self,
        df: pd.DataFrame,
        *,
        data_type: str,
        root: str | Path = "data/historical",
    ) -> Path:
        """
        Save a DataFrame into a simple exchange/symbol partition.
        """
        safe_symbol = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.symbol)
        out_dir = Path(root) / self.exchange / safe_symbol / data_type
        out_dir.mkdir(parents=True, exist_ok=True)

        if df.empty:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        else:
            start = pd.to_datetime(df["timestamp"].min(), unit="ms", utc=True)
            end = pd.to_datetime(df["timestamp"].max(), unit="ms", utc=True)
            stamp = f"{start:%Y%m%dT%H%M%SZ}_{end:%Y%m%dT%H%M%SZ}"

        path = out_dir / f"{stamp}.parquet"
        df.to_parquet(path, index=False)
        return path

    # ------------------------------------------------------------------
    # Symbol helpers
    # ------------------------------------------------------------------

    def _binance_symbol(self) -> str:
        return re.sub(r"[-_/]", "", self.symbol)

    def _hyperliquid_coin(self) -> str:
        if self.symbol.startswith("@"):
            return self.symbol

        # For common perp notation such as BTC-USDC / BTC-USDT, Hyperliquid's
        # API coin is normally simply BTC.
        return re.split(r"[-_/]", self.symbol)[0]

    def _independent_reserve_pair(self) -> tuple[str, str]:
        parts = [p for p in re.split(r"[-_/]", self.symbol) if p]
        if len(parts) != 2:
            raise ValueError(
                "Independent Reserve symbols must look like BTC-AUD, BTC-USD, etc."
            )

        primary, secondary = parts
        if primary == "BTC":
            primary = "XBT"

        return primary.lower(), secondary.lower()

    def _cryptocom_symbol(self) -> str:
        s = self.symbol

        if s.endswith("-PERP"):
            pieces = s.split("-")
            if len(pieces) >= 3:
                return "".join(pieces[:-1]) + "-PERP"

        if "_" in s:
            return s

        if "-" in s:
            return s.replace("-", "_")

        return s

    # ------------------------------------------------------------------
    # Binance
    # ------------------------------------------------------------------

    def _binance_routes(self) -> tuple[str, str, str, int]:
        if self.market == "spot":
            return (
                "https://data-api.binance.vision",
                "/api/v3/klines",
                "/api/v3/aggTrades",
                1000,
            )
        if self.market == "usdm":
            return (
                "https://fapi.binance.com",
                "/fapi/v1/klines",
                "/fapi/v1/aggTrades",
                1500,
            )
        return (
            "https://dapi.binance.com",
            "/dapi/v1/klines",
            "/dapi/v1/aggTrades",
            1500,
        )

    async def _binance_candles(
        self,
        start_ms: int,
        end_ms: int,
        interval: str,
    ) -> pd.DataFrame:
        base, kline_path, _, max_limit = self._binance_routes()
        symbol = self._binance_symbol()
        step = _interval_ms(interval)

        rows: list[dict[str, Any]] = []
        cursor = start_ms

        while cursor < end_ms:
            payload = await self._request_json(
                "GET",
                base + kline_path,
                params={
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": cursor,
                    "endTime": end_ms,
                    "limit": max_limit,
                },
            )

            if not payload:
                break

            for r in payload:
                rows.append(
                    {
                        "exchange": "binance",
                        "symbol": self.symbol,
                        "timestamp": int(r[0]),
                        "open": float(r[1]),
                        "high": float(r[2]),
                        "low": float(r[3]),
                        "close": float(r[4]),
                        "volume": float(r[5]),
                        "close_timestamp": int(r[6]),
                        "quote_volume": float(r[7]),
                        "trade_count": int(r[8]),
                    }
                )

            last_open = int(payload[-1][0])
            new_cursor = last_open + step
            if new_cursor <= cursor:
                break
            cursor = new_cursor

            if len(payload) < max_limit:
                break

        df = _finalise(pd.DataFrame(rows), dedupe=["timestamp"])
        if not df.empty:
            df = df[(df["timestamp"] >= start_ms) & (df["timestamp"] <= end_ms)]
        return df.reset_index(drop=True)

    async def _binance_trades(
        self,
        start_ms: int,
        end_ms: int,
        max_rows: int | None,
    ) -> pd.DataFrame:
        base, _, trades_path, _ = self._binance_routes()
        symbol = self._binance_symbol()

        rows: list[dict[str, Any]] = []
        first = True
        from_id: int | None = None

        while True:
            if first:
                params = {
                    "symbol": symbol,
                    "startTime": start_ms,
                    "endTime": end_ms,
                    "limit": 1000,
                }
            else:
                params = {
                    "symbol": symbol,
                    "fromId": from_id,
                    "limit": 1000,
                }

            payload = await self._request_json(
                "GET",
                base + trades_path,
                params=params,
            )

            if not payload:
                break

            for t in payload:
                ts = int(t["T"])
                if ts > end_ms:
                    break
                if ts >= start_ms:
                    rows.append(
                        {
                            "exchange": "binance",
                            "symbol": self.symbol,
                            "timestamp": ts,
                            "trade_id": str(t["a"]),
                            "price": float(t["p"]),
                            "amount": float(t["q"]),
                            # m == buyer is maker -> taker was selling
                            "side": "sell" if bool(t["m"]) else "buy",
                            "first_trade_id": t.get("f"),
                            "last_trade_id": t.get("l"),
                            "is_aggregate": True,
                        }
                    )

            if max_rows is not None and len(rows) >= max_rows:
                rows = rows[:max_rows]
                break

            last = payload[-1]
            last_ts = int(last["T"])
            if last_ts >= end_ms:
                break

            next_id = int(last["a"]) + 1
            if from_id is not None and next_id <= from_id:
                break

            from_id = next_id
            first = False

            if len(payload) < 1000:
                break

        return _finalise(pd.DataFrame(rows), dedupe=["trade_id"])

    # ------------------------------------------------------------------
    # Hyperliquid
    # ------------------------------------------------------------------

    async def _hyperliquid_candles(
        self,
        start_ms: int,
        end_ms: int,
        interval: str,
    ) -> pd.DataFrame:
        step = _interval_ms(interval)
        requested = math.ceil((end_ms - start_ms) / step)

        if requested > 5000:
            raise DataAvailabilityError(
                f"Hyperliquid's candleSnapshot exposes only the most recent "
                f"5,000 candles. Your requested range is about {requested:,} "
                f"{interval} candles. Request a shorter range or use another "
                f"historical source."
            )

        payload = await self._request_json(
            "POST",
            "https://api.hyperliquid.xyz/info",
            json={
                "type": "candleSnapshot",
                "req": {
                    "coin": self._hyperliquid_coin(),
                    "interval": interval,
                    "startTime": start_ms,
                    "endTime": end_ms,
                },
            },
        )

        rows = [
            {
                "exchange": "hyperliquid",
                "symbol": self.symbol,
                "timestamp": int(r["t"]),
                "close_timestamp": int(r["T"]),
                "open": float(r["o"]),
                "high": float(r["h"]),
                "low": float(r["l"]),
                "close": float(r["c"]),
                "volume": float(r["v"]),
                "trade_count": int(r.get("n", 0)),
            }
            for r in payload
        ]

        return _finalise(pd.DataFrame(rows), dedupe=["timestamp"])

    # ------------------------------------------------------------------
    # Independent Reserve
    # ------------------------------------------------------------------

    async def _independent_reserve_candles(
        self,
        start_ms: int,
        end_ms: int,
        interval: str,
    ) -> pd.DataFrame:
        if interval != "1h":
            raise DataAvailabilityError(
                "Independent Reserve's public historical summary is fixed at "
                "1-hour intervals. Use interval='1h'."
            )

        now_ms = _utc_now_ms()
        oldest_public_ms = now_ms - 240 * 3_600_000

        if start_ms < oldest_public_ms:
            raise DataAvailabilityError(
                "Independent Reserve's public GetTradeHistorySummary exposes at "
                "most the previous 240 hours."
            )

        primary, secondary = self._independent_reserve_pair()
        hours = min(240, max(1, math.ceil((now_ms - start_ms) / 3_600_000)))

        payload = await self._request_json(
            "GET",
            "https://api.independentreserve.com/Public/GetTradeHistorySummary",
            params={
                "primaryCurrencyCode": primary,
                "secondaryCurrencyCode": secondary,
                "numberOfHoursInThePastToRetrieve": hours,
            },
        )

        rows = []
        for r in payload.get("HistorySummaryItems", []):
            ts = _to_ms(r["StartTimestampUtc"])
            rows.append(
                {
                    "exchange": "independent_reserve",
                    "symbol": self.symbol,
                    "timestamp": ts,
                    "open": float(r["OpeningSecondaryCurrencyPrice"]),
                    "high": float(r["HighestSecondaryCurrencyPrice"]),
                    "low": float(r["LowestSecondaryCurrencyPrice"]),
                    "close": float(r["ClosingSecondaryCurrencyPrice"]),
                    "volume": float(r["PrimaryCurrencyVolume"]),
                    "trade_count": int(r["NumberOfTrades"]),
                }
            )

        df = _finalise(pd.DataFrame(rows), dedupe=["timestamp"])
        if not df.empty:
            df = df[(df["timestamp"] >= start_ms) & (df["timestamp"] <= end_ms)]
        return df.reset_index(drop=True)

    async def _independent_reserve_trades(
        self,
        start_ms: int | None,
        end_ms: int | None,
        *,
        max_rows: int | None,
        strict: bool,
    ) -> pd.DataFrame:
        primary, secondary = self._independent_reserve_pair()
        n = min(50, max_rows or 50)

        payload = await self._request_json(
            "GET",
            "https://api.independentreserve.com/Public/GetRecentTrades",
            params={
                "primaryCurrencyCode": primary,
                "secondaryCurrencyCode": secondary,
                "numberOfRecentTradesToRetrieve": n,
            },
        )

        rows = []
        for t in payload.get("Trades", []):
            ts = _to_ms(t["TradeTimestampUtc"])
            rows.append(
                {
                    "exchange": "independent_reserve",
                    "symbol": self.symbol,
                    "timestamp": ts,
                    "trade_id": str(t["TradeGuid"]),
                    "price": float(t["SecondaryCurrencyTradePrice"]),
                    "amount": float(t["PrimaryCurrencyAmount"]),
                    "side": "buy" if str(t["Taker"]).lower() == "bid" else "sell",
                }
            )

        df = _finalise(pd.DataFrame(rows), dedupe=["trade_id"])

        if start_ms is not None and not df.empty and strict:
            oldest_returned = int(df["timestamp"].min())
            if start_ms < oldest_returned:
                raise DataAvailabilityError(
                    "Independent Reserve only exposes the latest 50 public trades. "
                    "The requested start time predates the oldest returned trade, "
                    "so returning this as complete history would be misleading. "
                    "Set strict=False if you intentionally want only the recent "
                    "subset."
                )

        if not df.empty:
            if start_ms is not None:
                df = df[df["timestamp"] >= start_ms]
            if end_ms is not None:
                df = df[df["timestamp"] <= end_ms]

        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Deribit
    # ------------------------------------------------------------------

    @staticmethod
    def _deribit_resolution(interval: str) -> str:
        mapping = {
            "1m": "1",
            "3m": "3",
            "5m": "5",
            "10m": "10",
            "15m": "15",
            "30m": "30",
            "1h": "60",
            "2h": "120",
            "3h": "180",
            "6h": "360",
            "12h": "720",
            "1d": "1D",
        }
        if interval not in mapping:
            raise ValueError(
                f"Unsupported Deribit interval {interval!r}. "
                f"Supported: {sorted(mapping)}"
            )
        return mapping[interval]

    async def _deribit_candles(
        self,
        start_ms: int,
        end_ms: int,
        interval: str,
    ) -> pd.DataFrame:
        payload = await self._request_json(
            "GET",
            "https://www.deribit.com/api/v2/public/get_tradingview_chart_data",
            params={
                "instrument_name": self.symbol,
                "start_timestamp": start_ms,
                "end_timestamp": end_ms,
                "resolution": self._deribit_resolution(interval),
            },
        )

        result = payload.get("result", {})
        if result.get("status") == "no_data":
            return pd.DataFrame()

        ticks = result.get("ticks", [])
        rows = []

        for i, ts in enumerate(ticks):
            rows.append(
                {
                    "exchange": "deribit",
                    "symbol": self.symbol,
                    "timestamp": int(ts),
                    "open": float(result["open"][i]),
                    "high": float(result["high"][i]),
                    "low": float(result["low"][i]),
                    "close": float(result["close"][i]),
                    "volume": float(result["volume"][i]),
                }
            )

        return _finalise(pd.DataFrame(rows), dedupe=["timestamp"])

    async def _deribit_trades(
        self,
        start_ms: int,
        end_ms: int,
        max_rows: int | None,
    ) -> pd.DataFrame:
        rows = []
        cursor = start_ms
        last_trade_id = None

        while cursor <= end_ms:
            payload = await self._request_json(
                "GET",
                "https://www.deribit.com/api/v2/public/get_last_trades_by_instrument_and_time",
                params={
                    "instrument_name": self.symbol,
                    "start_timestamp": cursor,
                    "end_timestamp": end_ms,
                    "count": 1000,
                    "sorting": "asc",
                },
            )

            result = payload.get("result", {})
            trades = result.get("trades", [])

            if not trades:
                break

            for t in trades:
                rows.append(
                    {
                        "exchange": "deribit",
                        "symbol": self.symbol,
                        "timestamp": int(t["timestamp"]),
                        "trade_id": str(t.get("trade_id")),
                        "price": float(t["price"]),
                        "amount": float(t["amount"]),
                        "side": str(t["direction"]).lower(),
                        "index_price": (
                            float(t["index_price"])
                            if t.get("index_price") is not None
                            else None
                        ),
                        "mark_price": (
                            float(t["mark_price"])
                            if t.get("mark_price") is not None
                            else None
                        ),
                    }
                )

            if max_rows is not None and len(rows) >= max_rows:
                rows = rows[:max_rows]
                break

            last = trades[-1]
            last_ts = int(last["timestamp"])
            current_trade_id = str(last.get("trade_id"))

            if last_ts >= end_ms or not result.get("has_more", False):
                break

            # Use the same millisecond once, then dedupe by trade_id.
            new_cursor = last_ts
            if new_cursor == cursor and current_trade_id == last_trade_id:
                new_cursor += 1

            cursor = new_cursor
            last_trade_id = current_trade_id

        return _finalise(pd.DataFrame(rows), dedupe=["trade_id"])

    # ------------------------------------------------------------------
    # OKX
    # ------------------------------------------------------------------

    async def _okx_candles(
        self,
        start_ms: int,
        end_ms: int,
        interval: str,
    ) -> pd.DataFrame:
        rows = []
        cursor = end_ms + 1

        while cursor > start_ms:
            payload = await self._request_json(
                "GET",
                "https://www.okx.com/api/v5/market/history-candles",
                params={
                    "instId": self.symbol,
                    "bar": interval,
                    "after": str(cursor),
                    "limit": "300",
                },
            )

            data = payload.get("data", [])
            if not data:
                break

            min_ts = None

            for r in data:
                ts = int(r[0])
                min_ts = ts if min_ts is None else min(min_ts, ts)

                if start_ms <= ts <= end_ms:
                    rows.append(
                        {
                            "exchange": "okx",
                            "symbol": self.symbol,
                            "timestamp": ts,
                            "open": float(r[1]),
                            "high": float(r[2]),
                            "low": float(r[3]),
                            "close": float(r[4]),
                            "volume": float(r[5]),
                            "volume_ccy": float(r[6]) if len(r) > 6 else None,
                            "volume_quote": float(r[7]) if len(r) > 7 else None,
                            "confirmed": int(r[8]) if len(r) > 8 else None,
                        }
                    )

            if min_ts is None or min_ts >= cursor:
                break

            cursor = min_ts

            if min_ts <= start_ms:
                break

        return _finalise(pd.DataFrame(rows), dedupe=["timestamp"])

    async def _okx_trades(
        self,
        start_ms: int,
        end_ms: int,
        max_rows: int | None,
    ) -> pd.DataFrame:
        rows = []
        cursor = end_ms + 1

        while cursor > start_ms:
            payload = await self._request_json(
                "GET",
                "https://www.okx.com/api/v5/market/history-trades",
                params={
                    "instId": self.symbol,
                    "type": "2",  # timestamp pagination
                    "after": str(cursor),
                    "limit": "100",
                },
            )

            data = payload.get("data", [])
            if not data:
                break

            min_ts = None

            for t in data:
                ts = int(t["ts"])
                min_ts = ts if min_ts is None else min(min_ts, ts)

                if start_ms <= ts <= end_ms:
                    rows.append(
                        {
                            "exchange": "okx",
                            "symbol": self.symbol,
                            "timestamp": ts,
                            "trade_id": str(t["tradeId"]),
                            "price": float(t["px"]),
                            "amount": float(t["sz"]),
                            "side": str(t["side"]).lower(),
                            "source": t.get("source"),
                        }
                    )

            if max_rows is not None and len(rows) >= max_rows:
                rows = rows[:max_rows]
                break

            if min_ts is None or min_ts >= cursor:
                break

            cursor = min_ts

            if min_ts <= start_ms:
                break

        return _finalise(pd.DataFrame(rows), dedupe=["trade_id"])

    # ------------------------------------------------------------------
    # Crypto.com
    # ------------------------------------------------------------------

    async def _cryptocom_candles(
        self,
        start_ms: int,
        end_ms: int,
        interval: str,
    ) -> pd.DataFrame:
        rows = []
        cursor = end_ms

        while cursor > start_ms:
            payload = await self._request_json(
                "GET",
                "https://api.crypto.com/exchange/v1/public/get-candlestick",
                params={
                    "instrument_name": self._cryptocom_symbol(),
                    "timeframe": interval,
                    "count": 100,
                    "start_ts": start_ms,
                    "end_ts": cursor,
                },
            )

            if int(payload.get("code", 0)) != 0:
                raise HistoricalDataError(f"Crypto.com returned error: {payload}")

            data = payload.get("result", {}).get("data", [])
            if not data:
                break

            min_ts = None

            for r in data:
                ts = int(r["t"])
                min_ts = ts if min_ts is None else min(min_ts, ts)

                if start_ms <= ts <= end_ms:
                    rows.append(
                        {
                            "exchange": "cryptocom",
                            "symbol": self.symbol,
                            "timestamp": ts,
                            "open": float(r["o"]),
                            "high": float(r["h"]),
                            "low": float(r["l"]),
                            "close": float(r["c"]),
                            "volume": float(r["v"]),
                        }
                    )

            if min_ts is None or min_ts >= cursor:
                break

            cursor = min_ts - 1

            if min_ts <= start_ms:
                break

        return _finalise(pd.DataFrame(rows), dedupe=["timestamp"])

    async def _cryptocom_trades(
        self,
        start_ms: int,
        end_ms: int,
        max_rows: int | None,
    ) -> pd.DataFrame:
        rows = []
        cursor = end_ms

        while cursor > start_ms:
            payload = await self._request_json(
                "GET",
                "https://api.crypto.com/exchange/v1/public/get-trades",
                params={
                    "instrument_name": self._cryptocom_symbol(),
                    "count": 100,
                    "start_ts": start_ms,
                    "end_ts": cursor,
                },
            )

            if int(payload.get("code", 0)) != 0:
                raise HistoricalDataError(f"Crypto.com returned error: {payload}")

            data = payload.get("result", {}).get("data", [])
            if not data:
                break

            min_ts = None

            for t in data:
                ts = int(t["t"])
                min_ts = ts if min_ts is None else min(min_ts, ts)

                if start_ms <= ts <= end_ms:
                    rows.append(
                        {
                            "exchange": "cryptocom",
                            "symbol": self.symbol,
                            "timestamp": ts,
                            "trade_id": str(t.get("d")),
                            "price": float(t["p"]),
                            "amount": float(t["q"]),
                            "side": str(t["s"]).lower(),
                        }
                    )

            if max_rows is not None and len(rows) >= max_rows:
                rows = rows[:max_rows]
                break

            if min_ts is None or min_ts >= cursor:
                break

            cursor = min_ts - 1

            if min_ts <= start_ms:
                break

        return _finalise(pd.DataFrame(rows), dedupe=["trade_id"])


__all__ = [
    "HistoricalData",
    "HistoricalDataError",
    "DataAvailabilityError",
    "CAPABILITIES",
]
