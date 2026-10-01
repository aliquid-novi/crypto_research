
"""
market_data.py

Live Cryptofeed collector supporting one or many markets simultaneously.

Features
--------
- One FeedHandler for all configured markets.
- Normalised trades + top-of-book features.
- Non-blocking asyncio queue between callbacks and storage.
- Batched Parquet writes in a background thread.
- Optional raw Cryptofeed PCAP recording for the whole collection session.
- Per-market statistics.

Dependencies
------------
pip install cryptofeed pandas pyarrow

Examples
--------
Single market:

    collector = MarketDataCollector(
        markets=("BINANCE", "BTC-USDT"),
        record_raw=True,
    )

    await collector.collect(60)

Multiple markets:

    collector = MarketDataCollector(
        markets=[
            ("BINANCE", "BTC-USDT"),
            ("HYPERLIQUID", "BTC"),
        ],
        record_raw=True,
    )

    await collector.collect(60)

    print(collector.stats)
"""

from __future__ import annotations

import asyncio
import re
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from cryptofeed import FeedHandler
from cryptofeed.defines import TRADES, L2_BOOK
from cryptofeed.exchanges import EXCHANGE_MAP


def _safe_path_component(value: str) -> str:
    """Make an exchange/symbol safe to use as a directory name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


class ParquetWriter:
    """
    Background batched Parquet writer.

    Market-data callbacks only enqueue rows. Disk I/O is performed separately,
    so a slow Parquet write does not directly block the websocket callback.
    """

    def __init__(
        self,
        root: str | Path = "data/raw/normalised",
        batch_size: int = 5000,
        flush_interval: float = 5.0,
        max_queue_size: int = 100_000,
    ):
        self.root = Path(root)
        self.batch_size = batch_size
        self.flush_interval = flush_interval

        self.queue = asyncio.Queue(maxsize=max_queue_size)
        self.buffers = defaultdict(list)

        self.task: asyncio.Task | None = None
        self.dropped_events = 0

        self._sentinel = object()

    async def start(self) -> None:
        if self.task is None:
            self.task = asyncio.create_task(self._run())

    async def put(self, event_type: str, row: dict) -> None:
        """
        Enqueue without waiting.

        If storage falls badly behind and the queue fills, drop the event and
        increment dropped_events rather than blocking the websocket callback.
        """
        try:
            self.queue.put_nowait((event_type, row))
        except asyncio.QueueFull:
            self.dropped_events += 1

    async def stop(self) -> None:
        if self.task is None:
            return

        await self.queue.put(self._sentinel)
        await self.task
        self.task = None

    async def _run(self) -> None:
        last_flush = time.monotonic()

        while True:
            timeout = max(
                0.01,
                self.flush_interval - (time.monotonic() - last_flush),
            )

            try:
                item = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                await self._flush_all()
                last_flush = time.monotonic()
                continue

            if item is self._sentinel:
                await self._flush_all()
                return

            event_type, row = item

            # Keep exchange/symbol streams separate on disk.
            key = (
                event_type,
                row["exchange"],
                row["symbol"],
            )

            self.buffers[key].append(row)

            if len(self.buffers[key]) >= self.batch_size:
                await self._flush(key)
                last_flush = time.monotonic()

    async def _flush_all(self) -> None:
        for key in list(self.buffers.keys()):
            if self.buffers[key]:
                await self._flush(key)

    async def _flush(self, key: tuple[str, str, str]) -> None:
        rows = self.buffers[key]

        if not rows:
            return

        # Swap the live buffer out immediately.
        self.buffers[key] = []

        # Pandas/Parquet disk I/O is blocking, so move it off the event loop.
        await asyncio.to_thread(
            self._write_parquet,
            key,
            rows,
        )

    def _write_parquet(
        self,
        key: tuple[str, str, str],
        rows: list[dict],
    ) -> None:
        event_type, exchange, symbol = key

        df = pd.DataFrame(rows)

        date = pd.to_datetime(
            rows[0]["receipt_timestamp"],
            unit="s",
            utc=True,
        ).strftime("%Y-%m-%d")

        output_dir = (
            self.root
            / _safe_path_component(exchange)
            / _safe_path_component(symbol)
            / event_type
            / date
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        filename = (
            f"part-{time.time_ns()}-"
            f"{uuid.uuid4().hex[:8]}.parquet"
        )

        df.to_parquet(
            output_dir / filename,
            index=False,
        )


class MarketDataCollector:
    """
    Collect one or more Cryptofeed markets concurrently.

    `markets` may be either:

        ("BINANCE", "BTC-USDT")

    or:

        [
            ("BINANCE", "BTC-USDT"),
            ("HYPERLIQUID", "BTC"),
        ]

    Internally, both cases are represented as a list of markets. There is no
    separate single-market vs multi-market code path.
    """

    def __init__(
        self,
        markets,
        *,
        data_dir: str | Path = "data/raw/normalised",
        store_trades: bool = True,
        store_book: bool = True,
        batch_size: int = 5000,
        flush_interval: float = 5.0,
        max_queue_size: int = 100_000,
        record_raw: bool = False,
        pcap_dir: str | Path = "data/raw/pcap",
    ):
        self.markets = self._normalise_markets(markets)

        if not store_trades and not store_book:
            raise ValueError(
                "At least one of store_trades or store_book must be True."
            )

        self.store_trades = store_trades
        self.store_book = store_book

        self.writer = ParquetWriter(
            root=data_dir,
            batch_size=batch_size,
            flush_interval=flush_interval,
            max_queue_size=max_queue_size,
        )

        # Raw PCAP capture for the entire multi-market session.
        self.record_raw = record_raw
        self.pcap_root = Path(pcap_dir)
        self.pcap_path: Path | None = None

        # Cryptofeed lifecycle.
        self.feed_handler: FeedHandler | None = None
        self.feed_task: asyncio.Task | None = None

        # Per-market counters.
        self.trade_counts = defaultdict(int)
        self.book_counts = defaultdict(int)

    @staticmethod
    def _normalise_markets(markets) -> list[tuple[str, str]]:
        """
        Convert single-market and multi-market inputs into one validated format.
        """

        # Single tuple:
        # ("BINANCE", "BTC-USDT")
        if (
            isinstance(markets, tuple)
            and len(markets) == 2
            and all(isinstance(x, str) for x in markets)
        ):
            markets = [markets]

        if not isinstance(markets, (list, tuple)):
            raise TypeError(
                "markets must be ('EXCHANGE', 'SYMBOL') or a list of such tuples."
            )

        if len(markets) == 0:
            raise ValueError("markets cannot be empty.")

        normalised = []

        for item in markets:
            if (
                not isinstance(item, (list, tuple))
                or len(item) != 2
                or not all(isinstance(x, str) for x in item)
            ):
                raise TypeError(
                    "Each market must be a two-item tuple/list: "
                    "('EXCHANGE', 'SYMBOL')."
                )

            exchange, symbol = item

            exchange = exchange.upper().strip()
            symbol = symbol.upper().strip()

            if exchange not in EXCHANGE_MAP:
                available = ", ".join(sorted(EXCHANGE_MAP.keys()))
                raise ValueError(
                    f"Unknown exchange: {exchange}\n\n"
                    f"Available exchanges:\n{available}"
                )

            if not symbol:
                raise ValueError(
                    f"Symbol cannot be empty for exchange {exchange}."
                )

            pair = (exchange, symbol)

            # Avoid accidentally opening the exact same feed twice.
            if pair not in normalised:
                normalised.append(pair)

        return normalised

    def _new_pcap_session_dir(self) -> Path:
        """
        Create one raw-recording directory for the entire collection session.

        This deliberately does NOT use self.exchange/self.symbol because the
        collector can now contain many markets.
        """
        session = datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%S_%fZ"
        )

        path = self.pcap_root / session

        path.mkdir(
            parents=True,
            exist_ok=True,
        )

        return path

    async def _trade_callback(
        self,
        trade,
        receipt_timestamp,
    ) -> None:
        key = (
            trade.exchange,
            trade.symbol,
        )

        self.trade_counts[key] += 1

        row = {
            "exchange": trade.exchange,
            "symbol": trade.symbol,
            "side": trade.side,
            "price": float(trade.price),
            "amount": float(trade.amount),
            "exchange_timestamp": trade.timestamp,
            "receipt_timestamp": receipt_timestamp,
        }

        await self.writer.put(
            "trades",
            row,
        )

    async def _book_callback(
        self,
        book,
        receipt_timestamp,
    ) -> None:
        if (
            not len(book.book.bids)
            or not len(book.book.asks)
        ):
            return

        key = (
            book.exchange,
            book.symbol,
        )

        self.book_counts[key] += 1

        bid_price, bid_size = book.book.bids.index(0)
        ask_price, ask_size = book.book.asks.index(0)

        bid_price = float(bid_price)
        ask_price = float(ask_price)
        bid_size = float(bid_size)
        ask_size = float(ask_size)

        mid = (bid_price + ask_price) / 2
        spread = ask_price - bid_price

        total_size = bid_size + ask_size

        imbalance = (
            (bid_size - ask_size) / total_size
            if total_size > 0
            else 0.0
        )

        # L1 microprice:
        # larger bid size pulls the estimate toward the ask and vice versa.
        microprice = (
            (
                ask_price * bid_size
                + bid_price * ask_size
            )
            / total_size
            if total_size > 0
            else mid
        )

        row = {
            "exchange": book.exchange,
            "symbol": book.symbol,
            "bid": bid_price,
            "bid_size": bid_size,
            "ask": ask_price,
            "ask_size": ask_size,
            "mid": mid,
            "spread": spread,
            "spread_bps": (
                spread / mid * 10_000
                if mid != 0
                else None
            ),
            "imbalance": imbalance,
            "microprice": microprice,
            "exchange_timestamp": book.timestamp,
            "receipt_timestamp": receipt_timestamp,
        }

        await self.writer.put(
            "book",
            row,
        )

    async def start(self) -> None:
        if self.feed_handler is not None:
            raise RuntimeError(
                "Collector already started."
            )

        await self.writer.start()

        record_path = None

        if self.record_raw:
            self.pcap_path = self._new_pcap_session_dir()

            # Cryptofeed treats this as the recording directory.
            record_path = str(self.pcap_path) + "/"

        # ONE FeedHandler owns ALL configured feeds.
        self.feed_handler = FeedHandler(
            on_feed_error="remove_feed",
            record=record_path,
        )

        channels = []
        callbacks = {}

        if self.store_trades:
            channels.append(TRADES)
            callbacks[TRADES] = self._trade_callback

        if self.store_book:
            channels.append(L2_BOOK)
            callbacks[L2_BOOK] = self._book_callback

        # Add each market to the same FeedHandler.
        for exchange, symbol in self.markets:
            self.feed_handler.add_feed(
                exchange,
                symbols=[symbol],
                channels=channels,
                callbacks=callbacks,
            )

            print(
                f"Added {exchange} {symbol}"
            )

        # Start all websocket feeds concurrently in the current asyncio loop.
        self.feed_task = asyncio.create_task(
            self.feed_handler.run_async(
                install_signal_handlers=False
            )
        )

        print(
            f"\nStarted {len(self.markets)} market(s)."
        )

        if self.record_raw:
            print(
                f"Raw PCAP directory: {self.pcap_path}"
            )

    async def stop(self) -> None:
        if self.feed_handler is None:
            return

        print("Stopping feeds...")

        self.feed_handler.request_stop()

        if self.feed_task is not None:
            await self.feed_task

        print("Flushing Parquet storage...")

        await self.writer.stop()

        self.feed_handler = None
        self.feed_task = None

        print("Stopped.")

    async def collect(
        self,
        seconds: float,
    ) -> None:
        if seconds <= 0:
            raise ValueError(
                "seconds must be greater than zero."
            )

        await self.start()

        try:
            await asyncio.sleep(seconds)

        finally:
            await self.stop()

    @property
    def stats(self) -> dict:
        market_stats = {}

        for exchange, symbol in self.markets:
            key = (
                exchange,
                symbol,
            )

            name = f"{exchange}:{symbol}"

            market_stats[name] = {
                "trades": self.trade_counts[key],
                "book_updates": self.book_counts[key],
            }

        return {
            "markets": market_stats,
            "queue_size": self.writer.queue.qsize(),
            "dropped_events": self.writer.dropped_events,
            "record_raw": self.record_raw,
            "pcap_path": (
                str(self.pcap_path)
                if self.pcap_path
                else None
            ),
        }


__all__ = [
    "MarketDataCollector",
    "ParquetWriter",
]
