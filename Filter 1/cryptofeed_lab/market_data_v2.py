
"""
market_data.py

Live Cryptofeed collector:
- normalised trades + top-of-book features -> batched Parquet
- optional raw feed capture -> Cryptofeed PCAP recording

Dependencies:
    pip install cryptofeed pandas pyarrow

Example:
    from market_data import MarketDataCollector

    collector = MarketDataCollector(
        exchange="BINANCE",
        symbol="BTC-USDT",
        record_raw=True,
    )

    await collector.collect(60)

    print(collector.stats)
    print(collector.pcap_path)
"""

from __future__ import annotations

import asyncio
import re
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from cryptofeed import FeedHandler
from cryptofeed.defines import TRADES, L2_BOOK
from cryptofeed.exchanges import EXCHANGE_MAP


def _safe_path_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


class ParquetWriter:
    def __init__(
        self,
        root="data/raw/normalised",
        batch_size=5000,
        flush_interval=5.0,
        max_queue_size=100_000,
    ):
        self.root = Path(root)
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.queue = asyncio.Queue(maxsize=max_queue_size)
        self.buffers = defaultdict(list)
        self.task = None
        self.dropped_events = 0
        self._sentinel = object()

    async def start(self):
        if self.task is None:
            self.task = asyncio.create_task(self._run())

    async def put(self, event_type, row):
        try:
            self.queue.put_nowait((event_type, row))
        except asyncio.QueueFull:
            self.dropped_events += 1

    async def stop(self):
        if self.task is None:
            return

        await self.queue.put(self._sentinel)
        await self.task
        self.task = None

    async def _run(self):
        last_flush = time.monotonic()

        while True:
            timeout = max(
                0.01,
                self.flush_interval - (time.monotonic() - last_flush),
            )

            try:
                item = await asyncio.wait_for(self.queue.get(), timeout=timeout)
            except asyncio.TimeoutError:
                await self._flush_all()
                last_flush = time.monotonic()
                continue

            if item is self._sentinel:
                await self._flush_all()
                return

            event_type, row = item
            # Buffer per stream as well as event type so a single Parquet file
            # never accidentally mixes exchanges/symbols.
            key = (event_type, row["exchange"], row["symbol"])
            self.buffers[key].append(row)

            if len(self.buffers[key]) >= self.batch_size:
                await self._flush(key)
                last_flush = time.monotonic()

    async def _flush_all(self):
        for key in list(self.buffers.keys()):
            if self.buffers[key]:
                await self._flush(key)

    async def _flush(self, key):
        rows = self.buffers[key]
        if not rows:
            return

        self.buffers[key] = []
        await asyncio.to_thread(self._write_parquet, key, rows)

    def _write_parquet(self, key, rows):
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
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = (
            f"part-{time.time_ns()}-{uuid.uuid4().hex[:8]}.parquet"
        )

        df.to_parquet(output_dir / filename, index=False)


class MarketDataCollector:
    def __init__(
        self,
        markets,
        *,
        data_dir="data/raw/normalised",
        store_trades=True,
        store_book=True,
        batch_size=5000,
        flush_interval=5.0,
        record_raw=False,
        pcap_dir="data/raw/pcap",
    ):
        """
        Parameters
        ----------
        markets:
            Either:

            [
                ("BINANCE", "BTC-USDT"),
                ("HYPERLIQUID", "BTC"),
            ]

            or a single tuple:

            ("BINANCE", "BTC-USDT")

        Example
        -------
        collector = MarketDataCollector(
            markets=[
                ("BINANCE", "BTC-USDT"),
                ("HYPERLIQUID", "BTC"),
            ],
            record_raw=True,
        )
        """

        # -------------------------------------------------------------
        # Normalise markets
        # -------------------------------------------------------------

        # Allow:
        # ("BINANCE", "BTC-USDT")
        #
        # instead of forcing:
        # [("BINANCE", "BTC-USDT")]
        if (
            isinstance(markets, tuple)
            and len(markets) == 2
            and isinstance(markets[0], str)
        ):
            markets = [markets]

        if not isinstance(markets, (list, tuple)):
            raise TypeError(
                "markets must be a tuple or list of tuples"
            )

        self.markets = []

        for exchange, symbol in markets:

            exchange = exchange.upper()
            symbol = symbol.upper()

            if exchange not in EXCHANGE_MAP:

                available = ", ".join(
                    sorted(EXCHANGE_MAP.keys())
                )

                raise ValueError(
                    f"Unknown exchange: {exchange}\n\n"
                    f"Available exchanges:\n{available}"
                )

            self.markets.append(
                {
                    "exchange": exchange,
                    "symbol": symbol,
                }
            )

        # -------------------------------------------------------------
        # Configuration
        # -------------------------------------------------------------

        self.store_trades = store_trades
        self.store_book = store_book

        self.writer = ParquetWriter(
            root=data_dir,
            batch_size=batch_size,
            flush_interval=flush_interval,
        )

        # Raw PCAP capture
        self.record_raw = record_raw
        self.pcap_root = Path(pcap_dir)
        self.pcap_path = None

        # Feed lifecycle
        self.feed_handler = None
        self.feed_task = None

        # -------------------------------------------------------------
        # Statistics
        # -------------------------------------------------------------

        # Stats per exchange/symbol
        self.trade_counts = defaultdict(int)
        self.book_counts = defaultdict(int)

    def _new_pcap_session_dir(self) -> Path:
        """
        Build a unique directory for one recording session.

        Cryptofeed accepts a directory in FeedHandler(record=...).
        It records the raw feed stream there as compressed PCAP and writes
        metadata alongside it.
        """
        session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")

        path = (
            self.pcap_root
            / _safe_path_component(self.exchange)
            / _safe_path_component(self.symbol)
            / session
        )

        path.mkdir(parents=True, exist_ok=True)
        return path

    async def _trade_callback(self, trade, receipt_timestamp):
        self.trade_count += 1

        row = {
            "exchange": trade.exchange,
            "symbol": trade.symbol,
            "side": trade.side,
            "price": float(trade.price),
            "amount": float(trade.amount),
            "exchange_timestamp": trade.timestamp,
            "receipt_timestamp": receipt_timestamp,
        }

        await self.writer.put("trades", row)

    async def _book_callback(self, book, receipt_timestamp):
        if not len(book.book.bids) or not len(book.book.asks):
            return

        self.book_count += 1

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

        microprice = (
            (ask_price * bid_size + bid_price * ask_size) / total_size
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
            "spread_bps": spread / mid * 10_000,
            "imbalance": imbalance,
            "microprice": microprice,
            "exchange_timestamp": book.timestamp,
            "receipt_timestamp": receipt_timestamp,
        }

        await self.writer.put("book", row)

    async def start(self):

        if self.feed_handler is not None:
            raise RuntimeError(
                "Collector already started."
            )

        # Start background Parquet writer
        await self.writer.start()

        # -------------------------------------------------------------
        # Raw PCAP session
        # -------------------------------------------------------------

        record_path = None

        if self.record_raw:

            self.pcap_path = (
                self._new_pcap_session_dir()
            )

            record_path = (
                str(self.pcap_path)
                + "/"
            )

        # -------------------------------------------------------------
        # ONE FeedHandler
        # -------------------------------------------------------------

        self.feed_handler = FeedHandler(
            on_feed_error="remove_feed",
            record=record_path,
        )

        # -------------------------------------------------------------
        # Channels shared by all feeds
        # -------------------------------------------------------------

        channels = []
        callbacks = {}

        if self.store_trades:

            channels.append(TRADES)

            callbacks[TRADES] = (
                self._trade_callback
            )

        if self.store_book:

            channels.append(L2_BOOK)

            callbacks[L2_BOOK] = (
                self._book_callback
            )

        # -------------------------------------------------------------
        # Add every market to the SAME FeedHandler
        # -------------------------------------------------------------

        for market in self.markets:

            exchange = market["exchange"]
            symbol = market["symbol"]

            self.feed_handler.add_feed(
                exchange,
                symbols=[symbol],
                channels=channels,
                callbacks=callbacks,
            )

            print(
                f"Added {exchange} {symbol}"
            )

        # -------------------------------------------------------------
        # Start all feeds concurrently
        # -------------------------------------------------------------

        self.feed_task = asyncio.create_task(
            self.feed_handler.run_async(
                install_signal_handlers=False
            )
        )

        print(
            f"\nStarted "
            f"{len(self.markets)} market(s)"
        )

        if self.record_raw:

            print(
                f"Raw PCAP directory: "
                f"{self.pcap_path}"
            )

    async def stop(self):
        if self.feed_handler is None:
            return

        print("Stopping feed...")
        self.feed_handler.request_stop()
        await self.feed_task

        print("Flushing Parquet storage...")
        await self.writer.stop()

        print("Stopped.")

        self.feed_handler = None
        self.feed_task = None

    async def collect(self, seconds):
        await self.start()

        try:
            await asyncio.sleep(seconds)
        finally:
            await self.stop()

    @property
    def stats(self):

        market_stats = {}

        for market in self.markets:

            key = (
                market["exchange"],
                market["symbol"],
            )

            name = (
                f"{market['exchange']}:"
                f"{market['symbol']}"
            )

            market_stats[name] = {
                "trades":
                    self.trade_counts[key],

                "book_updates":
                    self.book_counts[key],
            }

        return {
            "markets": market_stats,

            "queue_size":
                self.writer.queue.qsize(),

            "dropped_events":
                self.writer.dropped_events,

            "record_raw":
                self.record_raw,

            "pcap_path":
                str(self.pcap_path)
                if self.pcap_path
                else None,
        }


__all__ = ["MarketDataCollector", "ParquetWriter"]
