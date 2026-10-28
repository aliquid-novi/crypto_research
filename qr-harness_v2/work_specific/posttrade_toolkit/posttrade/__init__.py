"""posttrade: exchange-agnostic post-trade analytics for spread trading and passive quoting.

Conventions used everywhere (repeated in every docstring that depends on them):

* Timestamps are int64 nanoseconds since the Unix epoch, UTC. ``schemas.prepare`` converts
  whatever you have (tz-aware datetimes, ms/us/s integers) into that.
* ``side`` / ``d`` = +1 for OUR buy, -1 for OUR sell. For market trades, ``aggressor_side``
  is +1 when the taker bought.
* markout_h = d * (m_{t+h} - p). Positive = good for us. Reported in bps of the fill price
  (= bps of notional) unless a docstring says otherwise.
* fee is signed in quote currency from our point of view of COST: positive = we paid,
  negative = rebate received.
* funding ``payment`` is signed from our point of view of CASH: positive = we received.
  A positive funding ``rate`` means longs pay shorts.

Venue specifics (sessions, multipliers, tick sizes, side encodings, timestamp units, fee
models) live in ``config.py`` and the optional ``instruments`` table, never in the maths.
"""

__version__ = "0.1.0"
