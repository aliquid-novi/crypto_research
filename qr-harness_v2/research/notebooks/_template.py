# %% [markdown]
# # {{ID}}: {{TITLE}}
# Card: `research/hypotheses/{{CARD}}`. **Prediction, falsification condition and confidence are on the card, are Mark's,
# and are locked.** This notebook only implements and measures.

# %% [markdown]
# ## Summary
# _Filled in after the run (results tables, key takeaways, data notes), matching the chat summary. Weak points first._

# %%
# ---- gate: refuses to run unless the card is registered and unchanged ----
import sys
from pathlib import Path

ROOT = Path.cwd()
while not (ROOT / "qrlib").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qrlib import audit, data, fairvalue, fills, leakage, pit, prereg, stats

HID = "{{ID}}"
CARD = prereg.require(HID)
print(f"{HID} registered {CARD['registered_at']}; trials so far: {audit.n_trials(HID)}")

# %% [markdown]
# ## KNOBS
# Everything tunable. `# ASSUMPTION:` marks every choice Mark didn't specify on the card.

# %%
VENUE, SYMBOL = "HYPERLIQUID", "HYPE-USD-PERP"
START, END = None, None          # ASSUMPTION: whole locally covered sample
LATENCY = "0ms"                  # extra delay on top of ts_recv before an event is usable
HORIZONS = ["250ms", "500ms", "1s"]
TRAIN_FRAC = 0.6                 # ASSUMPTION: one chronological split
GRID = "100ms"                   # decision grid
SEED = 0

# %% [markdown]
# ## Data
# Loaded through `qrlib.data`. Features join on `ts_recv + LATENCY`, strictly backward, via `qrlib.pit` (rule 3).

# %%
print(data.coverage(VENUE, SYMBOL).tail(24))
quotes = data.load("l1", VENUE, SYMBOL, START, END)
trades = data.load("trades", VENUE, SYMBOL, START, END)
print(leakage.hygiene(quotes, name="l1"))
print(leakage.hygiene(trades, name="trades"))

# %% [markdown]
# ## Model
# One function holds the **whole** estimator, including any fitting or normalisation (rule 4). Inputs: the data dict and the
# decision times. Output: one row per time.

# %%
T_ALL = pd.date_range(quotes.ts_recv.iloc[0] + pd.Timedelta("1s"), quotes.ts_recv.iloc[-1] - pd.Timedelta("10s"), freq=GRID)
TRAIN, TEST, T_SPLIT = pit.chrono_split(T_ALL, TRAIN_FRAC)

def model(d, t):
    book = fairvalue.l1_at(d["quotes"], t, latency=LATENCY)
    return pd.DataFrame({"mid": fairvalue.mid(book), "weighted mid": fairvalue.weighted_mid(book)})

# %% [markdown]
# ## Leakage gate
# Must pass before any result is reported. If it fails, report the failure; don't work around it.

# %%
rep = leakage.truncation_test(model, {"quotes": quotes, "trades": trades}, T_ALL[: min(len(T_ALL), 20_000)], latency=LATENCY)
print(rep)
_ = audit.log_event("leakage", HID, passed=rep.passed, cuts=len(rep.cuts))

# %% [markdown]
# ## Evaluation (test only)

# %%
FV = model({"quotes": quotes}, T_ALL[TEST])
ev = fairvalue.evaluate(FV.drop(columns="mid"), quotes, T_ALL[TEST], HORIZONS, latency=LATENCY)
for (h, fv), r in ev.iterrows():
    audit.log_run(HID, f"{fv} @ {h}", params={"latency": LATENCY, "grid": GRID}, metrics=r.to_dict(), notebook=f"{HID}")
ev

# %% [markdown]
# ## Fills / markouts (quoting hypotheses only; always both bounds, rule 5)

# %%
# my_q = pd.DataFrame({"my_bid": ..., "my_ask": ...}, index=decision_times)
# for mode in ("through", "touch"):
#     f = fills.simulate_fills(my_q, trades, live_delay=LATENCY, mode=mode)
#     mk = fills.markouts(f, quotes)
#     audit.log_run(HID, f"quoter X [{mode}]", metrics=mk.mean().to_dict())

# %% [markdown]
# ## Trials

# %%
print(f"n_trials for {HID}: {audit.n_trials(HID)}")

# %% [markdown]
# ## Findings
# _Written to match the tables. Weak points first; CIs, n, n_trials; both fill bounds; "not distinguishable from mid" is a result._
#
# ## Assumptions (every `# ASSUMPTION:` above, listed)
#
# ## Open questions for Mark
