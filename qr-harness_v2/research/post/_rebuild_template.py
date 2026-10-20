# %% [markdown]
# # Rebuild: {{NAME}}
# Original notebook: `{{NOTEBOOK}}`. Card: {{CARD}}. Review: `research/post/{{NAME}}_post.md` (Part C).
#
# **Mark writes this.** Rebuild one key number from scratch: same definition, smallest data slice, no copying from the
# original notebook. If your number disagrees, the disagreement is the finding. Work out which of you is wrong before
# trusting either.

# %%
import sys
from pathlib import Path

ROOT = Path.cwd()
while not (ROOT / "qrlib").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qrlib import data   # data.load(table, venue, symbol, start, end) -> standardised frame, sorted by ts_recv

# %% [markdown]
# ## 1. The target
# The number you are rebuilding, copied from the original notebook, with the cell it came from and its exact definition.

# %%
TARGET_NAME = ""        # e.g. "extended micro-price MSE skill vs mid at 500 ms, test set"
TARGET_VALUE = None     # the original notebook's number
TARGET_CELL = ""        # where it is in the original
TOLERANCE = None        # how close counts as a match, and why

# %% [markdown]
# ## 2. Data: the smallest slice that defines the number

# %%


# %% [markdown]
# ## 3. My implementation

# %%


# %% [markdown]
# ## 4. Compare

# %%
MY_VALUE = None
print(f"{TARGET_NAME}: original {TARGET_VALUE}, mine {MY_VALUE}")

# %% [markdown]
# ## 5. (Optional) known-answer check for Part B4
# Simulate data where you know the true answer and check that your implementation recovers it.

# %%


# %% [markdown]
# ## 6. Notes
# What matched, what didn't, and what you learned about the method. Carry the gaps back to the review file.
