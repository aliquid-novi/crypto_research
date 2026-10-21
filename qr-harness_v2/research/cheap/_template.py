# %% [markdown]
# # {{ID}}: {{TITLE}}
#
# ## Pre-implementation (Mark, before any code runs on real data)
# **Hypothesis:** _(Mark)_
#
# **Why I think it works:** _(Mark)_
#
# **Assumptions:** _(Mark)_
#
# **Test statistic / method, and why:** _(Mark)_
#
# ## Post-implementation (Mark, after reading the results)
# **What actually happened:** _(Mark)_
#
# **What changed vs my expectation:** _(Mark)_
#
# **Why it worked / didn't:** _(Mark)_
#
# **Was the test the right one? Why:** _(Mark)_
#
# **What changes to the data or initial conditions could have changed the result:** _(Mark)_

# %%
# ---- gate: refuses to run until Mark's pre-implementation fields are filled ----
import sys
from pathlib import Path

ROOT = Path.cwd()
while not (ROOT / "qrlib").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qrlib import audit, cheap, data, fairvalue, fills, leakage, pit, stats

CID = "{{ID}}"
PRE = cheap.require(CID)

# %% [markdown]
# ## Implementation (agent)
# Request (Mark, verbatim):
# > {{REQUEST}}

# %%
# ---- KNOBS: everything tunable; `# ASSUMPTION:` on every choice Mark didn't specify ----

# %%
# ---- data ----

# %%
# ---- method ----

# %%
# ---- leakage check (anything fitted, or features built from data) / or say why it isn't needed ----

# %%
# ---- evaluation: n, CIs; log every evaluated variant with audit.log_run(CID, ...) ----

# %%
print(f"n_trials for {CID}: {audit.n_trials(CID)}")

# %% [markdown]
# ## Results (agent)
# _Weak points first. Numbers with n, CIs and n_trials. No verdict on whether the hypothesis held: that's Mark's post-implementation._
#
# **Assumptions made (every `# ASSUMPTION:`):**
#
# **Questions for Mark:**
