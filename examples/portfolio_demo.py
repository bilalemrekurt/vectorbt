"""Self-contained Portfolio demo: no internet or API keys needed.

Run:
    pip install vectorbt
    python examples/portfolio_demo.py

Generates synthetic prices, backtests a moving-average crossover over a parameter
grid, and writes an interactive HTML dashboard (portfolio_demo.html) that opens in the browser.
"""
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.io as pio

import vectorbt as vbt

# 1) Synthetic market data (geometric random walks, 3 assets)
np.random.seed(42)
index = pd.date_range("2020-01-01", periods=750, freq="D")
rets = np.random.normal(0.0004, 0.02, size=(len(index), 3))
close = pd.DataFrame(100 * np.exp(rets.cumsum(axis=0)), index=index, columns=["AAA", "BBB", "CCC"])

# 2) Parameter sweep: dual moving average crossover, fast x slow windows
windows = np.arange(5, 55, 5)
fast_ma, slow_ma = vbt.MA.run_combs(close["AAA"], windows, short_names=["fast", "slow"])
entries = fast_ma.ma_crossed_above(slow_ma)
exits = fast_ma.ma_crossed_below(slow_ma)
pf_grid = vbt.Portfolio.from_signals(close["AAA"], entries, exits, fees=0.001, freq="D")

heatmap_fig = pf_grid.total_return().vbt.heatmap(
    x_level="fast_window", y_level="slow_window", symmetric=True,
    trace_kwargs=dict(colorbar=dict(title="Total return")),
)
heatmap_fig.update_layout(title="Parameter sweep: total return (AAA)")

# 3) Best parameter pair applied to all three assets
best = pf_grid.total_return().idxmax()
print("Best (fast, slow) windows:", best)
fast = vbt.MA.run(close, best[0])
slow = vbt.MA.run(close, best[1])
entries_all = fast.ma_crossed_above(slow)
exits_all = fast.ma_crossed_below(slow)
entries_all.columns = exits_all.columns = close.columns  # drop the window level
pf = vbt.Portfolio.from_signals(
    close, entries_all, exits_all,
    fees=0.001, init_cash=10_000, freq="D", cash_sharing=False,
)
hold = vbt.Portfolio.from_holding(close, init_cash=10_000, freq="D")

print("\n=== Strategy stats (AAA) ===")
print(pf["AAA"].stats())
print("\nTotal return strategy vs buy & hold:")
print(pd.DataFrame({"strategy": pf.total_return(), "hold": hold.total_return()}))

# 4) Figures
col = "AAA"
figs = [
    heatmap_fig,
    pf[col].plot(),                      # orders, trade PnL, cumulative returns, etc.
    pf[col].plot_trades(),
    pf[col].plot_drawdowns(),
    pf[col].plot_underwater(),
    pf.value().vbt.plot(title="Portfolio value per asset (strategy)"),
    hold.value().vbt.plot(title="Portfolio value per asset (buy & hold)"),
]

# 5) Single HTML dashboard
stats_html = pf.stats(agg_func=None).to_html()
parts = [
    "<html><head><meta charset='utf-8'><title>vectorbt Portfolio demo</title></head>"
    "<body style='font-family:sans-serif;max-width:1100px;margin:auto'>"
    "<h1>vectorbt Portfolio demo (synthetic data)</h1>"
    f"<p>Best window pair: <b>{best}</b></p><h2>Stats per asset</h2>{stats_html}"
]
for i, fig in enumerate(figs):
    parts.append(pio.to_html(fig, full_html=False, include_plotlyjs="cdn" if i == 0 else False))
parts.append("</body></html>")

out = Path(__file__).with_name("portfolio_demo.html")
out.write_text("\n".join(parts), encoding="utf-8")
print(f"\nDashboard written to {out}")
webbrowser.open(out.resolve().as_uri())
