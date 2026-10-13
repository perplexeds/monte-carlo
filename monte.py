"""
Monte Carlo Simulation Suite — Quantitative Finance
====================================================
Demonstrates: GBM asset pricing, options pricing (European & Asian),
VaR/CVaR estimation, portfolio optimization, and variance reduction
techniques (antithetic variates, control variates).

Usage:
    python monte_carlo_sim.py                          # default tickers
    python monte_carlo_sim.py AAPL MSFT GOOGL          # custom tickers
    python monte_carlo_sim.py TSLA NVDA AMD --period 5y
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm
from scipy.optimize import minimize
from dataclasses import dataclass
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)


# ─────────────────────────────────────────────────────────────
# 0. DATA FETCHING — yfinance with fallback
# ─────────────────────────────────────────────────────────────

FALLBACK_DATA = {
    "AAPL": {"S0": 175.0, "mu": 0.14, "sigma": 0.22},
    "MSFT": {"S0": 330.0, "mu": 0.13, "sigma": 0.20},
    "SPY":  {"S0": 450.0, "mu": 0.10, "sigma": 0.15},
    "GOOGL":{"S0": 140.0, "mu": 0.12, "sigma": 0.21},
    "TSLA": {"S0": 250.0, "mu": 0.18, "sigma": 0.55},
    "NVDA": {"S0": 480.0, "mu": 0.30, "sigma": 0.50},
    "AMD":  {"S0": 120.0, "mu": 0.20, "sigma": 0.45},
    "AMZN": {"S0": 180.0, "mu": 0.15, "sigma": 0.28},
}

DEFAULT_FALLBACK_CORR = 0.65   # assumed pairwise correlation when using fallback


def fetch_market_data(tickers: list[str], period: str = "2y") -> dict:
    """
    Pull historical price data from Yahoo Finance via yfinance.
    Falls back to hardcoded estimates if download fails.

    Returns a dict with keys: S0, mu, sigma, corr, source
    """
    try:
        import yfinance as yf
        print(f"  Fetching {period} of data for: {', '.join(tickers)} ...")
        raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)

        # yfinance returns MultiIndex columns when >1 ticker
        if len(tickers) == 1:
            prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
        else:
            prices = raw["Close"][tickers]

        prices = prices.dropna()

        if prices.empty or len(prices) < 30:
            raise ValueError("Insufficient data returned from Yahoo Finance.")

        log_rets = np.log(prices / prices.shift(1)).dropna()

        S0    = prices.iloc[-1].values.astype(float)
        mu    = (log_rets.mean() * 252).values.astype(float)
        sigma = (log_rets.std() * np.sqrt(252)).values.astype(float)
        corr  = log_rets.corr().values.astype(float)

        # Ensure correlation matrix is positive semi-definite (numerical safety)
        eigvals = np.linalg.eigvalsh(corr)
        if eigvals.min() < 0:
            corr += (-eigvals.min() + 1e-8) * np.eye(len(tickers))
            corr /= corr.diagonal() ** 0.5
            corr /= corr.diagonal()[:, None] ** 0.5

        print(f"  ✓ Live data loaded  ({len(prices)} trading days)")
        for i, t in enumerate(tickers):
            print(f"    {t:6s}  price=${S0[i]:.2f}  "
                  f"mu={mu[i]*100:.1f}%  sigma={sigma[i]*100:.1f}%")

        return {"S0": S0, "mu": mu, "sigma": sigma, "corr": corr,
                "source": "yfinance", "prices": prices, "log_rets": log_rets}

    except Exception as e:
        print(f"  ⚠  Could not fetch live data ({e})")
        print("     Falling back to hardcoded estimates …")

        S0, mu, sigma = [], [], []
        for t in tickers:
            fb = FALLBACK_DATA.get(t.upper(), {"S0": 100.0, "mu": 0.10, "sigma": 0.25})
            S0.append(fb["S0"])
            mu.append(fb["mu"])
            sigma.append(fb["sigma"])
            print(f"    {t:6s}  price=${fb['S0']:.2f}  "
                  f"mu={fb['mu']*100:.1f}%  sigma={fb['sigma']*100:.1f}%")

        n = len(tickers)
        corr = DEFAULT_FALLBACK_CORR * np.ones((n, n))
        np.fill_diagonal(corr, 1.0)

        return {"S0": np.array(S0), "mu": np.array(mu),
                "sigma": np.array(sigma), "corr": corr,
                "source": "fallback", "prices": None, "log_rets": None}


# ─────────────────────────────────────────────────────────────
# 1. GEOMETRIC BROWNIAN MOTION — Multi-Asset Path Simulator
# ─────────────────────────────────────────────────────────────

@dataclass
class MarketParams:
    S0: np.ndarray          # Initial prices
    mu: np.ndarray          # Drift vector (annual)
    sigma: np.ndarray       # Volatility vector (annual)
    corr: np.ndarray        # Correlation matrix
    T: float                # Time horizon (years)
    n_steps: int            # Time steps
    n_sims: int             # Number of simulations

    @property
    def dt(self) -> float:
        return self.T / self.n_steps

    @property
    def cov(self) -> np.ndarray:
        """Covariance matrix from correlation and vols."""
        vol_outer = np.outer(self.sigma, self.sigma)
        return self.corr * vol_outer


def simulate_gbm(params: MarketParams) -> np.ndarray:
    """
    Simulate correlated GBM paths via Cholesky decomposition.

    Returns
    -------
    paths : ndarray, shape (n_assets, n_sims, n_steps + 1)
    """
    n = len(params.S0)
    L = np.linalg.cholesky(params.corr)

    paths = np.zeros((n, params.n_sims, params.n_steps + 1))
    paths[:, :, 0] = params.S0[:, None]

    dt = params.dt
    for t in range(1, params.n_steps + 1):
        Z = np.random.standard_normal((n, params.n_sims))
        Z_corr = L @ Z
        drift     = (params.mu - 0.5 * params.sigma ** 2)[:, None] * dt
        diffusion = params.sigma[:, None] * np.sqrt(dt) * Z_corr
        paths[:, :, t] = paths[:, :, t - 1] * np.exp(drift + diffusion)

    return paths


# ─────────────────────────────────────────────────────────────
# 2. OPTIONS PRICING — European & Asian with Variance Reduction
# ─────────────────────────────────────────────────────────────

@dataclass
class OptionParams:
    S0: float
    K: float
    r: float
    sigma: float
    T: float
    n_steps: int = 252
    n_sims: int = 100_000


def black_scholes(opt: OptionParams, call: bool = True) -> float:
    d1 = (np.log(opt.S0 / opt.K) + (opt.r + 0.5 * opt.sigma ** 2) * opt.T) / (
        opt.sigma * np.sqrt(opt.T)
    )
    d2 = d1 - opt.sigma * np.sqrt(opt.T)
    if call:
        return opt.S0 * norm.cdf(d1) - opt.K * np.exp(-opt.r * opt.T) * norm.cdf(d2)
    return opt.K * np.exp(-opt.r * opt.T) * norm.cdf(-d2) - opt.S0 * norm.cdf(-d1)


def price_european_mc(opt: OptionParams, call: bool = True,
                      antithetic: bool = True) -> dict:
    Z = np.random.standard_normal(opt.n_sims // 2)
    if antithetic:
        Z = np.concatenate([Z, -Z])
    ST = opt.S0 * np.exp(
        (opt.r - 0.5 * opt.sigma ** 2) * opt.T
        + opt.sigma * np.sqrt(opt.T) * Z
    )
    payoff = np.maximum(ST - opt.K, 0) if call else np.maximum(opt.K - ST, 0)
    price  = np.exp(-opt.r * opt.T) * payoff
    return {
        "price": price.mean(),
        "std_error": price.std() / np.sqrt(opt.n_sims),
        "ci_95": (
            price.mean() - 1.96 * price.std() / np.sqrt(opt.n_sims),
            price.mean() + 1.96 * price.std() / np.sqrt(opt.n_sims),
        ),
    }


def price_asian_mc(opt: OptionParams, call: bool = True) -> dict:
    dt  = opt.T / opt.n_steps
    Z   = np.random.standard_normal((opt.n_sims, opt.n_steps))
    log_S = np.log(opt.S0) + np.cumsum(
        (opt.r - 0.5 * opt.sigma ** 2) * dt
        + opt.sigma * np.sqrt(dt) * Z,
        axis=1,
    )
    S         = np.exp(log_S)
    arith_avg = S.mean(axis=1)
    geo_avg   = np.exp(log_S.mean(axis=1))

    arith_payoff = np.exp(-opt.r * opt.T) * np.maximum(
        arith_avg - opt.K if call else opt.K - arith_avg, 0
    )
    geo_payoff = np.exp(-opt.r * opt.T) * np.maximum(
        geo_avg - opt.K if call else opt.K - geo_avg, 0
    )
    geo_mean = geo_payoff.mean()
    beta     = np.cov(arith_payoff, geo_payoff)[0, 1] / np.var(geo_payoff)
    cv_price = arith_payoff - beta * (geo_payoff - geo_mean)

    return {
        "price": cv_price.mean(),
        "std_error": cv_price.std() / np.sqrt(opt.n_sims),
        "naive_price": arith_payoff.mean(),
        "variance_reduction_pct": (1 - cv_price.var() / arith_payoff.var()) * 100,
    }


# ─────────────────────────────────────────────────────────────
# 3. VALUE-AT-RISK & CVaR
# ─────────────────────────────────────────────────────────────

def compute_risk_metrics(returns: np.ndarray, confidence: float = 0.95,
                         horizon: int = 1,
                         portfolio_value: float = 1_000_000) -> dict:
    scaled    = returns * np.sqrt(horizon)
    hist_var  = -np.percentile(scaled, (1 - confidence) * 100) * portfolio_value
    cutoff    = np.percentile(scaled, (1 - confidence) * 100)
    hist_cvar = -scaled[scaled <= cutoff].mean() * portfolio_value
    mu_r, sigma_r = scaled.mean(), scaled.std()
    param_var = -(mu_r + norm.ppf(1 - confidence) * sigma_r) * portfolio_value
    return {"hist_var": hist_var, "hist_cvar": hist_cvar, "param_var": param_var,
            "confidence": confidence, "horizon_days": horizon,
            "portfolio_value": portfolio_value}


# ─────────────────────────────────────────────────────────────
# 4. PORTFOLIO OPTIMIZATION
# ─────────────────────────────────────────────────────────────

def optimize_portfolio(mu: np.ndarray, cov: np.ndarray,
                       n_frontier: int = 200, n_random: int = 5_000) -> dict:
    n = len(mu)

    def portfolio_stats(w):
        ret    = w @ mu
        vol    = np.sqrt(w @ cov @ w)
        sharpe = ret / vol
        return ret, vol, sharpe

    rand_w    = np.random.dirichlet(np.ones(n), n_random)
    rand_stats = np.array([portfolio_stats(w) for w in rand_w])

    target_rets = np.linspace(mu.min(), mu.max(), n_frontier)
    frontier    = []
    for target in target_rets:
        result = minimize(
            lambda w: w @ cov @ w,
            x0=np.ones(n) / n,
            method="SLSQP",
            bounds=[(0, 1)] * n,
            constraints=[  # type: ignore
                {"type": "eq", "fun": lambda w: w.sum() - 1},
                {"type": "eq", "fun": lambda w: w @ mu - target},
            ],
        )
        if result.success:
            r, v, s = portfolio_stats(result.x)
            frontier.append({"return": r, "vol": v, "sharpe": s,
                             "weights": result.x})

    result_ms = minimize(
        lambda w: -(w @ mu) / np.sqrt(w @ cov @ w),
        x0=np.ones(n) / n,
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],  # type: ignore
    )
    ms_w = result_ms.x
    ms_ret, ms_vol, ms_sharpe = portfolio_stats(ms_w)

    return {"frontier": frontier, "random_portfolios": rand_stats,
            "max_sharpe": {"weights": ms_w, "return": ms_ret,
                           "vol": ms_vol, "sharpe": ms_sharpe}}


# ─────────────────────────────────────────────────────────────
# 5. VISUALIZATION
# ─────────────────────────────────────────────────────────────

def plot_results(paths, returns, risk, port, option_results,
                 asset_names, data_source, prices_df) -> None:

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#0d1117")
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    ACCENT = "#58a6ff"
    ACCENT2 = "#3fb950"
    ACCENT3 = "#f78166"
    TEXT = "#e6edf3"
    GRID = "#21262d"
    BG   = "#161b22"
    COLORS = [ACCENT, ACCENT2, ACCENT3, "#d2a8ff", "#ffa657", "#79c0ff"]

    def styled_ax(ax, title):
        ax.set_facecolor(BG)
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.set_title(title, color=TEXT, fontsize=9, fontweight="bold", pad=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.yaxis.label.set_color(TEXT)
        ax.xaxis.label.set_color(TEXT)
        ax.grid(color=GRID, linewidth=0.5, alpha=0.7)

    # ── 0. Historical prices (if live data available) ──────────
    ax0 = fig.add_subplot(gs[0, :2])
    if prices_df is not None:
        styled_ax(ax0, f"Historical Prices — {', '.join(asset_names)}  "
                       f"({'Live data' if data_source == 'yfinance' else 'Fallback'})")
        norm_prices = prices_df / prices_df.iloc[0] * 100
        for i, (col, color) in enumerate(zip(norm_prices.columns,
                                              COLORS[:len(asset_names)])):
            ax0.plot(norm_prices.index, norm_prices[col],
                     color=color, linewidth=1.4, label=col)
        ax0.set_ylabel("Normalised Price (base=100)")
        ax0.legend(fontsize=8, facecolor=BG, labelcolor=TEXT, framealpha=0.8)
        ax0.tick_params(axis="x", rotation=20)
    else:
        # Fall back to GBM paths panel
        styled_ax(ax0, "Simulated GBM Paths (fallback data — no live prices)")
        t_axis = np.linspace(0, 1, paths.shape[2])
        for i, (name, color) in enumerate(zip(asset_names, COLORS)):
            ax0.plot(t_axis, paths[i, :200, :].T, alpha=0.04,
                     color=color, linewidth=0.5)
            ax0.plot(t_axis, paths[i].mean(axis=0),
                     color=color, linewidth=2, label=f"{name} (mean)")
        ax0.set_xlabel("Time (years)")
        ax0.set_ylabel("Price ($)")
        ax0.legend(fontsize=8, facecolor=BG, labelcolor=TEXT, framealpha=0.8)

    # ── 1. Simulated terminal distribution ────────────────────
    ax1 = fig.add_subplot(gs[0, 2])
    styled_ax(ax1, "1-Year Terminal Price Distribution")
    for i, (name, color) in enumerate(zip(asset_names, COLORS)):
        pct_lo = np.percentile(paths[i, :, -1], 5)
        pct_hi = np.percentile(paths[i, :, -1], 95)
        ax1.hist(paths[i, :, -1], bins=60, alpha=0.5, color=color,
                 label=f"{name} [5%={pct_lo:.0f}, 95%={pct_hi:.0f}]",
                 edgecolor="none", density=True)
    ax1.set_xlabel("Terminal Price ($)")
    ax1.set_ylabel("Density")
    ax1.legend(fontsize=7, facecolor=BG, labelcolor=TEXT, framealpha=0.8)

    # ── 2. VaR / CVaR ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    styled_ax(ax2, f"P&L Distribution — {int(risk['confidence']*100)}% VaR/CVaR")
    pnl = returns * risk["portfolio_value"]
    ax2.hist(pnl, bins=80, color=ACCENT, alpha=0.6, density=True, edgecolor="none")
    ax2.axvline(-risk["hist_var"],  color=ACCENT3, linewidth=1.8,
                label=f"VaR  ${risk['hist_var']:,.0f}")
    ax2.axvline(-risk["hist_cvar"], color="#f0883e", linewidth=1.8, linestyle="--",
                label=f"CVaR ${risk['hist_cvar']:,.0f}")
    ax2.set_xlabel("Daily P&L ($)")
    ax2.set_ylabel("Density")
    ax2.legend(fontsize=7.5, facecolor=BG, labelcolor=TEXT, framealpha=0.8)

    # ── 3. Efficient Frontier ──────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1:])
    styled_ax(ax3, "Efficient Frontier & Random Portfolios")
    rs = port["random_portfolios"]
    sc = ax3.scatter(rs[:, 1] * 100, rs[:, 0] * 100, c=rs[:, 2],
                     cmap="plasma", alpha=0.3, s=4, linewidths=0)
    cbar = plt.colorbar(sc, ax=ax3)
    cbar.set_label("Sharpe Ratio", color=TEXT, fontsize=8)
    cbar.ax.yaxis.set_tick_params(color=TEXT)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT, fontsize=7)

    frontier_vols = [p["vol"] * 100 for p in port["frontier"]]
    frontier_rets = [p["return"] * 100 for p in port["frontier"]]
    ax3.plot(frontier_vols, frontier_rets, color=ACCENT, linewidth=2.5,
             label="Efficient Frontier")
    ms = port["max_sharpe"]
    ax3.scatter(ms["vol"] * 100, ms["return"] * 100, color=ACCENT2,
                s=120, zorder=5, marker="*",
                label=f"Max Sharpe ({ms['sharpe']:.2f})")
    ax3.set_xlabel("Annualised Volatility (%)")
    ax3.set_ylabel("Annualised Return (%)")
    ax3.legend(fontsize=8, facecolor=BG, labelcolor=TEXT, framealpha=0.8)

    # ── 4. Options table ──────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, :2])
    styled_ax(ax4, "Options Pricing Results")
    ax4.axis("off")
    headers = ["Contract", "MC Price", "BS / Naive", "Std Error",
               "95% CI / VR%", "Method"]
    rows = []
    for name, res in option_results.items():
        if "ci_95" in res:
            rows.append([name, f"${res['price']:.4f}",
                         f"${res.get('bs_price', 0):.4f}",
                         f"${res['std_error']:.4f}",
                         f"[{res['ci_95'][0]:.3f}, {res['ci_95'][1]:.3f}]",
                         "Antithetic variates"])
        else:
            rows.append([name, f"${res['price']:.4f}",
                         f"${res.get('naive_price', 0):.4f}",
                         f"${res['std_error']:.4f}",
                         f"{res['variance_reduction_pct']:.1f}%",
                         "Control variate"])
    tbl = ax4.table(cellText=rows, colLabels=headers,
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("#1c2128" if r == 0 else BG)
        cell.set_text_props(color=ACCENT if r == 0 else TEXT)
        cell.set_edgecolor(GRID)
    tbl.scale(1, 1.6)

    # ── 5. MC Convergence ─────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 2])
    styled_ax(ax5, "MC Convergence — ATM European Call")
    opt_ref = list(option_results.values())[0]
    bs_ref  = opt_ref.get("bs_price", None)
    n_range = np.logspace(2, 5, 60).astype(int)
    mc_prices = []
    for n in n_range:
        Z  = np.random.standard_normal(n)
        p  = OptionParams(S0=100, K=100, r=0.05, sigma=0.2, T=1.0, n_sims=n)
        ST = p.S0 * np.exp((p.r - 0.5*p.sigma**2)*p.T + p.sigma*np.sqrt(p.T)*Z)
        mc_prices.append(np.exp(-p.r*p.T) * np.maximum(ST - p.K, 0).mean())
    ax5.semilogx(n_range, mc_prices, color=ACCENT, linewidth=1.2, label="MC Estimate")
    if bs_ref:
        ax5.axhline(bs_ref, color=ACCENT2, linewidth=1.5,
                    linestyle="--", label=f"BS = ${bs_ref:.4f}")
    ax5.set_xlabel("Simulations")
    ax5.set_ylabel("Option Price ($)")
    ax5.legend(fontsize=8, facecolor=BG, labelcolor=TEXT, framealpha=0.8)

    source_label = "Live Data via yfinance" if data_source == "yfinance" else "Hardcoded Fallback Data"
    fig.suptitle(
        f"Monte Carlo Simulation Suite — {', '.join(asset_names)}  [{source_label}]",
        color=TEXT, fontsize=12, fontweight="bold", y=0.98,
    )
    plt.savefig("mc_results.png", dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print("  → Saved: mc_results.png")


# ─────────────────────────────────────────────────────────────
# 6. CLI + MAIN
# ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Monte Carlo Simulation Suite — Quantitative Finance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python monte_carlo_sim.py
  python monte_carlo_sim.py AAPL MSFT GOOGL
  python monte_carlo_sim.py TSLA NVDA AMD --period 5y --sims 20000
        """,
    )
    parser.add_argument(
        "tickers", nargs="*", default=["AAPL", "MSFT", "SPY"],
        help="Stock tickers to analyse (default: AAPL MSFT SPY)",
    )
    parser.add_argument(
        "--period", default="2y",
        help="History window for yfinance: 1y 2y 5y etc. (default: 2y)",
    )
    parser.add_argument(
        "--sims", type=int, default=10_000,
        help="Number of Monte Carlo simulations (default: 10000)",
    )
    parser.add_argument(
        "--horizon", type=float, default=1.0,
        help="Forecast horizon in years (default: 1.0)",
    )
    parser.add_argument(
        "--portfolio-value", type=float, default=1_000_000,
        help="Portfolio value for VaR/CVaR in $ (default: 1000000)",
    )
    return parser.parse_args()


def main():
    args   = parse_args()
    tickers = [t.upper() for t in args.tickers]

    print("=" * 60)
    print("  Monte Carlo Simulation Suite — Quantitative Finance")
    print("=" * 60)
    print(f"  Tickers  : {', '.join(tickers)}")
    print(f"  Period   : {args.period}  |  Sims: {args.sims:,}  "
          f"|  Horizon: {args.horizon}y")
    print(f"  Portfolio: ${args.portfolio_value:,.0f}")
    print("=" * 60)

    # ── Fetch data ─────────────────────────────────────────────
    print("\n[0/5] Loading market data …")
    mkt_data = fetch_market_data(tickers, period=args.period)

    market = MarketParams(
        S0      = mkt_data["S0"],
        mu      = mkt_data["mu"],
        sigma   = mkt_data["sigma"],
        corr    = mkt_data["corr"],
        T       = args.horizon,
        n_steps = int(252 * args.horizon),
        n_sims  = args.sims,
    )

    # ── GBM simulation ─────────────────────────────────────────
    print("\n[1/5] Simulating correlated GBM paths …")
    paths = simulate_gbm(market)
    print(f"      Shape: {paths.shape}  (assets × sims × steps)")

    # ── Options (priced on first ticker) ──────────────────────
    print(f"\n[2/5] Pricing options on {tickers[0]} …")
    S0_opt  = float(market.S0[0])
    sig_opt = float(market.sigma[0])
    opt = OptionParams(S0=S0_opt, K=S0_opt,        # ATM option
                       r=0.05, sigma=sig_opt, T=1.0,
                       n_steps=252, n_sims=100_000)
    bs_call = black_scholes(opt, call=True)
    bs_put  = black_scholes(opt, call=False)
    eu_call = price_european_mc(opt, call=True,  antithetic=True)
    eu_put  = price_european_mc(opt, call=False, antithetic=True)
    asian_c = price_asian_mc(opt, call=True)
    eu_call["bs_price"] = bs_call
    eu_put["bs_price"]  = bs_put
    option_results = {
        f"{tickers[0]} EU Call (MC)": eu_call,
        f"{tickers[0]} EU Put  (MC)": eu_put,
        f"{tickers[0]} Asian Call  ": asian_c,
    }
    print(f"      BS Call={bs_call:.4f}  MC={eu_call['price']:.4f}  "
          f"err={abs(bs_call-eu_call['price']):.4f}")
    print(f"      BS Put ={bs_put:.4f}  MC={eu_put['price']:.4f}  "
          f"err={abs(bs_put-eu_put['price']):.4f}")
    print(f"      Asian Call={asian_c['price']:.4f}  "
          f"VR={asian_c['variance_reduction_pct']:.1f}%")

    # ── VaR / CVaR ─────────────────────────────────────────────
    print("\n[3/5] Computing VaR / CVaR …")
    w_eq       = np.ones(len(tickers)) / len(tickers)
    port_rets  = paths[:, :, 1:] / paths[:, :, :-1] - 1
    port_daily = (port_rets * w_eq[:, None, None]).sum(axis=0).flatten()
    risk = compute_risk_metrics(port_daily, confidence=0.95, horizon=1,
                                portfolio_value=args.portfolio_value)
    print(f"      95% 1-day Hist VaR  = ${risk['hist_var']:>12,.0f}")
    print(f"      95% 1-day Hist CVaR = ${risk['hist_cvar']:>12,.0f}")
    print(f"      95% 1-day Para VaR  = ${risk['param_var']:>12,.0f}")

    # ── Portfolio optimisation ─────────────────────────────────
    print("\n[4/5] Running Markowitz optimisation …")
    port = optimize_portfolio(market.mu, market.cov,
                              n_frontier=200, n_random=5_000)
    ms = port["max_sharpe"]
    print("      Max Sharpe portfolio:")
    for name, w in zip(tickers, ms["weights"]):
        print(f"        {name}: {w*100:.1f}%")
    print(f"      Return={ms['return']*100:.1f}%  "
          f"Vol={ms['vol']*100:.1f}%  Sharpe={ms['sharpe']:.3f}")

    # ── Plot ───────────────────────────────────────────────────
    print("\n[5/5] Generating dashboard …")
    plot_results(paths, port_daily, risk, port, option_results,
                 tickers, mkt_data["source"], mkt_data["prices"])

    print("\n" + "=" * 60)
    print("  Done.  Output saved to: mc_results.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
