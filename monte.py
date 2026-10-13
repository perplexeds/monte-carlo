"""
Monte Carlo Simulation Suite — Quantitative Finance
====================================================
Demonstrates: GBM asset pricing, options pricing (European & Asian),
VaR/CVaR estimation, portfolio optimization, and variance reduction
techniques (antithetic variates, control variates).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Optional
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)


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
    L = np.linalg.cholesky(params.corr)          # Cholesky factor

    paths = np.zeros((n, params.n_sims, params.n_steps + 1))
    paths[:, :, 0] = params.S0[:, None]

    dt = params.dt
    for t in range(1, params.n_steps + 1):
        Z = np.random.standard_normal((n, params.n_sims))
        Z_corr = L @ Z                            # Introduce correlation
        drift = (params.mu - 0.5 * params.sigma ** 2)[:, None] * dt
        diffusion = params.sigma[:, None] * np.sqrt(dt) * Z_corr
        paths[:, :, t] = paths[:, :, t - 1] * np.exp(drift + diffusion)

    return paths


# ─────────────────────────────────────────────────────────────
# 2. OPTIONS PRICING — European & Asian with Variance Reduction
# ─────────────────────────────────────────────────────────────

@dataclass
class OptionParams:
    S0: float
    K: float                # Strike
    r: float                # Risk-free rate
    sigma: float            # Volatility
    T: float                # Maturity (years)
    n_steps: int = 252
    n_sims: int = 100_000


def black_scholes(opt: OptionParams, call: bool = True) -> float:
    """Closed-form Black-Scholes price for benchmarking."""
    d1 = (np.log(opt.S0 / opt.K) + (opt.r + 0.5 * opt.sigma ** 2) * opt.T) / (
        opt.sigma * np.sqrt(opt.T)
    )
    d2 = d1 - opt.sigma * np.sqrt(opt.T)
    if call:
        return opt.S0 * norm.cdf(d1) - opt.K * np.exp(-opt.r * opt.T) * norm.cdf(d2)
    return opt.K * np.exp(-opt.r * opt.T) * norm.cdf(-d2) - opt.S0 * norm.cdf(-d1)


def price_european_mc(
    opt: OptionParams,
    call: bool = True,
    antithetic: bool = True,
) -> dict:
    """
    European option pricing via MC with antithetic variates.
    Variance reduction: antithetic variates halve the standard error.
    """
    Z = np.random.standard_normal(opt.n_sims // 2)
    if antithetic:
        Z = np.concatenate([Z, -Z])

    ST = opt.S0 * np.exp(
        (opt.r - 0.5 * opt.sigma ** 2) * opt.T
        + opt.sigma * np.sqrt(opt.T) * Z
    )
    payoff = np.maximum(ST - opt.K, 0) if call else np.maximum(opt.K - ST, 0)
    price = np.exp(-opt.r * opt.T) * payoff

    return {
        "price": price.mean(),
        "std_error": price.std() / np.sqrt(opt.n_sims),
        "ci_95": (
            price.mean() - 1.96 * price.std() / np.sqrt(opt.n_sims),
            price.mean() + 1.96 * price.std() / np.sqrt(opt.n_sims),
        ),
    }


def price_asian_mc(opt: OptionParams, call: bool = True) -> dict:
    """
    Arithmetic-average Asian option pricing (path-dependent).
    Uses geometric Asian as control variate for variance reduction.
    """
    dt = opt.T / opt.n_steps
    Z = np.random.standard_normal((opt.n_sims, opt.n_steps))

    # Simulate paths
    log_S = np.log(opt.S0) + np.cumsum(
        (opt.r - 0.5 * opt.sigma ** 2) * dt
        + opt.sigma * np.sqrt(dt) * Z,
        axis=1,
    )
    S = np.exp(log_S)                             # (n_sims, n_steps)

    arith_avg = S.mean(axis=1)
    geo_avg   = np.exp(log_S.mean(axis=1))        # geometric average

    # Arithmetic payoff (target)
    arith_payoff = np.exp(-opt.r * opt.T) * np.maximum(
        arith_avg - opt.K if call else opt.K - arith_avg, 0
    )

    # Geometric payoff (control — has closed form)
    geo_payoff = np.exp(-opt.r * opt.T) * np.maximum(
        geo_avg - opt.K if call else opt.K - geo_avg, 0
    )

    # Control variate correction (OLS beta)
    geo_mean = geo_payoff.mean()
    beta = np.cov(arith_payoff, geo_payoff)[0, 1] / np.var(geo_payoff)
    cv_price = arith_payoff - beta * (geo_payoff - geo_mean)

    return {
        "price": cv_price.mean(),
        "std_error": cv_price.std() / np.sqrt(opt.n_sims),
        "naive_price": arith_payoff.mean(),
        "variance_reduction_pct": (
            1 - cv_price.var() / arith_payoff.var()
        ) * 100,
    }


# ─────────────────────────────────────────────────────────────
# 3. VALUE-AT-RISK & CONDITIONAL VALUE-AT-RISK (CVaR)
# ─────────────────────────────────────────────────────────────

def compute_risk_metrics(
    returns: np.ndarray,
    confidence: float = 0.95,
    horizon: int = 1,
    portfolio_value: float = 1_000_000,
) -> dict:
    """
    Historical and parametric VaR / CVaR.

    Parameters
    ----------
    returns : 1-D array of daily log-returns (simulated or historical)
    """
    # Scale to horizon
    scaled = returns * np.sqrt(horizon)

    # Historical VaR (non-parametric)
    hist_var = -np.percentile(scaled, (1 - confidence) * 100) * portfolio_value

    # Historical CVaR (Expected Shortfall)
    cutoff = np.percentile(scaled, (1 - confidence) * 100)
    hist_cvar = -scaled[scaled <= cutoff].mean() * portfolio_value

    # Parametric VaR (Gaussian assumption)
    mu_r, sigma_r = scaled.mean(), scaled.std()
    param_var = -(mu_r + norm.ppf(1 - confidence) * sigma_r) * portfolio_value

    return {
        "hist_var": hist_var,
        "hist_cvar": hist_cvar,
        "param_var": param_var,
        "confidence": confidence,
        "horizon_days": horizon,
        "portfolio_value": portfolio_value,
    }


# ─────────────────────────────────────────────────────────────
# 4. PORTFOLIO OPTIMIZATION — Mean-Variance (Markowitz)
# ─────────────────────────────────────────────────────────────

def optimize_portfolio(
    mu: np.ndarray,
    cov: np.ndarray,
    n_frontier: int = 200,
    n_random: int = 5_000,
) -> dict:
    """
    Compute efficient frontier via sequential quadratic programming.
    Also samples random portfolios for scatter comparison.
    """
    n = len(mu)

    def portfolio_stats(w):
        ret = w @ mu
        vol = np.sqrt(w @ cov @ w)
        sharpe = ret / vol
        return ret, vol, sharpe

    # Random portfolios
    rand_w = np.random.dirichlet(np.ones(n), n_random)
    rand_stats = np.array([portfolio_stats(w) for w in rand_w])

    # Efficient frontier
    target_rets = np.linspace(mu.min(), mu.max(), n_frontier)
    frontier = []

    for target in target_rets:
        result = minimize(
            lambda w: w @ cov @ w,                          # Minimize variance
            x0=np.ones(n) / n,
            method="SLSQP",
            bounds=[(0, 1)] * n,
            constraints=[
                {"type": "eq", "fun": lambda w: w.sum() - 1},
                {"type": "eq", "fun": lambda w: w @ mu - target},
            ],
        )
        if result.success:
            r, v, s = portfolio_stats(result.x)
            frontier.append({"return": r, "vol": v, "sharpe": s, "weights": result.x})

    # Max Sharpe portfolio
    result_ms = minimize(
        lambda w: -(w @ mu) / np.sqrt(w @ cov @ w),
        x0=np.ones(n) / n,
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
    )
    ms_w = result_ms.x
    ms_ret, ms_vol, ms_sharpe = portfolio_stats(ms_w)

    return {
        "frontier": frontier,
        "random_portfolios": rand_stats,
        "max_sharpe": {
            "weights": ms_w,
            "return": ms_ret,
            "vol": ms_vol,
            "sharpe": ms_sharpe,
        },
    }


# ─────────────────────────────────────────────────────────────
# 5. VISUALIZATION
# ─────────────────────────────────────────────────────────────

def plot_results(
    paths: np.ndarray,
    returns: np.ndarray,
    risk: dict,
    port: dict,
    option_results: dict,
    asset_names: list[str],
) -> None:

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#0d1117")
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    ACCENT   = "#58a6ff"
    ACCENT2  = "#3fb950"
    ACCENT3  = "#f78166"
    TEXT     = "#e6edf3"
    GRID     = "#21262d"
    BG       = "#161b22"

    def styled_ax(ax, title: str):
        ax.set_facecolor(BG)
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.set_title(title, color=TEXT, fontsize=9, fontweight="bold", pad=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.yaxis.label.set_color(TEXT)
        ax.xaxis.label.set_color(TEXT)
        ax.grid(color=GRID, linewidth=0.5, alpha=0.7)

    # ── 1. GBM Paths ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    styled_ax(ax1, "Simulated GBM Asset Paths (first 200 sims)")
    colors = [ACCENT, ACCENT2, ACCENT3]
    t_axis = np.linspace(0, 1, paths.shape[2])
    for i, (name, color) in enumerate(zip(asset_names, colors)):
        ax1.plot(t_axis, paths[i, :200, :].T, alpha=0.04, color=color, linewidth=0.5)
        ax1.plot(t_axis, paths[i, :, :].mean(axis=0), color=color,
                 linewidth=2, label=f"{name} (mean)")
    ax1.set_xlabel("Time (years)")
    ax1.set_ylabel("Price ($)")
    ax1.legend(fontsize=8, facecolor=BG, labelcolor=TEXT, framealpha=0.8)

    # ── 2. Terminal Distribution ───────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    styled_ax(ax2, "Terminal Price Distribution")
    for i, (name, color) in enumerate(zip(asset_names, colors)):
        ax2.hist(paths[i, :, -1], bins=60, alpha=0.55, color=color,
                 label=name, edgecolor="none", density=True)
    ax2.set_xlabel("Terminal Price ($)")
    ax2.set_ylabel("Density")
    ax2.legend(fontsize=8, facecolor=BG, labelcolor=TEXT, framealpha=0.8)

    # ── 3. VaR / CVaR Return Distribution ─────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    styled_ax(ax3, f"P&L Distribution — {int(risk['confidence']*100)}% VaR/CVaR")
    pnl = returns * risk["portfolio_value"]
    ax3.hist(pnl, bins=80, color=ACCENT, alpha=0.6, density=True, edgecolor="none")
    ax3.axvline(-risk["hist_var"], color=ACCENT3, linewidth=1.8,
                label=f"VaR  ${risk['hist_var']:,.0f}")
    ax3.axvline(-risk["hist_cvar"], color="#f0883e", linewidth=1.8, linestyle="--",
                label=f"CVaR ${risk['hist_cvar']:,.0f}")
    ax3.set_xlabel("Daily P&L ($)")
    ax3.set_ylabel("Density")
    ax3.legend(fontsize=7.5, facecolor=BG, labelcolor=TEXT, framealpha=0.8)

    # ── 4. Efficient Frontier ──────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1:])
    styled_ax(ax4, "Efficient Frontier & Random Portfolios")
    rs = port["random_portfolios"]
    sc = ax4.scatter(rs[:, 1] * 100, rs[:, 0] * 100, c=rs[:, 2],
                     cmap="plasma", alpha=0.3, s=4, linewidths=0)
    cbar = plt.colorbar(sc, ax=ax4)
    cbar.set_label("Sharpe Ratio", color=TEXT, fontsize=8)
    cbar.ax.yaxis.set_tick_params(color=TEXT)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT, fontsize=7)

    frontier_vols = [p["vol"] * 100 for p in port["frontier"]]
    frontier_rets = [p["return"] * 100 for p in port["frontier"]]
    ax4.plot(frontier_vols, frontier_rets, color=ACCENT, linewidth=2.5,
             label="Efficient Frontier")

    ms = port["max_sharpe"]
    ax4.scatter(ms["vol"] * 100, ms["return"] * 100,
                color=ACCENT2, s=120, zorder=5, marker="*",
                label=f"Max Sharpe ({ms['sharpe']:.2f})")
    ax4.set_xlabel("Annualised Volatility (%)")
    ax4.set_ylabel("Annualised Return (%)")
    ax4.legend(fontsize=8, facecolor=BG, labelcolor=TEXT, framealpha=0.8)

    # ── 5. Options Summary Table ───────────────────────────────
    ax5 = fig.add_subplot(gs[2, :2])
    styled_ax(ax5, "Options Pricing Results")
    ax5.axis("off")

    headers = ["Contract", "MC Price", "BS Price / Naive", "Std Error",
               "95% CI / VR%", "Note"]
    rows = []
    for name, res in option_results.items():
        if "ci_95" in res:
            rows.append([
                name,
                f"${res['price']:.4f}",
                f"${res.get('bs_price', '—')}" if isinstance(res.get("bs_price"), str)
                    else f"${res.get('bs_price', 0):.4f}",
                f"${res['std_error']:.4f}",
                f"[{res['ci_95'][0]:.3f}, {res['ci_95'][1]:.3f}]",
                "Antithetic variates",
            ])
        else:
            rows.append([
                name,
                f"${res['price']:.4f}",
                f"${res.get('naive_price', 0):.4f}",
                f"${res['std_error']:.4f}",
                f"{res['variance_reduction_pct']:.1f}%",
                "Control variate (geo avg)",
            ])

    tbl = ax5.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("#1c2128" if r == 0 else BG)
        cell.set_text_props(color=ACCENT if r == 0 else TEXT)
        cell.set_edgecolor(GRID)
    tbl.scale(1, 1.6)

    # ── 6. Convergence of MC Estimate ─────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    styled_ax(ax6, "MC Convergence — European Call")

    opt_ref = list(option_results.values())[0]
    bs_ref  = opt_ref.get("bs_price", None)

    n_range = np.logspace(2, 5, 60).astype(int)
    mc_prices = []
    for n in n_range:
        Z = np.random.standard_normal(n)
        raw_params = OptionParams(
            S0=100, K=100, r=0.05, sigma=0.2, T=1.0, n_sims=n
        )
        ST = raw_params.S0 * np.exp(
            (raw_params.r - 0.5 * raw_params.sigma ** 2) * raw_params.T
            + raw_params.sigma * np.sqrt(raw_params.T) * Z
        )
        payoff = np.maximum(ST - raw_params.K, 0)
        mc_prices.append(np.exp(-raw_params.r * raw_params.T) * payoff.mean())

    ax6.semilogx(n_range, mc_prices, color=ACCENT, linewidth=1.2, label="MC Estimate")
    if bs_ref:
        ax6.axhline(bs_ref, color=ACCENT2, linewidth=1.5,
                    linestyle="--", label=f"BS = ${bs_ref:.4f}")
    ax6.set_xlabel("Simulations")
    ax6.set_ylabel("Option Price ($)")
    ax6.legend(fontsize=8, facecolor=BG, labelcolor=TEXT, framealpha=0.8)

    fig.suptitle(
        "Monte Carlo Simulation Suite — Quantitative Finance",
        color=TEXT, fontsize=13, fontweight="bold", y=0.98,
    )
    plt.savefig("mc_results.png",
                dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print("  → Saved chart: mc_results.png")


# ─────────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 58)
    print("  Monte Carlo Simulation Suite — Quantitative Finance")
    print("=" * 58)

    # ── Asset universe ─────────────────────────────────────────
    asset_names = ["AAPL", "MSFT", "SPY"]
    market = MarketParams(
        S0    = np.array([175.0, 330.0, 450.0]),
        mu    = np.array([0.14,  0.13,  0.10]),    # annual drift
        sigma = np.array([0.22,  0.20,  0.15]),    # annual vol
        corr  = np.array([
            [1.00, 0.72, 0.85],
            [0.72, 1.00, 0.78],
            [0.85, 0.78, 1.00],
        ]),
        T       = 1.0,
        n_steps = 252,
        n_sims  = 10_000,
    )

    print("\n[1/5] Simulating correlated GBM paths …")
    paths = simulate_gbm(market)
    print(f"      Paths shape: {paths.shape}  (assets × sims × steps)")

    # ── Options pricing ────────────────────────────────────────
    print("\n[2/5] Pricing options …")
    opt = OptionParams(S0=100, K=100, r=0.05, sigma=0.20, T=1.0,
                       n_steps=252, n_sims=100_000)
    bs_call  = black_scholes(opt, call=True)
    bs_put   = black_scholes(opt, call=False)
    eu_call  = price_european_mc(opt, call=True,  antithetic=True)
    eu_put   = price_european_mc(opt, call=False, antithetic=True)
    asian_c  = price_asian_mc(opt, call=True)

    eu_call["bs_price"] = bs_call
    eu_put["bs_price"]  = bs_put

    option_results = {
        "European Call (MC)": eu_call,
        "European Put  (MC)": eu_put,
        "Asian Call    (MC)": asian_c,
    }

    print(f"      BS Call  = ${bs_call:.4f}   MC Call  = ${eu_call['price']:.4f}  "
          f"(err: ${abs(bs_call - eu_call['price']):.4f})")
    print(f"      BS Put   = ${bs_put:.4f}    MC Put   = ${eu_put['price']:.4f}  "
          f"(err: ${abs(bs_put - eu_put['price']):.4f})")
    print(f"      Asian Call = ${asian_c['price']:.4f}  "
          f"(VR: {asian_c['variance_reduction_pct']:.1f}%)")

    # ── VaR / CVaR ─────────────────────────────────────────────
    print("\n[3/5] Computing VaR / CVaR …")
    # Simulate daily returns for the equal-weight portfolio
    w_eq      = np.ones(3) / 3
    port_rets = (paths[:, :, 1:] / paths[:, :, :-1] - 1)  # (3, sims, steps)
    port_daily = (port_rets * w_eq[:, None, None]).sum(axis=0).flatten()
    risk = compute_risk_metrics(port_daily, confidence=0.95,
                                horizon=1, portfolio_value=1_000_000)

    print(f"      95% 1-day Hist VaR  = ${risk['hist_var']:>12,.0f}")
    print(f"      95% 1-day Hist CVaR = ${risk['hist_cvar']:>12,.0f}")
    print(f"      95% 1-day Para VaR  = ${risk['param_var']:>12,.0f}")

    # ── Portfolio optimisation ─────────────────────────────────
    print("\n[4/5] Running Markowitz optimisation …")
    port = optimize_portfolio(market.mu, market.cov, n_frontier=200, n_random=5_000)
    ms = port["max_sharpe"]
    print(f"      Max Sharpe portfolio:")
    for name, w in zip(asset_names, ms["weights"]):
        print(f"        {name}: {w * 100:.1f}%")
    print(f"      Return={ms['return']*100:.1f}%  Vol={ms['vol']*100:.1f}%  "
          f"Sharpe={ms['sharpe']:.3f}")

    # ── Plotting ───────────────────────────────────────────────
    print("\n[5/5] Generating dashboard …")
    plot_results(paths, port_daily, risk, port, option_results, asset_names)

    print("\n" + "=" * 58)
    print("  Done.  Output: mc_results.png")
    print("=" * 58)


if __name__ == "__main__":
    main()
