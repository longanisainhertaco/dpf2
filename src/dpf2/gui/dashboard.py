"""Utilities for launching the built-in web dashboard and analysis helpers."""
from __future__ import annotations

from typing import Callable, Mapping, Sequence, Tuple, Dict, Any

import numpy as np

try:  # pragma: no cover - optional plotting dependency
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - fallback when matplotlib missing
    plt = None

from ..web.app import create_app
from ..uq import sampling, analysis, calibration

Bounds = Mapping[str, Tuple[float, float]]


def launch(host: str = "127.0.0.1", port: int = 5000, **kwargs) -> None:
    """Start the Flask-based dashboard.

    Parameters
    ----------
    host, port:
        Network location where the application should listen.
    kwargs:
        Additional keyword arguments forwarded to :func:`Flask.run`.
    """

    app = create_app()
    app.run(host=host, port=port, **kwargs)


def run_sampling(
    model: Callable[[np.ndarray], float | Sequence[float]],
    bounds: Bounds,
    n_samples: int = 100,
    method: str = "lhs",
    alpha: float = 0.95,
    seed: int | None = None,
    ax: Any | None = None,
) -> Dict[str, Any]:
    """Run sampling experiments and visualize uncertainty bands.

    Parameters
    ----------
    model:
        Callable returning a scalar prediction for each parameter vector.
    bounds:
        Mapping of parameter names to ``(min, max)`` tuples.
    n_samples:
        Number of samples to generate.
    method:
        ``"lhs"`` for Latin hypercube or ``"sobol"`` for Sobol sequences.
    alpha:
        Central credibility interval for the uncertainty band.
    seed:
        Optional seed for reproducibility.
    ax:
        Optional :class:`matplotlib.axes.Axes` to draw on.  If omitted a new
        figure is created when :mod:`matplotlib` is available.

    Returns
    -------
    dict
        Dictionary containing the samples, model values and computed
        uncertainty band statistics.  When plotting is available the axis
        object is also included under ``"ax"``.
    """

    if method.lower() in {"lhs", "latin", "latin_hypercube"}:
        samples = sampling.latin_hypercube(bounds, n_samples, seed=seed)
    elif method.lower() in {"sobol", "sobol_sequence"}:
        samples = sampling.sobol_sample(bounds, n_samples, seed=seed)
    else:  # pragma: no cover - invalid input path
        raise ValueError(f"Unknown sampling method '{method}'")

    values = [float(np.asarray(model(s))) for s in samples]
    band = analysis.uncertainty_band(values, alpha=alpha)

    if plt is not None:
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(range(len(values)), values, marker="o", linestyle="-", label="model")
        lower = [band["lower"]] * len(values)
        upper = [band["upper"]] * len(values)
        ax.fill_between(range(len(values)), lower, upper, color="C0", alpha=0.2,
                        label=f"{alpha*100:.0f}% interval")
        ax.axhline(band["mean"], color="C1", linestyle="--", label="mean")
        ax.set_xlabel("sample")
        ax.set_ylabel("model output")
        ax.legend()

    result: Dict[str, Any] = {"samples": samples, "values": values, "band": band}
    if plt is not None:
        result["ax"] = ax
    return result


def launch_sampling(
    model: Callable[[np.ndarray], float | Sequence[float]],
    bounds: Bounds,
    n_samples: int = 100,
    *,
    method: str = "lhs",
    alpha: float = 0.95,
    seed: int | None = None,
    ax: Any | None = None,
) -> Dict[str, Any]:
    """Convenience wrapper for :func:`run_sampling` used by dashboards.

    The function simply forwards all arguments to :func:`run_sampling` but
    exposes a more GUI-friendly name so frontends can trigger Latin hypercube
    or Sobol sampling without dealing with implementation details.
    """

    return run_sampling(
        model, bounds, n_samples, method=method, alpha=alpha, seed=seed, ax=ax
    )


def calibrate_from_file(
    model: Callable[[np.ndarray], np.ndarray],
    bounds: Bounds,
    data_file: str,
    method: str = "bayesian",
    ax: Any | None = None,
    **kwargs: Any,
) -> Dict[str, np.ndarray]:
    """Calibrate model parameters against experimental data in ``data_file``.

    The function loads numerical data from ``data_file`` using
    :func:`numpy.loadtxt`, runs the selected calibration routine and, when
    :mod:`matplotlib` is available, plots posterior histograms.
    """

    data = np.loadtxt(data_file)
    if method == "bayesian":
        post = calibration.bayesian_calibration(model, bounds, data, **kwargs)
    elif method == "nested":
        post = calibration.nested_calibration(model, bounds, data, **kwargs)
    else:  # pragma: no cover - invalid method
        raise ValueError(f"Unknown calibration method '{method}'")

    if plt is not None:
        names = list(post)
        samples = [post[n] for n in names]
        n = len(names)
        if ax is None:
            fig, axes = plt.subplots(1, n, figsize=(4 * n, 3))
            if n == 1:
                axes = [axes]
        else:
            axes = [ax]
        for axis, name, vals in zip(axes, names, samples):
            axis.hist(vals, bins=30, density=True)
            axis.set_title(name)
        if ax is None:
            fig.tight_layout()

    return post


def plot_posterior_distributions(
    posterior: Mapping[str, Sequence[float]],
    ax: Any | None = None,
) -> Any:
    """Plot marginal posterior distributions for calibrated parameters.

    Parameters
    ----------
    posterior:
        Mapping of parameter names to posterior samples, typically the output
        of :func:`calibrate_from_file`.
    ax:
        Optional :class:`matplotlib.axes.Axes` to draw on.  When ``None`` a new
        figure is created when :mod:`matplotlib` is available.  Multiple
        parameters result in a row of subplots.

    Returns
    -------
    Any
        The axis or list of axes used for plotting, or ``None`` when plotting
        is unavailable.
    """

    if plt is None:  # pragma: no cover - plotting not available
        return None

    names = list(posterior)
    samples = [posterior[n] for n in names]
    n = len(names)
    if ax is None:
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 3))
        if n == 1:
            axes = [axes]
    else:
        axes = [ax]
    for axis, name, vals in zip(axes, names, samples):
        axis.hist(vals, bins=30, density=True)
        axis.set_title(name)
    if ax is None:
        fig.tight_layout()
    return axes if ax is None else axes[0]


def plot_kpi_with_domain(
    x: Sequence[float],
    y: Sequence[float],
    y_err: Sequence[float],
    training_domain: Tuple[float, float],
    ax: Any | None = None,
) -> Any:
    """Plot KPI values with error bars and highlight training domain.

    Parameters
    ----------
    x, y:
        Coordinates of the KPI samples.
    y_err:
        Symmetric error bars associated with ``y``.
    training_domain:
        ``(min, max)`` range of the surrogate model's training data.  The
        region is shaded on the plot.
    ax:
        Optional :class:`matplotlib.axes.Axes` to draw on.  A new figure is
        created when omitted and :mod:`matplotlib` is available.
    """

    if plt is None:  # pragma: no cover - plotting not available
        return None

    if ax is None:
        _, ax = plt.subplots()
    ax.errorbar(x, y, yerr=y_err, fmt="o", label="KPI")
    ax.axvspan(training_domain[0], training_domain[1], color="grey", alpha=0.1,
               label="training domain")
    lower = [val - err for val, err in zip(y, y_err)]
    upper = [val + err for val, err in zip(y, y_err)]
    ax.fill_between(x, lower, upper, color="C0", alpha=0.2, label="error band")
    ax.set_xlabel("parameter")
    ax.set_ylabel("KPI")
    ax.legend()
    return ax


__all__ = [
    "launch",
    "run_sampling",
    "launch_sampling",
    "calibrate_from_file",
    "plot_posterior_distributions",
    "plot_kpi_with_domain",
]
