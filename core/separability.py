
import logging
import math
import statistics
import numpy as np
import scipy.stats
from typing import Dict, List
from utils.stats import normalize

try:
    from scipy.stats import wasserstein_distance
    HAS_WASSERSTEIN = True
except ImportError:
    HAS_WASSERSTEIN = False

def cohen_d(scores1: List[float], scores2: List[float]) -> float:
    """
    Compute Cohen's d for two sets of scores.
    d = (mean2 - mean1) / pooled_stdev
    """
    if len(scores1) < 2 or len(scores2) < 2:
        return 0.0
    mean1, mean2 = statistics.mean(scores1), statistics.mean(scores2)
    var1, var2 = statistics.pvariance(scores1), statistics.pvariance(scores2)
    n1, n2 = len(scores1), len(scores2)
    pooled_var = ((n1 - 1)*var1 + (n2 - 1)*var2) / (n1 + n2 - 2)
    if pooled_var <= 1e-12:
        return 0.0
    d = (mean2 - mean1) / math.sqrt(pooled_var)
    return d

def ci_interval(scores: List[float], ci_level=0.99) -> tuple[float, float]:
    """
    Compute mean ± z*(stdev/sqrt(n)) for the specified CI level.
    Returns (low, high).
    """
    if len(scores) < 2:
        # trivial or empty
        mean_ = statistics.mean(scores) if len(scores) == 1 else 0.0
        return (mean_, mean_)
    mean_ = statistics.mean(scores)
    stdev_ = statistics.stdev(scores)
    n = len(scores)
    z = scipy.stats.norm.ppf(0.5 + ci_level/2.0)  # ~2.575 for 99% CI
    half_width = z * (stdev_ / math.sqrt(n))
    return (mean_ - half_width, mean_ + half_width)

def ci_intervals_overlap(ci1: tuple[float, float], ci2: tuple[float, float]) -> bool:
    """
    Returns True if two confidence intervals overlap.
    """
    return not (ci1[1] < ci2[0] or ci2[1] < ci1[0])

def compute_distributions_distance(scores_by_model: Dict[str, List[float]]):
    """
    Example EMD computation across all pairs, if you still want it.
    Returns an average distance plus each pair's distance.
    """
    models = list(scores_by_model.keys())
    distances = {}
    sum_dist = 0.0
    pair_count = 0

    for i in range(len(models)):
        for j in range(i+1, len(models)):
            mA, mB = models[i], models[j]
            d = -1.0
            if HAS_WASSERSTEIN and scores_by_model[mA] and scores_by_model[mB]:
                d = wasserstein_distance(scores_by_model[mA], scores_by_model[mB])
            distances[f"{mA}__{mB}"] = d
            if d >= 0.0:
                sum_dist += d
            pair_count += 1

    avg_dist = (sum_dist / pair_count) if pair_count else 0.0
    return {
        "average": avg_dist,
        "pairs": distances
    }

def compute_average_ci95(model_scores: Dict[str, List[float]]) -> float:
    """
    Compute the average 95% CI half-width across models.
    """
    if not model_scores:
        return 0.0
    half_widths = []
    z95 = 1.96
    for scores in model_scores.values():
        if len(scores) < 2:
            half_widths.append(0.0)
            continue
        stdev_ = statistics.stdev(scores)
        mean_ = statistics.mean(scores)
        n = len(scores)
        hw = z95 * (stdev_ / math.sqrt(n))
        half_widths.append(hw)
    return statistics.mean(half_widths) if half_widths else 0.0

def compute_separability_metrics(
    run_data: dict,
    scores_by_model: Dict[str, List[float]],
    label: str = "raw",
    scale_factor: float = 1.5,
) -> None:
    """
    Compute a few custom “separability” metrics:
     • 99% CI overlap only for adjacent models (fraction)
     • The *magnitude* of 99% CI overlap between adjacent models (with optional scaling of intervals)
     • Single summary measure of Cohen’s d (e.g., average of absolute values)
     • EMD across pairs (optional)
     • Weighted or “modulated” metric for average CI95

    Args:
        run_data: A dictionary to store results.
        scores_by_model: Dict of model -> list of scores.
        label: String label for grouping these results in run_data.
        scale_factor: If >1.0, intervals are expanded by that factor when
                      computing overlap magnitude. For example, 1.5 means
                      you increase the half-width of each interval by 50%.
    """
    if "separability_metrics" not in run_data:
        run_data["separability_metrics"] = {}
    run_data["separability_metrics"][label] = {}

    # ----------------------------------------------------------------
    # 1) Basic stats: model means + 99% CI
    # ----------------------------------------------------------------
    model_means = {}
    model_ci99 = {}
    for m, sc in scores_by_model.items():
        if sc:
            model_means[m] = statistics.mean(sc)
            model_ci99[m] = ci_interval(sc, ci_level=0.99)
        else:
            model_means[m] = 0.0
            model_ci99[m] = (0.0, 0.0)

    # Sort models by mean descending
    models_sorted = sorted(model_means.keys(), key=lambda x: model_means[x], reverse=True)

    # ----------------------------------------------------------------
    # 2) Original “adjacent overlap fraction” (no scaling)
    # ----------------------------------------------------------------
    adjacent_overlap = {}
    overlap_count = 0
    for i in range(len(models_sorted) - 1):
        mA, mB = models_sorted[i], models_sorted[i + 1]
        overlap = ci_intervals_overlap(model_ci99[mA], model_ci99[mB])
        adjacent_overlap[f"{mA}__{mB}"] = overlap
        if overlap:
            overlap_count += 1

    adj_frac_overlap = overlap_count / (len(models_sorted) - 1) if len(models_sorted) > 1 else 0.0

    # ----------------------------------------------------------------
    # 3) “Magnitude” of 99% CI overlap between adjacent models
    #    with optional scaling factor
    # ----------------------------------------------------------------
    def scale_interval(ci: tuple[float, float], factor: float) -> tuple[float, float]:
        """
        Given an interval (low, high), expand it about its midpoint by 'factor'.
        E.g. if factor=1.5, the half-width becomes 1.5 * (original half-width).
        """
        low, high = ci
        mid = (low + high) / 2.0
        half_width = (high - low) / 2.0
        new_half = factor * half_width
        return (mid - new_half, mid + new_half)

    def interval_overlap(ciA: tuple[float, float], ciB: tuple[float, float]) -> float:
        """Return the length of the overlap between two intervals."""
        return max(0.0, min(ciA[1], ciB[1]) - max(ciA[0], ciB[0]))

    adjacent_overlap_magnitude = {}
    sum_overlap_magnitude = 0.0
    for i in range(len(models_sorted) - 1):
        mA, mB = models_sorted[i], models_sorted[i + 1]
        # Scale each interval before computing overlap
        scaledA = scale_interval(model_ci99[mA], scale_factor)
        scaledB = scale_interval(model_ci99[mB], scale_factor)
        overlap_mag = interval_overlap(scaledA, scaledB)
        adjacent_overlap_magnitude[f"{mA}__{mB}"] = overlap_mag
        sum_overlap_magnitude += overlap_mag

    # ----------------------------------------------------------------
    # 4) Single measure for Cohen’s d (average of absolute Cohen’s d across adjacent pairs)
    # ----------------------------------------------------------------
    d_vals = []
    for i in range(len(models_sorted) - 1):
        mA, mB = models_sorted[i], models_sorted[i + 1]
        d_val = cohen_d(scores_by_model[mA], scores_by_model[mB])
        d_vals.append(abs(d_val))
    avg_cohens_d = sum(d_vals) / len(d_vals) if d_vals else 0.0

    # ----------------------------------------------------------------
    # 5) Optional EMD across all pairs
    # ----------------------------------------------------------------
    emd_data = compute_distributions_distance(scores_by_model)

    # ----------------------------------------------------------------
    # 6) Weighted or modulated average CI95
    # ----------------------------------------------------------------
    avg_ci95 = compute_average_ci95(scores_by_model)
    norm_ci95 = normalize(avg_ci95, 0.15, 0.45, False)
    norm_cohens_d = normalize(avg_cohens_d, 0, 0.4)
    modulated_ci95 = norm_ci95 * norm_cohens_d

    # ----------------------------------------------------------------
    # Store or log results
    # ----------------------------------------------------------------
    metrics_label = run_data["separability_metrics"][label]
    metrics_label["ci99_overlap_adjacent"] = adjacent_overlap
    metrics_label["adjacent_overlap_fraction"] = adj_frac_overlap

    # New overlap magnitude stats (with scaling)
    metrics_label["ci99_overlap_magnitude_adjacent"] = adjacent_overlap_magnitude
    metrics_label["ci99_overlap_magnitude_sum"] = sum_overlap_magnitude
    metrics_label["ci99_overlap_scale_factor"] = scale_factor

    metrics_label["average_cohens_d_adjacent"] = avg_cohens_d
    metrics_label["emd"] = emd_data
    metrics_label["average_ci95"] = avg_ci95
    metrics_label["modulated_ci95"] = modulated_ci95

    # Logging summary
    logging.info(f"\n--- {label.upper()} SEPARABILITY METRICS ---")
    logging.info(f"Adjacent 99% CI Overlap fraction: {adj_frac_overlap:.3f}")
    logging.info(f"Sum of adjacent 99% CI Overlap magnitude (scale={scale_factor}): "
                 f"{sum_overlap_magnitude:.3f}")
    logging.info(f"Avg. |Cohen's d| for adjacent pairs: {avg_cohens_d:.3f}")
    logging.info(f"Average EMD across all pairs: {emd_data['average']:.3f}")
    logging.info(f"Avg. CI95 half-width: {avg_ci95:.3f} (modulated: {modulated_ci95:.3f})")