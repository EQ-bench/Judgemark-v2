import math
import statistics
import numpy as np
import scipy.stats
import logging
from typing import Dict, List
import re
from config.constants import REFERENCE_MODEL_SCORES, NEGATIVE_MARKERS
from utils.stats import normalize

import re

def parse_scores(judge_model_response: str) -> Dict[str, float]:
    """
    Parses score lines with flexible formatting. Accepts any of these formats:
    
      **Quality:** 7.5
      **Quality**: 7.5
      Quality: 7.5
      **Quality:** **7.5**
      **Quality:** [7.5]
      Quality: [**7.5**]
      
    Any extra text outside the valid one-line score format is ignored.
    """
    # The pattern explanation:
    # 1. ^\s*                            - Allow leading whitespace at the start of the line.
    # 2. (?:\*\*)?([^\n:\[\]]{2,100}):(?:\*\*)?
    #      - Optionally match leading '**'
    #      - Capture the metric name (any characters except newline, colon, or brackets)
    #        (length between 2 and 100 characters)
    #      - Match the colon, optionally preceded/followed by '**'
    # 3. \s*                             - Allow optional whitespace after the colon.
    # 4. (?:\[)?(?:\*\*)?(-?\d+(?:\.\d+)?)(?:\*\*)?(?:\])?
    #      - Optionally match an opening bracket '['
    #      - Optionally match a '**' to indicate a bold number
    #      - Capture the number (an integer or decimal, possibly negative)
    #      - Optionally match a closing '**'
    #      - Optionally match a closing bracket ']'
    # 5. \s*$                           - Allow trailing whitespace until the end of the line.
    
    pattern = (
        r'^\s*'
        r'(?:\*\*)?([^\n:\[\]]{2,100}):(?:\*\*)?'
        r'\s*'
        r'(?:\[)?(?:\*\*)?'
        r'(-?\d+(?:\.\d+)?)'
        r'(?:\*\*)?(?:\])?'
        r'\s*$'
    )
    
    matches = re.findall(pattern, judge_model_response, re.MULTILINE)
    scores = {metric.strip(): float(score) for metric, score in matches}
    return scores


def compute_raw_score(scores: Dict[str, float], scoring_min: float = 0, scoring_max: float = 10) -> float:
    """
    Given a dict of {criteria: numeric score}, compute a single aggregated raw score by:
      1. Filtering scores that fall within the provided scoring range [scoring_min, scoring_max].
      2. For negative-themed criteria (e.g. "melodramatic", "shallow resolution", etc.), inverting 
         the value via: new_value = (scoring_min + scoring_max) - value.
      3. Taking the average over all (transformed) valid scores.
      4. Scaling the average from the original range [scoring_min, scoring_max] into a canonical 1–10 range.

    If fewer than 10 valid scores are found, returns None.
    """
    valid_scores = {k: v for k, v in scores.items() if scoring_min <= v <= scoring_max}
    
    if len(valid_scores) < 5:
        return None

    total = 0.0
    count = 0
    for criteria, val in valid_scores.items():
        crit_lower = criteria.lower().strip()
        if crit_lower in NEGATIVE_MARKERS:
            # Invert the score so that a lower original score becomes higher;
            # using (scoring_min + scoring_max) - value guarantees that if value == scoring_max (worst),
            # the inverted value is scoring_min (lowest), and vice-versa.
            new_val = (scoring_min + scoring_max) - val
        else:
            new_val = val
        total += new_val
        count += 1

    avg = total / count

    # Now, linearly map the average from [scoring_min, scoring_max] to [1, 10].
    # Formula: scaled = 1 + (avg - scoring_min) * ((10 - 1) / (scoring_max - scoring_min))
    if scoring_max == scoring_min:
        scaled = 1  # avoid division by zero; degenerate case
    else:
        scaled = 1 + (avg - scoring_min) * (9 / (scoring_max - scoring_min))
    
    return round(scaled, 2)

def confidence_interval_95(data: List[float]) -> float:
    """
    Computes the 95% confidence interval for the mean using normal approximations:
    CI95 = 1.96 * (std / sqrt(n)), for n>30 or so.
    """
    n = len(data)
    if n < 2:
        return 0.0
    mean_ = statistics.mean(data)
    stdev_ = statistics.pstdev(data) if n == 1 else statistics.stdev(data)
    ci95 = 1.96 * (stdev_ / math.sqrt(n))
    return ci95

def compute_detailed_distribution(scores):
    if not scores:
        return {}
    return {
        "count": len(scores),
        "min": round(min(scores), 3),
        "max": round(max(scores), 3),
        "mean": round(statistics.mean(scores), 3),
        "median": round(statistics.median(scores), 3),
        "stdev": round(statistics.stdev(scores) if len(scores) > 1 else 0.0, 3),
        "p10": round(float(np.percentile(scores, 10)), 3),
        "p25": round(float(np.percentile(scores, 25)), 3),
        "p75": round(float(np.percentile(scores, 75)), 3),
        "p90": round(float(np.percentile(scores, 90)), 3)
    }

def compute_model_level_stats(scores_by_model, lengths_by_model):
    model_stats = {}
    for model_name, scores in scores_by_model.items():
        lengths = lengths_by_model[model_name]
        stats = {
            "count": len(scores),
            "mean": statistics.mean(scores),
            "median": statistics.median(scores),
            "stdev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "ci95": confidence_interval_95(scores),
            "min": min(scores),
            "max": max(scores)
        }
        
        # Length correlation
        if len(lengths) == len(scores):
            corr, corr_p = scipy.stats.pearsonr(lengths, scores)
            stats["length_correlation"] = corr
            stats["length_correlation_p"] = corr_p
        
        model_stats[model_name] = stats
    return model_stats

def compute_cross_model_stats(scores_by_model_all, scores_by_model_by_iter):
    """
    Computes cross-model stats. ANOVA, Kruskal, and std_dev_across_models
    are calculated over all scores. Pearson/Kendall correlations are now
    computed on a per-iteration basis and then averaged.
    """
    # ANOVA/Kruskal portion remains over ALL model scores
    arrays = list(scores_by_model_all.values())
    f_stat, f_p = scipy.stats.f_oneway(*arrays)
    kw_stat, kw_p = scipy.stats.kruskal(*arrays)

    # Standard deviation across model means (over entire distribution)
    model_means = [statistics.mean(scores) for scores in arrays]
    std_across_models = statistics.pstdev(model_means)

    # --------------------
    # Compute correlation stats per iteration, then average
    # --------------------
    iteration_keys = set()
    for model, by_iter_dict in scores_by_model_by_iter.items():
        iteration_keys |= set(by_iter_dict.keys())  # union of all iteration keys

    iteration_pearsons = []
    iteration_kendalls = []
    
    for it_key in iteration_keys:
        # Gather means for each model in this iteration
        model_means_dict = {}
        for model, by_iter_dict in scores_by_model_by_iter.items():
            if it_key in by_iter_dict and len(by_iter_dict[it_key]) > 0:
                model_means_dict[model] = statistics.mean(by_iter_dict[it_key])
        
        # Pair each model's iteration-mean with reference score
        ref_pairs = []
        for m, mean_val in model_means_dict.items():
            if m in REFERENCE_MODEL_SCORES:
                ref_pairs.append((mean_val, REFERENCE_MODEL_SCORES[m]))
        
        # If enough models exist for correlation, compute it
        if len(ref_pairs) >= 2:
            means, refs = zip(*ref_pairs)
            p_r, _ = scipy.stats.pearsonr(means, refs)
            k_tau, _ = scipy.stats.kendalltau(means, refs)
        else:
            p_r, k_tau = (0.0, 0.0)
        
        iteration_pearsons.append(p_r)
        iteration_kendalls.append(k_tau)
    
    # Final correlation = average across iteration-level correlations
    if iteration_pearsons:
        pearson_r = statistics.mean(iteration_pearsons)
    else:
        pearson_r = 0.0

    if iteration_kendalls:
        kendall_tau = statistics.mean(iteration_kendalls)
    else:
        kendall_tau = 0.0

    return {
        "anova_f": f_stat,
        "anova_p": f_p,
        "kw_stat": kw_stat,
        "kw_p": kw_p,
        "std_dev_across_models": std_across_models,
        "pearson_r": pearson_r,
        "kendall_tau": kendall_tau,
        "normalized_components": {
            "pearson_r": normalize(pearson_r, 0.15, 0.7),
            "kendall_tau": normalize(kendall_tau, 0.15, 0.75),     # 0.60 ≈ very strong
            "anova_f": normalize(f_stat, 0.0, 350.0),
            "kw_stat":      normalize(kw_stat,      100.0, 1300.0),  # rare to exceed 400
            "std_dev": normalize(std_across_models, 0.0, 2.6)
        }
    }

def build_landmark_calibration_config(scores, desired_points=None):
    """
    Creates a piecewise-linear calibration from these raw distribution 
    landmarks: [min, Q1, median, Q3, max]
    to the given desired_points, e.g. [0, 3, 5, 7, 10].
    Returns a dict describing how to transform future scores.
    """
    if not scores or len(scores) < 2:
        # Degenerate case: no meaningful distribution
        return {
            "method": "piecewise_landmark",
            "in_landmarks": [],
            "out_landmarks": []
        }

    if desired_points is None:
        desired_points = [0, 3, 5, 7, 10]

    in_min = min(scores)
    in_q1 = float(np.percentile(scores, 25))
    in_med = float(statistics.median(scores))
    in_q3 = float(np.percentile(scores, 75))
    in_max = max(scores)

    return {
        "method": "piecewise_landmark",
        "in_landmarks": [in_min, in_q1, in_med, in_q3, in_max],
        "out_landmarks": desired_points
    }

def apply_landmark_calibration(x, config):
    """
    Apply the piecewise-linear transform defined by config:
      "in_landmarks" = [minVal, q1Val, medVal, q3Val, maxVal]
      "out_landmarks" = [outMin, outQ1, outMed, outQ3, outMax].
    If x is < min or > max, we extrapolate linearly beyond that segment.
    """
    inL = config.get("in_landmarks", [])
    outL = config.get("out_landmarks", [])
    if len(inL) != 5 or len(outL) != 5:
        # Invalid or degenerate config => just return x unchanged
        return x

    in_min, in_q1, in_med, in_q3, in_max = inL
    out_min, out_q1, out_med, out_q3, out_max = outL

    def linear_map(val, old_lo, old_hi, new_lo, new_hi):
        if abs(old_hi - old_lo) < 1e-12:
            return new_lo
        frac = (val - old_lo) / (old_hi - old_lo)
        return new_lo + frac * (new_hi - new_lo)

    # Determine which segment x belongs to:
    if x <= in_q1:
        # (in_min -> in_q1) -> (out_min -> out_q1), but possibly x < in_min => extrapolate
        return linear_map(x, in_min, in_q1, out_min, out_q1)
    elif x <= in_med:
        return linear_map(x, in_q1, in_med, out_q1, out_med)
    elif x <= in_q3:
        return linear_map(x, in_med, in_q3, out_med, out_q3)
    else:
        # (in_q3 -> in_max) -> (out_q3 -> out_max), possibly x > in_max => extrapolate
        return linear_map(x, in_q3, in_max, out_q3, out_max)

def log_score_summary(score_type: str, cross_stats: Dict, model_stats: Dict):
    """Log a readable summary of score statistics."""
    logging.info(f"\n------- {score_type} Summary -------")
    logging.info(f"ANOVA F-value: {cross_stats['anova_f']:.4f}, p={cross_stats['anova_p']:.4f}")
    logging.info(f"Kruskal-Wallis: {cross_stats['kw_stat']:.4f}, p={cross_stats['kw_p']:.4f}")
    logging.info(f"Pearson r={cross_stats['pearson_r']:.4f}")
    logging.info(f"Kendall τ={cross_stats['kendall_tau']:.4f}")
    logging.info(f"Std.Dev across models: {cross_stats['std_dev_across_models']:.4f}")
    
    logging.info("\nModel Scores:")
    sorted_models = sorted(
        model_stats.items(),
        key=lambda kv: kv[1]["mean"],
        reverse=True
    )
    for model, stats in sorted_models:
        line = f"{model:.<40} {stats['mean']:.3f} ±{stats['ci95']:.3f}"
        logging.info(line)
    logging.info("------------------------------------")
