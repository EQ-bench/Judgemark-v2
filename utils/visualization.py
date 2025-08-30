import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Dict
from scipy.stats import linregress
from scipy.stats import spearmanr, theilslopes
from config.constants import NEGATIVE_MARKERS, MODEL_NAME_REPLACEMENTS

def create_side_by_side_score_charts(run_data: Dict, judge_model: str, samples_data: Dict,
                                     scoring_min: float = 0, scoring_max: float = 10):
    """
    Produces three figures:
      • Figure #1 with three subplots side-by-side:
          (1) Raw Scores bar chart (+ 95 % CI)
          (2) Calibrated Scores bar chart (+ 95 % CI)
          (3) Heat-map of all per-criterion scores across each model (with negative markers flipped),
              with an extra spacer row (all 0 %) second-to-bottom and the combined row at the bottom.
              The colour-map scale is fixed from 0 to 45 %.
      • Figure #2: A 4×4 grid of mini scatter-plots (one per model, up to 16),
          showing item length (chars) vs. aggregated_score_raw. A linear regression
          line and correlation stats are included for each model if enough points exist.
      • Figure #3: A standalone heat-map showing the Numeric Scoring Distribution.
          Title: "Numeric Scoring Distribution — Judge: [judge model]"
          (The colour-map scale is fixed from 0 to 45 %.)
    """
    # -------------------------------------------------------------------
    # 1) The main (raw / calibrated / heat-map) figure
    # -------------------------------------------------------------------
    raw_stats = run_data["raw_model_stats"]
    cal_stats = run_data["calibrated_model_stats"]

    if judge_model in MODEL_NAME_REPLACEMENTS:
        judge_model = MODEL_NAME_REPLACEMENTS[judge_model]

    # All model names in raw_stats
    model_names = list(raw_stats.keys())

    # Convert to arrays for sorting
    raw_means = [raw_stats[m]["mean"] for m in model_names]
    cal_means = [cal_stats[m]["mean"] for m in model_names]
    raw_cis   = [raw_stats[m]["ci95"] for m in model_names]
    cal_cis   = [cal_stats[m]["ci95"] for m in model_names]

    # Sort by calibrated score descending
    sorted_indices = np.argsort(cal_means)[::-1]
    model_names = [model_names[i] for i in sorted_indices]
    raw_means   = [raw_means[i]   for i in sorted_indices]
    cal_means   = [cal_means[i]   for i in sorted_indices]
    raw_cis     = [raw_cis[i]     for i in sorted_indices]
    cal_cis     = [cal_cis[i]     for i in sorted_indices]

    # 1.A) Build data for the heat-map: per-criterion scores (with negative flips)
    bins = np.arange(scoring_min - 0.5, scoring_max + 1, 1)        # bin edges
    bin_centers = np.arange(scoring_min, scoring_max + 1)          # labels

    all_scores_by_model = {m: [] for m in model_names}
    results = run_data.get("results", {})

    for model_name in model_names:
        iter_dict = results.get(model_name, {})
        for iteration_key, item_dict in iter_dict.items():
            if not isinstance(item_dict, dict):
                continue
            for item_id, item_info in item_dict.items():
                if not isinstance(item_info, dict):
                    continue

                # -----------  NEW: collect scores from both old and new paths -----------
                score_sources = []

                # (1) legacy single-judge path
                if "parsed_scores" in item_info and isinstance(item_info["parsed_scores"], dict):
                    score_sources.append(item_info["parsed_scores"])

                # (2) ensemble path introduced in the refactor
                for jo in item_info.get("judge_outputs", []):
                    if isinstance(jo, dict) and "parsed_scores" in jo and isinstance(jo["parsed_scores"], dict):
                        score_sources.append(jo["parsed_scores"])
                # -----------------------------------------------------------------------

                for parsed_scores in score_sources:
                    for crit_name, val in parsed_scores.items():
                        if isinstance(val, (int, float)) and scoring_min <= val <= scoring_max:
                            crit_lower = crit_name.strip().lower()
                            # Flip negative markers
                            final_val = (scoring_max - val + scoring_min) if crit_lower in NEGATIVE_MARKERS else val
                            all_scores_by_model[model_name].append(final_val)

    heatmap_rows = []
    for m in model_names:
        scores = all_scores_by_model[m]
        if scores:
            counts, _ = np.histogram(scores, bins=bins)
            pct = (counts / len(scores)) * 100.0
        else:
            pct = np.zeros(len(bins) - 1, dtype=float)
        heatmap_rows.append(pct)
    heatmap_data = np.array(heatmap_rows, dtype=float)

    # ------------------------------- spacer + combined rows -------------------------------
    all_combined_scores = []
    for scores in all_scores_by_model.values():
        all_combined_scores.extend(scores)
    if all_combined_scores:
        counts_combined, _ = np.histogram(all_combined_scores, bins=bins)
        combined_pct = (counts_combined / len(all_combined_scores)) * 100.0
    else:
        combined_pct = np.zeros(len(bins) - 1, dtype=float)

    spacer = np.zeros(heatmap_data.shape[1], dtype=float)
    heatmap_data = np.vstack([heatmap_data, spacer, combined_pct])
    heatmap_model_names = model_names + ["", "Combined"]
    # --------------------------------------------------------------------------------------

    # 1.B) Plot main figure
    fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))
    plt.rcParams.update({'font.size': 14})

    # (A) Raw bar-chart
    y_pos = np.arange(len(model_names))
    ax1.barh(y_pos, raw_means, color='skyblue', alpha=0.7)
    for i, (mean_val, ci95) in enumerate(zip(raw_means, raw_cis)):
        ax1.errorbar(mean_val, i, xerr=ci95, color='red', capsize=5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(model_names, fontsize=12)
    ax1.invert_yaxis()
    ax1.set_xlabel("Raw Scores", fontsize=14)
    ax1.set_title("Raw Model Scores (95 % CI)", fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='y', pad=10)

    # (B) Calibrated bar-chart
    ax2.barh(y_pos, cal_means, color='lightgreen', alpha=0.7)
    for i, (mean_val, ci95) in enumerate(zip(cal_means, cal_cis)):
        ax2.errorbar(mean_val, i, xerr=ci95, color='red', capsize=5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(model_names, fontsize=12)
    ax2.invert_yaxis()
    ax2.set_xlabel("Calibrated Scores", fontsize=14)
    ax2.set_title("Calibrated Model Scores (95 % CI)", fontsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='y', pad=10)

    # (C) Heat-map
    ax3.set_xticks(np.arange(len(bin_centers)))
    ax3.set_xticklabels([str(int(bc)) for bc in bin_centers])
    im = ax3.imshow(heatmap_data, aspect='auto', origin='upper',
                    cmap='plasma', vmin=0, vmax=45)
    ax3.set_yticks(np.arange(len(heatmap_model_names)))
    ax3.set_yticklabels(heatmap_model_names, fontsize=12)
    ax3.set_xlabel(f"Score Bin ({scoring_min}–{scoring_max})", fontsize=14)
    ax3.set_title("Per-Criterion Score Distribution (Heat-map)", fontsize=16)
    ax3.tick_params(axis='y', pad=10)

    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label("% of Criteria in Bin", fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.formatter = ticker.PercentFormatter(decimals=1)
    cbar.update_ticks()

    sanitized_judge = re.sub(r"[^\w\-]", "-", judge_model.replace("/", "__"))
    fig1.suptitle(f"Judgemark: Raw / Calibrated / Heat-map – Judge: {judge_model}", fontsize=20)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"results/charts/judgemark_3chart_{sanitized_judge}.png",
                bbox_inches='tight', dpi=150, pad_inches=0.5)
    plt.close(fig1)

    # -------------------------------------------------------------------
    # 2) Scatter-grid (unchanged)
    # -------------------------------------------------------------------
    excluded_models = {"gemini-1.5-pro-001"}
    model_list_for_scatter = [m for m in model_names if m not in excluded_models]

    if len(model_list_for_scatter) > 16:
        model_list_for_scatter = model_list_for_scatter[:16]

    fig2, axes2 = plt.subplots(4, 4, figsize=(20, 20))
    fig2.suptitle(f"Judgemark: Per-Model Length vs. Score – Judge: {judge_model}", fontsize=18)

    for idx, mname in enumerate(model_list_for_scatter):
        row = idx // 4
        col = idx % 4
        ax = axes2[row, col]

        length_vals = []
        score_vals  = []

        model_res = run_data["results"].get(mname, {})
        for it_key, it_dict in model_res.items():
            if not isinstance(it_dict, dict):
                continue
            for item_id, item_info in it_dict.items():
                if not isinstance(item_info, dict):
                    continue
                raw_score = item_info.get("aggregated_score_raw", None)
                if not isinstance(raw_score, (int, float)):
                    continue

                text = (samples_data
                        .get(mname, {})
                        .get("samples", {})
                        .get(it_key, {})
                        .get(item_id, ""))
                text_len = len(text)

                if text_len > 0:
                    length_vals.append(text_len)
                    score_vals.append(raw_score)

        ax.set_title(mname, fontsize=12)
        ax.set_xlabel("Length")
        ax.set_ylabel("Raw Score")

        if len(length_vals) > 1:
            ax.scatter(length_vals, score_vals, alpha=0.4, color='blue')

            rho, p_value = spearmanr(length_vals, score_vals)
            slope, intercept, _, _ = theilslopes(score_vals, length_vals, alpha=0.95)

            xline = np.linspace(min(length_vals), max(length_vals), 200)
            yline = slope * xline + intercept
            ax.plot(xline, yline, color='red', linewidth=2,
                    label=f"Spearman ρ={rho:.2f}, p={p_value:.2g}")
            ax.legend(loc="best")
        else:
            ax.text(0.5, 0.5, "Not enough data",
                    ha='center', va='center', color='red', transform=ax.transAxes)

    total_subplots = 16
    for i in range(len(model_list_for_scatter), total_subplots):
        row = i // 4
        col = i % 4
        axes2[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(f"results/charts/judgemark_scattergrid_{sanitized_judge}.png",
                bbox_inches='tight', dpi=200)
    plt.close(fig2)

    # -------------------------------------------------------------------
    # 3) Stand-alone Numeric-Distribution Heat-map (unchanged)
    # -------------------------------------------------------------------
    fig3, ax_heatmap = plt.subplots(figsize=(10, 7))
    im_heatmap = ax_heatmap.imshow(heatmap_data, aspect='auto', origin='upper',
                                   cmap='plasma', vmin=0, vmax=45)
    ax_heatmap.set_xticks(np.arange(len(bin_centers)))
    ax_heatmap.set_xticklabels([str(int(bc)) for bc in bin_centers], fontsize=12)
    ax_heatmap.set_yticks(np.arange(len(heatmap_model_names)))
    ax_heatmap.set_yticklabels(heatmap_model_names, fontsize=12)
    ax_heatmap.set_xlabel(f"Score Bin ({scoring_min}–{scoring_max})", fontsize=14)
    ax_heatmap.set_ylabel("Model", fontsize=14)
    ax_heatmap.set_title(f"Numeric Scoring Distribution – Judge: {judge_model}", fontsize=16)

    cbar_heatmap = plt.colorbar(im_heatmap, ax=ax_heatmap)
    cbar_heatmap.set_label("% of Criteria in Bin", fontsize=14)
    cbar_heatmap.ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(f"results/charts/judgemark_heatmap_{sanitized_judge}.png",
                bbox_inches='tight', dpi=150)
    plt.close(fig3)