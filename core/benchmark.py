import os
import re
import uuid
import time
import signal
import logging
import threading
import concurrent.futures
from datetime import datetime
from typing import Dict, List
from tqdm import tqdm
from collections import defaultdict
import itertools
import hashlib

from utils.file_io import load_json_file, save_json_file, load_text_file
from utils.api import send_to_judge_model
from utils.visualization import create_side_by_side_score_charts
import statistics
from core.scoring import (
    parse_scores, compute_raw_score, compute_detailed_distribution,
    compute_model_level_stats, compute_cross_model_stats,
    build_landmark_calibration_config, apply_landmark_calibration,
    log_score_summary, confidence_interval_95
)
from core.scoring import compute_detailed_distribution, compute_detailed_distribution  # etc
from core.separability import compute_separability_metrics
from core.stability import run_stability_test, compute_iteration_stability, compute_randomized_iteration_rank_stability_by_item
from utils.stats import normalize, modulate_x_by_y
from utils.state import should_exit, executor

def aggregate_ensemble_scores(judge_outputs_list: List[Dict], method: str) -> float:
    """Aggregates raw scores from a list of judge outputs for a single item."""
    scores = []
    for judge_output in judge_outputs_list:
        if judge_output and "aggregated_score_raw" in judge_output and judge_output["aggregated_score_raw"] is not None:
            scores.append(judge_output["aggregated_score_raw"])
    
    if not scores:
        return None
    
    if method == 'average':
        return statistics.mean(scores)
    elif method == 'median':
        return statistics.median(scores)
    return None

def process_sample(model_name: str, iteration_key: str, item_id: str, item_text: str, 
                  prompt_template: str, run_key: str, runs: Dict, runs_file: str,
                  lock: threading.Lock, judge_model: str, judge_index: int, save_raw_judge_output: bool):
    """Process a single sample for a single judge instance, storing the result by index."""
    global should_exit
    if should_exit:
        return
    
    run_data = runs.get(run_key, {})
    
    try:
        final_prompt = prompt_template.replace("[TEST MODEL RESPONSE]", '[TEST MODEL RESPONSE]\n' + item_text)
        
        messages = [{"role": "user", "content": final_prompt}]
        judge_response = send_to_judge_model(messages, judge_model=judge_model)
        
        extracted_scores = parse_scores(judge_response)
        scoring_range = run_data.get("scoring_range", {"min": 0, "max": 10})
        scoring_min = scoring_range.get("min", 0)
        scoring_max = scoring_range.get("max", 10)
        raw_score = compute_raw_score(extracted_scores, scoring_min, scoring_max)
        
        with lock:
            item_storage = runs[run_key]["results"][model_name][iteration_key][item_id]
            all_judges_in_run = run_data.get("judge_models", [])
            
            # Ensure judge_outputs is a list of the correct size
            if "judge_outputs" not in item_storage or not isinstance(item_storage["judge_outputs"], list):
                item_storage["judge_outputs"] = [None] * len(all_judges_in_run)
            # Handle case where list is too short from a previous interrupted run
            if len(item_storage["judge_outputs"]) < len(all_judges_in_run):
                item_storage["judge_outputs"].extend([None] * (len(all_judges_in_run) - len(item_storage["judge_outputs"])))


            # This is the result from a single judge instance
            judge_result = {
                "judge_model": judge_model,
                "parsed_scores": extracted_scores,
                "timestamp": datetime.now().isoformat(),
                "text_length": len(item_text)
            }
            if raw_score is not None:
                judge_result["aggregated_score_raw"] = raw_score
            if save_raw_judge_output:
                judge_result["judge_response"] = judge_response
            
            # Store the single judge's result at its designated index
            item_storage["judge_outputs"][judge_index] = judge_result

            # Check if all judges in the ensemble have finished (no None placeholders left)
            if None not in item_storage["judge_outputs"]:
                final_aggregated_score = aggregate_ensemble_scores(item_storage["judge_outputs"], run_data.get("ensemble_method", "average"))
                if final_aggregated_score is not None:
                    item_storage["aggregated_score_raw"] = final_aggregated_score
            
            save_json_file(runs, runs_file)
        
        if raw_score is not None:
            logging.debug(f"Processed {model_name}/{iteration_key}/{item_id} with {judge_model} (index {judge_index}), raw score: {raw_score:.2f}")
        else:
            logging.warning(f"Failed to parse enough scores for {model_name}/{iteration_key}/{item_id} with {judge_model} (index {judge_index})")
            
    except Exception as e:
        logging.error(f"Error processing item {model_name}/{iteration_key}/{item_id} with {judge_model} (index {judge_index}): {str(e)}")
        with lock:
            item_storage = runs[run_key]["results"][model_name][iteration_key][item_id]
            all_judges_in_run = run_data.get("judge_models", [])
            if "judge_outputs" not in item_storage or not isinstance(item_storage["judge_outputs"], list):
                item_storage["judge_outputs"] = [None] * len(all_judges_in_run)

            item_storage["judge_outputs"][judge_index] = {
                "judge_model": judge_model,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            if "errors" not in run_data:
                run_data["errors"] = []
            run_data["errors"].append({
                "model": model_name,
                "iteration": iteration_key,
                "item_id": item_id,
                "judge_model": judge_model,
                "judge_index": judge_index,
                "error": str(e)
            })
            save_json_file(runs, runs_file)

def get_deterministic_permutation(items: List, key: str) -> List:
    """Gets a deterministic permutation of a list based on a string key."""
    if not items:
        return []
    permutations = sorted(list(itertools.permutations(items)))
    key_hash = int(hashlib.sha256(key.encode('utf-8')).hexdigest(), 16)
    index = key_hash % len(permutations)
    return list(permutations[index])

def process_sample_book_club(
    model_name: str,
    iteration_key: str,
    item_id: str,
    item_text: str,
    base_prompt_template: str,
    writing_prompt: str,
    run_key: str,
    runs: Dict,
    runs_file: str,
    lock: threading.Lock,
    judge_models: List[str],
    save_raw_judge_output: bool,
    rubric_criteria: str,
    score_anchoring: str,
    merits_discussion_rounds: int = 1,
    scoring_discussion_rounds: int = 1
):
    """
    Orchestrates a multi-step, conversational Book Club judging process for a single item.
    This version correctly handles duplicate judge models and allows for multiple discussion rounds.
    """
    global should_exit
    if should_exit:
        return

    run_data = runs[run_key]
    item_storage = runs[run_key]["results"].setdefault(model_name, {}).setdefault(iteration_key, {}).setdefault(item_id, {})
    
    # If final scores are already present, the entire process for this item is complete.
    if "aggregated_score_raw" in item_storage and item_storage["aggregated_score_raw"] is not None:
        return

    logging.info(f"Starting Book Club for {model_name}/{iteration_key}/{item_id}")

    try:
        item_storage["book_club_data"] = book_club_data = {}
        
        # Treat each entry in judge_models as a unique participant, even if names are duplicated.
        # A participant is identified by an (index, model_name) tuple.
        participants = list(enumerate(judge_models))
        turn_order = get_deterministic_permutation(participants, f"{run_key}-{item_id}")
        
        book_club_data["participants"] = participants
        book_club_data["turn_order"] = turn_order
        discussion_history = []

        # --- Stage 1: Initial Synopses ---
        logging.debug(f"Book Club Stage 1: Initial Synopses for {item_id}")
        prompt_stage1 = f"""You are participating in a lit crit session. Your primary role is to be discriminative of good & bad writing. Just be honest and objective.
Read the following text carefully.

[WRITING PROMPT]
{writing_prompt}
[WRITING PROMPT END]

[TEST MODEL RESPONSE]
{item_text}
[TEST MODEL RESPONSE END]

Your first task is to independently write a brief synopsis of the three things that stood out to you the most, both positive and negative, as good fodder for discussing the merits of the piece. Bear in mind the writing task's instruction was to write 1000 words, so expect a short piece. Length is not the focus here; just discuss what is there.
Do NOT score the piece yet. Focus only on these impressions.
"""
        
        # Use a list to store synopses, indexed by the participant's original index.
        initial_synopses = [None] * len(participants)
        print(prompt_stage1)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(participants)) as sub_executor:
            future_to_participant = {
                sub_executor.submit(send_to_judge_model, [{"role": "user", "content": prompt_stage1}], model, 0.5): (idx, model)
                for idx, model in participants
            }
            for future in concurrent.futures.as_completed(future_to_participant):
                idx, model = future_to_participant[future]
                try:
                    synopsis = future.result()
                    initial_synopses[idx] = {"participant_id": idx, "model": model, "synopsis": synopsis}
                    discussion_history.append(f"Initial thoughts from {model} (participant #{idx}):\n{synopsis}\n")
                except Exception as e:
                    logging.error(f"Book Club Stage 1 failed for participant #{idx} ({model}) on item {item_id}: {e}")
                    initial_synopses[idx] = {"participant_id": idx, "model": model, "error": str(e)}
        book_club_data["initial_synopses"] = initial_synopses

        # --- Stage 2: Discussion on Merits (can have multiple rounds) ---
        logging.debug(f"Book Club Stage 2: {merits_discussion_rounds} round(s) of discussion on merits for {item_id}")
        book_club_data["discussion_log_merits"] = []
        for round_num in range(merits_discussion_rounds):
            logging.debug(f"Merits Discussion Round {round_num + 1}/{merits_discussion_rounds}")
            
            for idx, model in turn_order:
                history_str = "\n---\n".join(discussion_history)
                if should_exit: return
                prompt_stage2 = f"""You are {model} (participant #{idx}) in a lit crit session. Your primary role is to be discriminative of good & bad writing. Just be honest and objective. Bear in mind the writing task's instruction was to write 1000 words, so expect a short piece. Length is not the focus here; just discuss what is there.
The group is discussing the following text:

[WRITING PROMPT]
{writing_prompt}
[WRITING PROMPT END]

[TEST MODEL RESPONSE]
{item_text}
[TEST MODEL RESPONSE END]

The discussion so far, based on everyone's initial independent thoughts and the ongoing conversation:
--- DISCUSSION HISTORY ---
{history_str}
--- END DISCUSSION HISTORY ---

It is now your turn. Please provide your deconstruction and discussion of the previous points and the text itself. Build upon, critique, or offer new perspectives on the piece's merits and failures. Aim to find new insights and points of contention, instead of simply agreeing with the other participants.
"""
                print(prompt_stage2)
                response = send_to_judge_model([{"role": "user", "content": prompt_stage2}], model, 0.5)
                turn_log = f"Discussion turn from {model} (participant #{idx}):\n{response}\n"
                discussion_history.append(turn_log)
                book_club_data["discussion_log_merits"].append({"participant_id": idx, "model": model, "turn_log": response})

        # --- Stage 3: Discussion on Scoring (can have multiple rounds) ---
        logging.debug(f"Book Club Stage 3: {scoring_discussion_rounds} round(s) of discussion on scoring for {item_id}")
        book_club_data["discussion_log_scoring"] = []
        i=0
        for round_num in range(scoring_discussion_rounds):
            logging.debug(f"Scoring Discussion Round {round_num + 1}/{scoring_discussion_rounds}")
            
            for idx, model in turn_order:
                history_str = "\n---\n".join(discussion_history)
                i += 1
                if i == 1:
                    scoring_instructions = "Discuss how you would approach assigning scores based on the rubric and the conversation so far"
                else:
                    scoring_instructions = "Continue the discussion of the scoring of the piece."
                if should_exit: return
                prompt_stage3 = f"""You are {model} (participant #{idx}) in a lit crit session. Your primary role is to be discriminative of good & bad writing. Just be honest and objective. Bear in mind the writing task's instruction was to write 1000 words, so expect a short piece. Length is not the focus here; just discuss what is there.
The group is discussing the following text:

[WRITING PROMPT]
{writing_prompt}
[WRITING PROMPT END]

[TEST MODEL RESPONSE]
{item_text}
[TEST MODEL RESPONSE END]
                
The discussion will now shift to scoring.
The full discussion history is below:
--- FULL DISCUSSION HISTORY ---
{history_str}
--- END FULL DISCUSSION HISTORY ---

The scoring rubric is as follows:
<RUBRIC_CRITERIA>
{rubric_criteria}
</RUBRIC_CRITERIA>

And the scores MUST be anchored to these descriptions:
<SCORE_ANCHORING>
{score_anchoring}
</SCORE_ANCHORING>

It's very important to note that this isn't a typical scoring range. Here, a score of 2-3 isn't necessarily *bad* writing. Consider the anchors very carefully and match the merits of the piece to ability range anchors.

A note: Some criteria may not be relevant to the piece (e.g. the piece may not have dialogue), in which case ignore the irrelevant criteria.

It is your turn, {model} (participant #{idx}). {scoring_instructions}.

"""             
                print(prompt_stage3)
                response = send_to_judge_model([{"role": "user", "content": prompt_stage3}], model, 0.5)
                turn_log = f"Scoring discussion turn from {model} (participant #{idx}):\n{response}\n"
                discussion_history.append(turn_log)
                book_club_data["discussion_log_scoring"].append({"participant_id": idx, "model": model, "turn_log": response})

        # --- Final Stage: Independent Scoring ---
        logging.debug(f"Book Club Final Stage: Independent Scoring for {item_id}")
        final_context = "\n---\n".join(discussion_history)
        judging_prompt = base_prompt_template.replace("[TEST MODEL RESPONSE]", '[TEST MODEL RESPONSE]\n' + item_text)
        final_prompt_template = f"""You have participated in a detailed lit crit discussion about a piece of writing. Your primary role is to be discriminative of good & bad writing. Just be honest and objective. Bear in mind the writing task's instruction was to write 1000 words, so expect a short piece. Length is not the focus here; just evaluate what is there.
The full discussion history is provided below for your context. Your task is now to provide your final, independent scores.
Base your judgment on the full text, the discussion, and the provided rubric.

--- FULL DISCUSSION HISTORY ---
{final_context}
--- END FULL DISCUSSION HISTORY ---

Please provide your final scores now using the official rubric.
{judging_prompt}
"""
        
        print(final_prompt_template)
        
        # Final scoring uses ALL judge instances from the original list, calling the standard
        # process_sample function which correctly handles list-based, indexed results.
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(judge_models)) as sub_executor:
            final_futures = []
            for judge_index, judge_model in enumerate(judge_models):
                fut = sub_executor.submit(process_sample, model_name, iteration_key, item_id, item_text, final_prompt_template, run_key, runs, runs_file, lock, judge_model, judge_index, save_raw_judge_output)
                final_futures.append(fut)
            
            for future in concurrent.futures.as_completed(final_futures):
                future.result() # Propagate exceptions if any

    except Exception as e:
        logging.error(f"Major error in Book Club for item {item_id}: {e}", exc_info=True)
        with lock:
            item_storage["error"] = f"Book Club Error: {str(e)}"
            save_json_file(runs, runs_file)

def finalize_scores_and_compute_judgemark(runs: dict, run_key: str, samples_data: dict, scoring_min: float, scoring_max: float):
    """
    Compute metrics for both raw and calibrated scores, including stability tests,
    normalized components, and detailed distributions.
    
    Now also returns a final_judgemark_score for BOTH raw and calibrated statistics.
    """
    run_data = runs[run_key]
    results = run_data.get("results", {})

    # 1. Collect raw scores, compute calibration, store calibrated values
    raw_scores_by_model_all = defaultdict(list)
    raw_scores_by_model_by_iter = defaultdict(lambda: defaultdict(list))
    calibrated_scores_by_model_all = defaultdict(list)
    calibrated_scores_by_model_by_iter = defaultdict(lambda: defaultdict(list))
    lengths_by_model = {}

    # -- Collect raw scores
    for model_name, iteration_data in results.items():
        if not isinstance(iteration_data, dict):
            continue
        
        lengths = []
        for it_key, it_val in iteration_data.items():
            if it_key == "__model_stats__":
                continue
            if not isinstance(it_val, dict):
                continue
                
            for item_id, item_info in it_val.items():
                if (isinstance(item_info, dict) and 
                    "aggregated_score_raw" in item_info and
                    item_info["aggregated_score_raw"] is not None):
                    raw_score = item_info["aggregated_score_raw"]
                    
                    # Collect raw score globally
                    raw_scores_by_model_all[model_name].append(raw_score)
                    # Collect raw score by iteration
                    raw_scores_by_model_by_iter[model_name][it_key].append(raw_score)

                    # Track text length for analyzing
                    text = (samples_data.get(model_name, {})
                            .get("samples", {})
                            .get(it_key, {})
                            .get(item_id, ""))
                    lengths.append(len(text))
        
        if len(raw_scores_by_model_all[model_name]) > 0:
            lengths_by_model[model_name] = lengths

    # 2. Distribution + calibration
    all_raw_scores = [s for scores in raw_scores_by_model_all.values() for s in scores]
    if not all_raw_scores:
        logging.error("No raw scores found to finalize. Aborting.")
        return
        
    run_data["raw_score_distribution"] = compute_detailed_distribution(all_raw_scores)

    calibration_config = build_landmark_calibration_config(all_raw_scores, [0, 3, 5, 7, 10])
    run_data["calibration_config"] = calibration_config

    # Apply calibration
    for model_name, iteration_data in results.items():
        if not isinstance(iteration_data, dict):
            continue
        
        # Flatten model's raw scores, calibrate them
        raw_list = raw_scores_by_model_all[model_name]
        calibrated = [apply_landmark_calibration(s, calibration_config) for s in raw_list]
        
        # Re-walk iteration_data to assign each calibration back
        idx = 0
        for it_key, it_val in iteration_data.items():
            if it_key == "__model_stats__":
                continue
            if not isinstance(it_val, dict):
                continue
            for item_id, item_info in it_val.items():
                if (isinstance(item_info, dict) and 
                    "aggregated_score_raw" in item_info and
                    item_info["aggregated_score_raw"] is not None):
                    item_info["aggregated_score_calibrated"] = calibrated[idx]
                    idx += 1
        
        # Update calibrated_scores_by_model_by_iter in the same breakdown
        idx2 = 0
        for it_key in raw_scores_by_model_by_iter[model_name]:
            count_for_iter = len(raw_scores_by_model_by_iter[model_name][it_key])
            these_cals = calibrated[idx2 : idx2 + count_for_iter]
            calibrated_scores_by_model_by_iter[model_name][it_key].extend(these_cals)
            idx2 += count_for_iter
        
        # Populate the single flattened list of calibrated scores
        calibrated_scores_by_model_all[model_name].extend(calibrated)

    # 3. Calibrated distributions
    all_calibrated_scores = [
        s for scores in calibrated_scores_by_model_all.values() for s in scores
    ]
    run_data["calibrated_score_distribution"] = compute_detailed_distribution(all_calibrated_scores)

    # 4. Model-level stats
    run_data["raw_model_stats"] = compute_model_level_stats(raw_scores_by_model_all, lengths_by_model)
    run_data["calibrated_model_stats"] = compute_model_level_stats(calibrated_scores_by_model_all, lengths_by_model)

    # Calculate length correlation stats averaged across the models
    raw_length_corrs = [stats["length_correlation"] for stats in run_data["raw_model_stats"].values() if "length_correlation" in stats]
    raw_corr_avg = statistics.mean(raw_length_corrs) if raw_length_corrs else 0.0
    raw_p_avg = statistics.mean([stats["length_correlation_p"] for stats in run_data["raw_model_stats"].values() if "length_correlation_p" in stats]) if raw_length_corrs else 0.0

    cal_length_corrs = [stats["length_correlation"] for stats in run_data["calibrated_model_stats"].values() if "length_correlation" in stats] 
    cal_corr_avg = statistics.mean(cal_length_corrs) if cal_length_corrs else 0.0
    cal_p_avg = statistics.mean([stats["length_correlation_p"] for stats in run_data["calibrated_model_stats"].values() if "length_correlation_p" in stats]) if cal_length_corrs else 0.0

    run_data["length_correlation"] = {
        "raw": {
            "pearson_corr": raw_corr_avg,
            "pearson_p": raw_p_avg
        },
        "calibrated": {
            "pearson_corr": cal_corr_avg,
            "pearson_p": cal_p_avg
        }
    }
    logging.info("--LENGTH CORRELATION--")
    logging.info(run_data["length_correlation"])

    # 5. Cross-model stats
    run_data["raw_cross_model_stats"] = compute_cross_model_stats(
        scores_by_model_all=raw_scores_by_model_all,
        scores_by_model_by_iter=raw_scores_by_model_by_iter
    )
    run_data["calibrated_cross_model_stats"] = compute_cross_model_stats(
        scores_by_model_all=calibrated_scores_by_model_all,
        scores_by_model_by_iter=calibrated_scores_by_model_by_iter
    )

    # 6. Separability metrics
    compute_separability_metrics(run_data, raw_scores_by_model_all, label="raw")
    compute_separability_metrics(run_data, calibrated_scores_by_model_all, label="calibrated")

    # 8. Compute iteration stability for raw & calibrated
    compute_iteration_stability(run_data, label="raw")  
    compute_iteration_stability(run_data, label="calibrated")
    random_tau_raw = compute_randomized_iteration_rank_stability_by_item(run_data, label="raw", n_shuffles=1000)
    random_tau_cal = compute_randomized_iteration_rank_stability_by_item(run_data, label="calibrated", n_shuffles=1000)
    logging.info("Score stability (RAW)")
    logging.info(f"Randomized average Kendall's tau (raw): {random_tau_raw:.3f}")
    logging.info("Score stability (CALIBRATED)") 
    logging.info(f"Randomized average Kendall's tau (calibrated): {random_tau_cal:.3f} "
                 f"({run_data['calibrated_cross_model_stats']['kendall_tau']})")

    # 9. Compute the final Judgemark scores (one using raw stats, one using calibrated)

    # -- (A) RAW Judgemark
    raw_stats = run_data["raw_cross_model_stats"]
    raw_norm = raw_stats["normalized_components"]
    
    raw_emd = run_data["separability_metrics"]["raw"]["emd"]["average"]
    raw_emd_norm = normalize(raw_emd, 0, 4)
    raw_overlap_mag = run_data["separability_metrics"]["raw"]["ci99_overlap_magnitude_sum"]
    raw_overlap_mag_norm = normalize(raw_overlap_mag, 0, 26, False)
    cohens_d_norm_raw = run_data["separability_metrics"]["raw"]["cohens_d_norm"]
    raw_overlap_mag_norm = modulate_x_by_y(raw_overlap_mag_norm, cohens_d_norm_raw)

    raw_norm["ci99_overlap_magnitude_sum_norm"] = raw_overlap_mag_norm
    raw_norm["ci99_overlap_magnitude_pct_norm"] = normalize(run_data["separability_metrics"]["raw"]["ci99_overlap_percentage_adjacent_avg"], 0.4, 0.85, False)

    raw_score_range = (
        max(run_data["raw_model_stats"][model]["mean"] for model in run_data["raw_model_stats"])
        - min(run_data["raw_model_stats"][model]["mean"] for model in run_data["raw_model_stats"])
    )
    run_data["raw_score_range"] = raw_score_range
    raw_score_range_norm = normalize(raw_score_range, 0, 10)
    raw_norm["raw_score_range_norm"] = raw_score_range_norm

    raw_norm["kendall_tau_bootstrapped"] = normalize(random_tau_raw, 0.7, 1.0)

    raw_separability = (
        #raw_norm["std_dev"]
        + raw_norm["kw_stat"]
        + raw_norm["ci99_overlap_magnitude_pct_norm"]
        #+ raw_norm["raw_score_range_norm"]
        #+ run_data["separability_metrics"]["raw"]["modulated_ci95"]
        #+ raw_emd_norm
    ) / 2.0

    final_score_raw = (
        raw_norm["kendall_tau_bootstrapped"] # stability
        + raw_norm["kendall_tau"] # arena correlation
        + 4 * raw_separability
    ) / 6.0
    run_data["final_judgemark_score_elements_raw"] = {
        "norm_stability_between_iterations": raw_norm["kendall_tau_bootstrapped"],
        "norm_correlation_with_lmsys_arena": raw_norm["kendall_tau"],
        "norm_std_dev_between_models": raw_norm["std_dev"],
        "norm_kruskall_wallis": raw_norm["kw_stat"],
        "norm_ci99_adjacent_overlap": raw_norm["ci99_overlap_magnitude_pct_norm"],
        "norm_score_range": raw_norm["raw_score_range_norm"],
        "norm_intra_model_ci95": run_data["separability_metrics"]["raw"]["modulated_ci95"],
        "norm_earth_movers_distance": raw_emd_norm
    }
    run_data["final_judgemark_score_raw"] = final_score_raw

    # -- (B) Calibrated Judgemark
    cal_stats = run_data["calibrated_cross_model_stats"]
    norm = cal_stats["normalized_components"]

    emd_norm = normalize(run_data["separability_metrics"]["calibrated"]["emd"]["average"], 0, 4)
    overlap_magnitude_norm = normalize(
        run_data["separability_metrics"]["calibrated"]["ci99_overlap_magnitude_sum"], 0, 26, False
    )
    cohens_d_norm_calibrated = run_data["separability_metrics"]["calibrated"]["cohens_d_norm"]
    overlap_magnitude_norm = modulate_x_by_y(overlap_magnitude_norm, cohens_d_norm_calibrated)
    norm["ci99_overlap_magnitude_sum_norm"] = overlap_magnitude_norm
    norm["ci99_overlap_magnitude_pct_norm"] = normalize(run_data["separability_metrics"]["calibrated"]["ci99_overlap_percentage_adjacent_avg"], 0.4, 0.85, False)

    calibrated_score_range = (
        max(run_data["calibrated_model_stats"][model]["mean"]
            for model in run_data["calibrated_model_stats"])
        - min(run_data["calibrated_model_stats"][model]["mean"]
              for model in run_data["calibrated_model_stats"])
    )
    run_data["calibrated_score_range"] = calibrated_score_range
    calibrated_score_range_norm = normalize(calibrated_score_range, 0, 10)
    norm["calibrated_score_range_norm"] = calibrated_score_range_norm

    norm["kendall_tau_bootstrapped"] = normalize(random_tau_cal, 0.6, 1.0)

    calibrated_separability = (
        #norm["std_dev"]
        + norm["kw_stat"]
        + norm["ci99_overlap_magnitude_pct_norm"]
        #+ norm["calibrated_score_range_norm"]
        #+ run_data["separability_metrics"]["calibrated"]["modulated_ci95"]
        #+ emd_norm
    ) / 2.0

    final_score_calibrated = (
        norm["kendall_tau_bootstrapped"]
        + norm["kendall_tau"]
        + 4 * calibrated_separability
    ) / 6.0
    run_data["final_judgemark_score_elements_calibrated"] = {
        "norm_stability_between_iterations": norm["kendall_tau_bootstrapped"],
        "norm_correlation_with_lmsys_arena": norm["kendall_tau"],
        "norm_std_dev_between_models": norm["std_dev"],
        "norm_kruskall_wallis": norm["kw_stat"],
        "norm_ci99_adjacent_overlap": norm["ci99_overlap_magnitude_pct_norm"],
        "norm_score_range": norm["calibrated_score_range_norm"],
        "norm_intra_model_ci95": run_data["separability_metrics"]["calibrated"]["modulated_ci95"],
        "norm_earth_movers_distance": emd_norm
    }
    run_data["final_judgemark_score"] = final_score_calibrated

    # 10. Create visualizations + logs
    judge_model_name = run_data.get("judge_model_name_for_charts", "Ensemble")
    create_side_by_side_score_charts(run_data, judge_model_name, samples_data, scoring_min, scoring_max)
    
    log_score_summary(
        "RAW SCORES", 
        run_data["raw_cross_model_stats"], 
        run_data["raw_model_stats"]
    )
    log_score_summary(
        "CALIBRATED SCORES", 
        run_data["calibrated_cross_model_stats"],
        run_data["calibrated_model_stats"]
    )

    logging.info(f"Final Judgemark (raw)   = {final_score_raw:.3f}")
    logging.info(f"Final Judgemark (cal)  = {final_score_calibrated:.3f}")


def sanitize_model_name(name: str) -> str:
    """Sanitize judge model name for use in the run key."""
    return re.sub(r'[^a-zA-Z0-9_-]+', '_', name)

def sanitize_ensemble_name(model_list: List[str]) -> str:
    """Creates a sanitized, sorted name for an ensemble of models."""
    sanitized_names = [sanitize_model_name(m) for m in model_list]
    return "ensemble_" + "_".join(sorted(sanitized_names))

def run_judgemark_v2(
    judge_models: str,
    samples_file: str,
    prompts_file: str,
    runs_file: str,
    num_threads: int,
    run_id: str = None,
    save_raw_judge_output: bool = False,
    scoring_min: float = 0,
    scoring_max: float = 10,
    ensemble_method: str = 'average',
    book_club_mode: bool = False
) -> str:
    global executor, should_exit
    
    judge_model_list = [m.strip() for m in judge_models.split(',') if m.strip()]
    logging.info(f"Starting Judgemark-v2 using judge models: {judge_model_list}")
    if book_club_mode:
        logging.info("Book Club mode ENABLED.")
        if len(judge_model_list) < 2:
            raise ValueError("Book Club mode requires at least 2 judge models.")

    runs = load_json_file(runs_file)
    
    if len(judge_model_list) > 1:
        sanitized_jm = sanitize_ensemble_name(judge_model_list)
        judge_model_name_for_charts = f"Ensemble ({len(judge_model_list)} judges, {ensemble_method})"
    else:
        sanitized_jm = sanitize_model_name(judge_model_list[0])
        judge_model_name_for_charts = judge_model_list[0]

    base_id = run_id if run_id else str(uuid.uuid4())
    run_key = f"{base_id}__{sanitized_jm}"
    if book_club_mode:
        run_key += "__bookclub"
    
    samples_data = load_json_file(samples_file)
    judge_prompts = load_json_file(prompts_file)
    writing_prompts = load_json_file("data/writing_prompts_v3.json")
    
    rubric_criteria = load_text_file("data/rubric_criteria.txt")
    score_anchoring = load_text_file("data/rubric_score_anchoring.txt")

    for key, prompt in judge_prompts.items():
        if isinstance(prompt, str):
            prompt = prompt.replace("<RUBRIC_CRITERIA>", rubric_criteria)
            prompt = prompt.replace("<SCORE_ANCHORING>", score_anchoring)
            if scoring_min != 0 or scoring_max != 10:
                prompt = prompt.replace("0-10 scale", f"{scoring_min}-{scoring_max} scale")
                prompt = prompt.replace("Score 0-10", f"Score {scoring_min}-{scoring_max}")
                prompt = prompt.replace("0 or 10", f"{scoring_min} or {scoring_max}")
            judge_prompts[key] = prompt

    if run_key not in runs:
        runs[run_key] = {
            "judge_models": judge_model_list,
            "judge_model_name_for_charts": judge_model_name_for_charts,
            "ensemble_method": ensemble_method,
            "book_club_mode": book_club_mode,
            "start_time": datetime.now().isoformat(),
            "status": "running",
            "samples_file": samples_file,
            "prompts_file": prompts_file,
            "results": {},
            "scoring_range": {"min": scoring_min, "max": scoring_max}
        }
        save_json_file(runs, runs_file)
    
    run_data = runs[run_key]
    run_data.setdefault("scoring_range", {"min": scoring_min, "max": scoring_max})
    
    items_to_process = []
    
    for model_name, model_info in samples_data.items():
        samples_dict = model_info.get("samples", {})
        for iteration_key, iteration_items in samples_dict.items():
            for item_id, item_text in iteration_items.items():
                if item_id not in judge_prompts:
                    raise(Exception("Item ID " + str(item_id) + " not found in judge prompts file."))
                
                items_to_process.append({
                    "model_name": model_name,
                    "iteration_key": iteration_key,
                    "item_id": item_id,
                    "item_text": item_text,
                    "prompt_template": judge_prompts[item_id],
                    "writing_prompt": writing_prompts.get(item_id, "")
                })


    lock = threading.Lock()
    all_futures = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as exec_:
            executor = exec_
            
            if items_to_process:
                for item in items_to_process:
                    if should_exit: break
                    
                    item_storage = run_data["results"].setdefault(item["model_name"], {}).setdefault(item["iteration_key"], {}).setdefault(item["item_id"], {})

                    if book_club_mode:
                        if "aggregated_score_raw" not in item_storage:
                            all_futures.append(
                                executor.submit(
                                    process_sample_book_club,
                                    item["model_name"], item["iteration_key"], item["item_id"], item["item_text"],
                                    item["prompt_template"], item["writing_prompt"],          # <-- NEW ARG
                                    run_key, runs, runs_file, lock, judge_model_list,
                                    save_raw_judge_output, rubric_criteria, score_anchoring
                                )
                            )
                    else: # Standard ensemble
                        outputs = item_storage.get("judge_outputs", [])
                        num_expected = len(judge_model_list)
                        
                        # Ensure list is correct size for resuming
                        if len(outputs) < num_expected:
                            outputs.extend([None] * (num_expected - len(outputs)))
                        item_storage["judge_outputs"] = outputs # Save back if modified

                        for judge_index, judge_model in enumerate(judge_model_list):
                            # Process if the slot is empty or contains an error from a previous attempt
                            if outputs[judge_index] is None or "error" in outputs[judge_index]:
                                all_futures.append(executor.submit(
                                    process_sample,
                                    item["model_name"], item["iteration_key"], item["item_id"], item["item_text"],
                                    item["prompt_template"], run_key, runs, runs_file, lock, judge_model,
                                    judge_index, save_raw_judge_output
                                ))
                
                if not all_futures:
                    logging.info(f"All items seem complete for run {run_key}. Finalizing scores.")
                else:
                    pbar_desc = "Book Club Judging" if book_club_mode else "Ensemble Judging"
                    for f in tqdm(concurrent.futures.as_completed(all_futures), total=len(all_futures), desc=pbar_desc):
                        if should_exit: break
                        try:
                            f.result()
                        except Exception as exc:
                            logging.error(f"Exception in worker thread: {exc}", exc_info=True)
    
    except KeyboardInterrupt:
        logging.warning("KeyboardInterrupt caught in main thread.")
        should_exit = True
        time.sleep(0.1)
    finally:
        status = "interrupted" if should_exit else "completed"
        runs[run_key]["status"] = status
        runs[run_key]["end_time"] = datetime.now().isoformat()
        
        if not should_exit:
            finalize_scores_and_compute_judgemark(runs, run_key, samples_data, scoring_min, scoring_max)

        save_json_file(runs, runs_file)
        
        if executor:
            logging.info("Shutting down executor")
            executor.shutdown(wait=False)
            executor = None
    
    logging.info(f"Judgemark-v2 run {run_key} ended with status: {status}")
    return run_key