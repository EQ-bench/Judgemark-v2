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

from utils.file_io import load_json_file, save_json_file
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

def process_sample(model_name: str, iteration_key: str, item_id: str, item_text: str, 
                  prompt_template: str, run_key: str, runs: Dict, runs_file: str,
                  lock: threading.Lock, judge_model: str, save_raw_judge_output: bool):
    """Process a single sample, retrying failed or empty results."""
    global should_exit
    if should_exit:
        return
    
    text_len = len(item_text)
    run_data = runs.get(run_key, {})
    results = run_data.get("results", {})
    model_dict = results.setdefault(model_name, {})
    iteration_dict = model_dict.setdefault(iteration_key, {})
    
    existing_item = iteration_dict.get(item_id, {})
    if (existing_item and 
        "aggregated_score_raw" in existing_item and 
        existing_item.get("parsed_scores") and 
        len(existing_item["parsed_scores"]) >= 10 and
        existing_item["aggregated_score_raw"] > 0.0):
        return
    
    try:
        final_prompt = prompt_template.replace("[TEST MODEL RESPONSE]", item_text)
        final_prompt = final_prompt.replace("[TEST MODEL RESPONSE END]", "")
        
        messages = [{"role": "user", "content": final_prompt}]
        judge_response = send_to_judge_model(messages, judge_model=judge_model)
        
        extracted_scores = parse_scores(judge_response)
        # Retrieve scoring range from run_data (defaults to 0–10 if not set)
        scoring_range = run_data.get("scoring_range", {"min": 0, "max": 10})
        scoring_min = scoring_range.get("min", 0)
        scoring_max = scoring_range.get("max", 10)
        raw_score = compute_raw_score(extracted_scores, scoring_min, scoring_max)
        
        with lock:
            storage_dict = {
                "parsed_scores": extracted_scores,
                "timestamp": datetime.now().isoformat(),
                "text_length": text_len
            }
            if raw_score is not None:
                storage_dict["aggregated_score_raw"] = raw_score
            if save_raw_judge_output:
                storage_dict["judge_response"] = judge_response
            
            iteration_dict[item_id] = storage_dict
            runs[run_key]["results"][model_name][iteration_key] = iteration_dict
            save_json_file(runs, runs_file)
        
        if raw_score is not None:
            logging.debug(f"Processed {model_name}/{iteration_key}/{item_id}, raw score: {raw_score:.2f}")
        else:
            print(judge_response)
            logging.warning(f"Failed to parse enough scores for {model_name}/{iteration_key}/{item_id}")
            
    except Exception as e:
        logging.error(f"Error processing item {model_name}/{iteration_key}/{item_id}: {str(e)}")
        with lock:
            iteration_dict[item_id] = {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            if "errors" not in runs[run_key]:
                runs[run_key]["errors"] = []
            runs[run_key]["errors"].append({
                "model": model_name,
                "iteration": iteration_key,
                "item_id": item_id,
                "error": str(e)
            })
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
                    "aggregated_score_raw" in item_info):
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
                    "aggregated_score_raw" in item_info):
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
    raw_length_corrs = [stats["length_correlation"] for stats in run_data["raw_model_stats"].values()]
    raw_corr_avg, raw_p_avg = statistics.mean(raw_length_corrs), statistics.mean([stats["length_correlation_p"] for stats in run_data["raw_model_stats"].values()])

    cal_length_corrs = [stats["length_correlation"] for stats in run_data["calibrated_model_stats"].values()] 
    cal_corr_avg, cal_p_avg = statistics.mean(cal_length_corrs), statistics.mean([stats["length_correlation_p"] for stats in run_data["calibrated_model_stats"].values()])

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
    raw_norm["ci99_overlap_magnitude_pct_norm"] = normalize(run_data["separability_metrics"]["raw"]["ci99_overlap_percentage_adjacent_avg"], 0, 1, False)

    raw_score_range = (
        max(run_data["raw_model_stats"][model]["mean"] for model in run_data["raw_model_stats"])
        - min(run_data["raw_model_stats"][model]["mean"] for model in run_data["raw_model_stats"])
    )
    run_data["raw_score_range"] = raw_score_range
    raw_score_range_norm = normalize(raw_score_range, 0, 10)
    raw_norm["raw_score_range_norm"] = raw_score_range_norm

    raw_norm["kendall_tau_bootstrapped"] = normalize(random_tau_raw, 0.4, 1.0)

    raw_separability = (
        + raw_norm["kw_stat"]
        + raw_norm["ci99_overlap_magnitude_pct_norm"]
    ) / 2.0

    final_score_raw = (
        raw_norm["kendall_tau_bootstrapped"]
        + raw_norm["kendall_tau"]
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
    norm["ci99_overlap_magnitude_pct_norm"] = normalize(run_data["separability_metrics"]["calibrated"]["ci99_overlap_percentage_adjacent_avg"], 0, 1, False)

    calibrated_score_range = (
        max(run_data["calibrated_model_stats"][model]["mean"]
            for model in run_data["calibrated_model_stats"])
        - min(run_data["calibrated_model_stats"][model]["mean"]
              for model in run_data["calibrated_model_stats"])
    )
    run_data["calibrated_score_range"] = calibrated_score_range
    calibrated_score_range_norm = normalize(calibrated_score_range, 0, 10)
    norm["calibrated_score_range_norm"] = calibrated_score_range_norm

    norm["kendall_tau_bootstrapped"] = normalize(random_tau_cal, 0.4, 1.0)

    calibrated_separability = (
        + norm["kw_stat"]
        + norm["ci99_overlap_magnitude_pct_norm"]
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
    create_side_by_side_score_charts(run_data, run_data["judge_model"], samples_data, scoring_min, scoring_max)
    
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

def run_judgemark_v2(
    judge_model: str,
    samples_file: str,
    prompts_file: str,
    runs_file: str,
    num_threads: int,
    run_id: str = None,
    save_raw_judge_output: bool = False,
    scoring_min: float = 0,
    scoring_max: float = 10
) -> str:
    global executor, should_exit
    
    logging.info(f"Starting Judgemark-v2 using judge model: {judge_model}")
    runs = load_json_file(runs_file)
    
    # Form the run key using run_id + "__" + sanitized judge model
    sanitized_jm = sanitize_model_name(judge_model)
    base_id = run_id if run_id else str(uuid.uuid4())
    run_key = f"{base_id}__{sanitized_jm}"
    
    # Load data files
    samples_data = load_json_file(samples_file)
    judge_prompts = load_json_file(prompts_file)
    
    # --- NEW: Replace hard-coded scoring range strings in judge prompts ---
    if scoring_min != 0 or scoring_max != 10:
        for key, prompt in judge_prompts.items():
            if isinstance(prompt, str):
                prompt = prompt.replace("0-10 scale", f"{scoring_min}-{scoring_max} scale")
                prompt = prompt.replace("Score 0-10", f"Score {scoring_min}-{scoring_max}")
                prompt = prompt.replace("0 or 10", f"{scoring_min} or {scoring_max}")
                judge_prompts[key] = prompt

    # Initialize or get existing run data
    if run_key not in runs:
        runs[run_key] = {
            "judge_model": judge_model,
            "start_time": datetime.now().isoformat(),
            "status": "running",
            "samples_file": samples_file,
            "prompts_file": prompts_file,
            "results": {},
            "scoring_range": {"min": scoring_min, "max": scoring_max}
        }
        save_json_file(runs, runs_file)
    
    run_data = runs[run_key]
    # Ensure scoring_range exists in case of resumed runs.
    run_data.setdefault("scoring_range", {"min": scoring_min, "max": scoring_max})
    
    items_to_process = []
    
    # If run exists, scan for items needing retry
    if "results" in run_data:
        results = run_data.get("results", {})
        
        # Scan all possible items
        for model_name, model_info in samples_data.items():
            samples_dict = model_info.get("samples", {})
            for iteration_key, iteration_items in samples_dict.items():
                for item_id, item_text in iteration_items.items():
                    # Check if this item needs processing
                    existing_result = (results.get(model_name, {})
                                            .get(iteration_key, {})
                                            .get(item_id, {}))
                    
                    needs_retry = (
                        not existing_result or
                        not existing_result.get("parsed_scores") or
                        len(existing_result.get("parsed_scores", {})) < 10 or
                        existing_result.get("aggregated_score_raw", 0.0) == 0.0 or
                        "error" in existing_result
                    )
                    
                    if needs_retry:
                        if item_id not in judge_prompts:
                            raise(Exception("Item ID " + str(item_id) + " not found in judge prompts file."))
                        items_to_process.append({
                        "model_name": model_name,
                        "iteration_key": iteration_key,
                        "item_id": item_id,
                        "item_text": item_text,
                        "prompt_template": judge_prompts[item_id]
                    })
        
        if items_to_process:
            logging.info(f"Found {len(items_to_process)} items to process in existing run {run_key}")
        else:
            logging.info(f"No items to process in existing run {run_key}")

    else:
        # New run - process all items
        for model_name, model_info in samples_data.items():
            samples_dict = model_info.get("samples", {})
            for iteration_key, iteration_items in samples_dict.items():
                print(iteration_key)
                for item_id, item_text in iteration_items.items():
                    if item_id not in judge_prompts:
                        raise(Exception("Item ID " + str(item_id) + " not found in judge prompts file."))
                    items_to_process.append({
                        "model_name": model_name,
                        "iteration_key": iteration_key,
                        "item_id": item_id,
                        "item_text": item_text,
                        "prompt_template": judge_prompts[item_id]
                    })
    
    # Ensure concurrency lock
    lock = threading.Lock()
    
    # Process any items that need retrying
    all_futures = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as exec_:
            executor = exec_
            
            if items_to_process:
                # Process all items (either retries or new run)
                for item in items_to_process:
                    if should_exit:
                        break
                        
                    all_futures.append(
                        executor.submit(
                            process_sample,
                            item["model_name"],
                            item["iteration_key"],
                            item["item_id"],
                            item["item_text"],
                            item["prompt_template"],
                            run_key,
                            runs,
                            runs_file,
                            lock,
                            judge_model,
                            save_raw_judge_output
                        )
                    )
                
                # Display progress bar for tasks
                for f in tqdm(concurrent.futures.as_completed(all_futures), 
                              total=len(all_futures), desc="Judging", leave=True):
                    if should_exit:
                        break
                    try:
                        f.result()
                    except Exception as exc:
                        logging.error(f"Exception in worker thread: {exc}")
    
    except KeyboardInterrupt:
        logging.warning("KeyboardInterrupt caught in main thread.")
        should_exit = True
        time.sleep(0.1)
    finally:
        # Mark run as interrupted or completed
        status = "interrupted" if should_exit else "completed"
        runs[run_key]["status"] = status
        runs[run_key]["end_time"] = datetime.now().isoformat()
        
        if not should_exit:
            # Run stability test
            if False:
                run_stability_test(
                    run_data, judge_model,
                    judge_prompts, samples_data,
                    runs, runs_file,
                    lock, num_threads
                )
            # Compute final stats
            finalize_scores_and_compute_judgemark(runs, run_key, samples_data, scoring_min, scoring_max)

        # Save final
        save_json_file(runs, runs_file)
        
        if executor:
            logging.info("Shutting down executor")
            executor.shutdown(wait=False)
            executor = None
    
    logging.info(f"Judgemark-v2 run {run_key} ended with status: {status}")
    return run_key
