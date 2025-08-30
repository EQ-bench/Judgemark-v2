import logging
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from typing import Dict

from utils.api import send_to_judge_model
from utils.file_io import save_json_file
from core.scoring import parse_scores, compute_raw_score
from config.constants import STABILITY_ITEMS, STABILITY_REPS
from utils.state import should_exit, executor
from collections import defaultdict
import statistics
import math
import scipy.stats
import random
import statistics

def extract_model_item_scores(run_data: dict, label: str = "raw"):
    """
    Collect a dictionary:
      model_item_scores[model][item_id] = { iteration_key: score }
    for either aggregated_score_raw or aggregated_score_calibrated (controlled by 'label').

    Also returns a sorted list of iteration_keys found in the data.
    We assume (model, item_id) either has exactly one score per iteration or is skipped.
    """
    results = run_data.get("results", {})
    model_item_scores = defaultdict(lambda: defaultdict(dict))
    all_iteration_keys = set()

    for model_name, model_dict in results.items():
        if not isinstance(model_dict, dict):
            continue
        for it_key, item_dict in model_dict.items():
            if not isinstance(item_dict, dict):
                continue
            all_iteration_keys.add(it_key)
            for item_id, item_info in item_dict.items():
                if not isinstance(item_info, dict):
                    continue
                if label == "raw":
                    score_val = item_info.get("aggregated_score_raw", None)
                else:
                    score_val = item_info.get("aggregated_score_calibrated", None)
                
                if isinstance(score_val, (int, float)):
                    model_item_scores[model_name][item_id][it_key] = score_val
    
    sorted_iteration_keys = sorted(list(all_iteration_keys))
    return model_item_scores, sorted_iteration_keys


def shuffle_iteration_assignments_by_item(
    model_item_scores, 
    iteration_keys
):
    """
    For each model, for each item_id, we have a dict of {it_key: score}, one entry per iteration.
    We do a random permutation among iteration_keys so that:

      For k in range(len(iteration_keys)):
        * The score originally from iteration_keys[perm[k]] is assigned to iteration_keys[k].

    This ensures each iteration_key gets exactly one sample for each item_id,
    but “which original iteration’s score” is placed there is random.

    Returns: iteration_averages = { new_it_key: { model: [scores...] } }

    We'll later convert those lists to mean(...) for each model.
    """
    # structure to gather new assignment
    # iteration_averages[new_it_key][model] = list of assigned scores
    from collections import defaultdict
    iteration_averages = defaultdict(lambda: defaultdict(list))
    n_iters = len(iteration_keys)
    
    for model_name, item_dict in model_item_scores.items():
        for item_id, it_score_map in item_dict.items():
            # We expect one score for each iteration key
            # If item_id is missing an iteration, that item won't be shuffled properly.
            # We can skip or handle partial if needed. Here, assume it's complete:
            if len(it_score_map) < n_iters:
                # skip incomplete
                continue
            
            # Build an array of scores in sorted iteration_keys order
            # so scores[i] corresponds to iteration_keys[i] in the original
            scores_in_order = [it_score_map[k] for k in iteration_keys]
            
            # random permutation of [0..n_iters-1]
            permutation = random.sample(range(n_iters), k=n_iters)
            
            # Now assign scores_in_order[permutation[k]] to iteration_keys[k].
            for k in range(n_iters):
                new_it_key = iteration_keys[k]
                old_index = permutation[k]
                assigned_score = scores_in_order[old_index]
                iteration_averages[new_it_key][model_name].append(assigned_score)
    
    return iteration_averages


def compute_average_iteration_averages(iteration_averages):
    """
    iteration_averages is { new_it_key: { model_name: [scores...] }}
    Convert that to { new_it_key: { model_name: mean_score }}.
    """
    final_avg = {}
    for it_key, model_dict in iteration_averages.items():
        final_avg[it_key] = {}
        for m, score_list in model_dict.items():
            if score_list:
                final_avg[it_key][m] = statistics.mean(score_list)
            else:
                final_avg[it_key][m] = 0.0
    return final_avg


def compute_kendall_tau_for_iterations(iteration_averages):
    """
    iteration_averages: { it_key: { model_name: avg_score } }
    We'll build rank lists, then compare each pair of iteration_keys for Kendall’s τ.
    Returns average τ across iteration pairs, or 0.0 if not enough data.
    """
    all_iter_keys = sorted(iteration_averages.keys())
    
    # Build rank ordering
    iteration_rank = {}
    for it_key in all_iter_keys:
        items = list(iteration_averages[it_key].items())  # (model, avg_score)
        # sort descending by score
        items.sort(key=lambda x: x[1], reverse=True)
        iteration_rank[it_key] = [m for (m, sc) in items]
    
    pairwise_taus = []
    for i in range(len(all_iter_keys)):
        for j in range(i+1, len(all_iter_keys)):
            ik = all_iter_keys[i]
            jk = all_iter_keys[j]
            rank_i = iteration_rank[ik] 
            rank_j = iteration_rank[jk]
            common_models = set(rank_i).intersection(rank_j)
            if len(common_models) < 2:
                continue
            pos_i = {m: rank_i.index(m) for m in common_models}
            pos_j = {m: rank_j.index(m) for m in common_models}
            x = []
            y = []
            for m in common_models:
                x.append(pos_i[m])
                y.append(pos_j[m])
            if len(x) > 1:
                tau, pval = scipy.stats.kendalltau(x, y)
                if not math.isnan(tau):
                    pairwise_taus.append(tau)
    if pairwise_taus:
        return statistics.mean(pairwise_taus)
    else:
        return 0.0


def compute_randomized_iteration_rank_stability_by_item(
    run_data: dict,
    label: str = "raw",
    n_shuffles: int = 1000
) -> float:
    """
    For each item_id in each model, we shuffle which iteration key 
    receives that item’s original score (a random permutation among 
    the iteration_keys).
    
    This ensures that each iteration_key still has exactly one sample for each item
    (thus preserving the structure of “N iterations, M items each”), 
    but randomizes *which original iteration’s score* ended up in each iteration.
    
    Then we compute the average Kendall's τ across iteration pairs 
    (ranking stability). We repeat n_shuffles times, returning the 
    average of those τ values.
    
    We'll store the result in:
      run_data["iteration_stability"][label]["randomized_average_kendall_tau_by_item"].
    
    This approach is more efficient than reconstructing big data structures 
    or re-running a large function every time.
    """
    model_item_scores, iteration_keys = extract_model_item_scores(run_data, label=label)
    
    if len(iteration_keys) < 2:
        return 0.0  # no real iteration comparison
    
    tau_values = []
    for _ in range(n_shuffles):
        # Step 1: Randomly assign each item’s scores to iteration_keys
        iteration_averages_shuffled = shuffle_iteration_assignments_by_item(model_item_scores, iteration_keys)
        # Step 2: Convert from [list of scores] to mean
        final_avg = compute_average_iteration_averages(iteration_averages_shuffled)
        # Step 3: Compute rank correlation across iteration_keys
        tau = compute_kendall_tau_for_iterations(final_avg)
        tau_values.append(tau)
    
    if tau_values:
        randomized_avg_tau = statistics.mean(tau_values)
    else:
        randomized_avg_tau = 0.0
    
    # Store in run_data
    if "iteration_stability" not in run_data:
        run_data["iteration_stability"] = {}
    if label not in run_data["iteration_stability"]:
        run_data["iteration_stability"][label] = {}
    run_data["iteration_stability"][label]["randomized_average_kendall_tau_by_item"] = randomized_avg_tau
    
    return randomized_avg_tau


def compute_iteration_stability(run_data: dict, label="raw"):
    """
    Example function that:
      1) Gathers iteration-average scores for each model.
      2) Computes 'scoring stability' across iterations.
      3) Computes 'ranking stability' across iterations.
      4) Stores results in run_data["iteration_stability"][label].
      
    label could be "raw" or "calibrated"—adjust as needed.
    """
    
    # Make a place to store results:
    if "iteration_stability" not in run_data:
        run_data["iteration_stability"] = {}
    run_data["iteration_stability"][label] = {}
    
    results = run_data.get("results", {})
    
    # 1) Gather iteration_averages[iter_key][model] = average_score
    iteration_averages = defaultdict(dict)  # iter_key -> {model -> avg_score}
    
    for model_name, model_data in results.items():
        # model_data: iter_key -> item_dict
        if not isinstance(model_data, dict):
            continue
        for iter_key, item_dict in model_data.items():
            if not isinstance(item_dict, dict):
                continue
            
            # Collect all items' scores for (this model, iter_key)
            scores = []
            for item_id, item_info in item_dict.items():
                if not isinstance(item_info, dict):
                    continue
                if label == "raw":
                    val = item_info.get("aggregated_score_raw", None)
                else:
                    val = item_info.get("aggregated_score_calibrated", None)
                
                if isinstance(val, (int, float)):
                    scores.append(val)
            
            if scores:
                iteration_averages[iter_key].setdefault(model_name, [])
                iteration_averages[iter_key][model_name] = statistics.mean(scores)
    
    # 2) Scoring Stability
    # We can do something like: for each model, gather all iteration_averages and store stdev
    model_stability = {}
    
    # We also want to track how many times each model appears (some might not appear in all iter_keys)
    all_iter_keys = sorted(iteration_averages.keys())
    
    for model_name in results.keys():
        # gather the iteration means
        vals = []
        for it_key in all_iter_keys:
            if model_name in iteration_averages[it_key]:
                vals.append(iteration_averages[it_key][model_name])
        if len(vals) > 1:
            stdev_ = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            model_stability[model_name] = {
                "mean_iter_score": statistics.mean(vals),
                "iteration_count": len(vals),
                "stdev_across_iters": stdev_
            }
        else:
            model_stability[model_name] = {
                "mean_iter_score": vals[0] if vals else 0.0,
                "iteration_count": len(vals),
                "stdev_across_iters": 0.0
            }
    
    run_data["iteration_stability"][label]["scoring_stability"] = model_stability
    
    # 3) Ranking Stability
    # For each iteration, produce a list of (model, iteration_average)
    # Then create a rank ordering. We'll store them in a dictionary: iteration_rank[iter_key] = [model1, model2, ...]
    
    iteration_rank = {}
    for iter_key in all_iter_keys:
        # build a list of (model, avg_score) then sort descending
        items = list(iteration_averages[iter_key].items())
        items.sort(key=lambda x: x[1], reverse=True)
        # The rank order is just model names in sorted order:
        iteration_rank[iter_key] = [m for (m, sc) in items]
    
    # now for each pair (i, j) of iteration_keys, compute rank correlation
    # you can pick Kendall or Spearman. We'll do Kendall as example:
    
    pairwise_rank_corr = {}
    sorted_iter_keys = list(all_iter_keys)
    
    for i in range(len(sorted_iter_keys)):
        for j in range(i+1, len(sorted_iter_keys)):
            ik = sorted_iter_keys[i]
            jk = sorted_iter_keys[j]

            rank_i = iteration_rank[ik]  # not iteration_rank[i]
            rank_j = iteration_rank[jk]  # not iteration_rank[j]
            # build positions
            # If a model is missing in either iteration, skip it
            common_models = set(rank_i).intersection(rank_j)
            # create index mappings
            pos_i = {m: rank_i.index(m) for m in common_models}
            pos_j = {m: rank_j.index(m) for m in common_models}
            
            # now build x/y from pos_i, pos_j
            x = []
            y = []
            for m in common_models:
                x.append(pos_i[m])
                y.append(pos_j[m])
            
            if len(x) > 1:
                tau, pval = scipy.stats.kendalltau(x, y)
            else:
                tau, pval = (0.0, 1.0)
            
            key_name = f"{ik}__vs__{jk}"
            pairwise_rank_corr[key_name] = {
                "common_model_count": len(common_models),
                "kendall_tau": tau,
                "p_value": pval
            }
    
    run_data["iteration_stability"][label]["ranking_stability"] = {
        "pairwise_correlation": pairwise_rank_corr
    }
    
    # Optionally, you might compute an "average" or "median" rank correlation across all iteration pairs
    if pairwise_rank_corr:
        all_taus = [v["kendall_tau"] for v in pairwise_rank_corr.values() if not math.isnan(v["kendall_tau"])]
        run_data["iteration_stability"][label]["ranking_stability"]["average_kendall_tau"] = statistics.mean(all_taus) if all_taus else 0.0

def process_stability_test_item(model_name, iteration_key, item_id, item_text, prompt_template, judge_model, scoring_min, scoring_max) -> float:
    """Process a single judge request for the stability test. Returns the aggregated score or 0.0 if failed."""
    global should_exit
    if should_exit:
        return 0.0
    
    try:
        final_prompt = prompt_template.replace("[TEST MODEL RESPONSE]", item_text)
        final_prompt = final_prompt.replace("[TEST MODEL RESPONSE END]", "")
        
        messages = [{"role": "user", "content": final_prompt}]
        judge_response = send_to_judge_model(messages, judge_model=judge_model)
        
        extracted_scores = parse_scores(judge_response)
        item_score = compute_raw_score(extracted_scores, scoring_min, scoring_max)
        
        # Only return actual valid scores, never None
        return item_score if isinstance(item_score, (int, float)) and item_score > 0.0 else 0.0
    except Exception as e:
        logging.error(f"Error in stability test item {model_name}/{iteration_key}/{item_id}: {str(e)}")
        return 0.0

def run_stability_test(run_data, judge_model, judge_prompts, samples_data, runs, runs_file, lock, num_threads, scoring_min, scoring_max):
    """Run stability test, retrying any missing entries to reach STABILITY_REPS per item."""
    logging.info("Running stability test for selected items...")
    
    if "stability_test_results" not in run_data:
        run_data["stability_test_results"] = {}
    
    items_to_process = []
    for (model, iteration, item_id) in STABILITY_ITEMS:
        key_name = f"{model}-{iteration}-{item_id}"
        existing_results = run_data["stability_test_results"].get(key_name, [])
        
        # Filter out failed results (0.0 scores, None values) from existing
        valid_results = [score for score in existing_results 
                        if isinstance(score, (int, float)) and score > 0.0]
        needed_count = STABILITY_REPS - len(valid_results)
        
        if needed_count > 0:
            item_text = samples_data.get(model, {}).get("samples", {}).get(iteration, {}).get(item_id, "")
            prompt_template = judge_prompts.get(item_id, "")
            
            for _ in range(needed_count):
                items_to_process.append({
                    "model": model,
                    "iteration": iteration,
                    "item_id": item_id,
                    "item_text": item_text,
                    "prompt_template": prompt_template,
                    "key_name": key_name
                })
            
            logging.info(f"Need {needed_count} more stability test results for {key_name}")
            
            # Clean up existing results, keeping only valid scores
            run_data["stability_test_results"][key_name] = valid_results
    
    if not items_to_process:
        logging.info("All stability test items already have complete results")
        return
    
    with ThreadPoolExecutor(max_workers=num_threads) as exec_:
        futures_to_items = {}
        
        # Launch futures for all needed retries
        for item in items_to_process:
            if should_exit:
                break
            future = exec_.submit(
                process_stability_test_item,
                item["model"], item["iteration"], item["item_id"],
                item["item_text"], item["prompt_template"],
                judge_model, scoring_min, scoring_max
            )
            futures_to_items[future] = item
        
        # Collect results as they complete
        for future in tqdm(concurrent.futures.as_completed(futures_to_items.keys()),
                         total=len(futures_to_items),
                         desc="Stability Test"):
            if should_exit:
                break
            try:
                score = future.result()
                item = futures_to_items[future]
                key_name = item["key_name"]
                
                # Only store actual valid scores
                if isinstance(score, (int, float)) and score > 0.0:
                    with lock:
                        run_data["stability_test_results"][key_name].append(score)
                        save_json_file(runs, runs_file)
                else:
                    logging.warning(f"Got invalid score for stability item {key_name}, will need retry")
            except Exception as exc:
                logging.error(f"Exception in stability test: {exc}")