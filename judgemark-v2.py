import sys
import signal
import argparse
import logging
import time


from utils.logging_setup import setup_logging, get_verbosity
from utils.file_io import load_json_file
from core.benchmark import run_judgemark_v2
from utils.api import API_KEY
from utils.state import should_exit, executor


def signal_handler(signum, frame):
    """Handle interrupt signals (SIGINT, SIGTERM)."""
    global executor, should_exit
    print(f"\n[DEBUG] Signal {signum} caught!")
    logging.warning("Signal handler called")
    should_exit = True
    time.sleep(0.1)  # Give workers a moment to see the flag
    if executor:
        logging.info("Shutting down executor from signal handler")
        executor.shutdown(wait=False)
        logging.info("Executor shutdown complete")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description='Run Judgemark-v2 Benchmark')
    parser.add_argument(
        '--judge-models',
        required=True,
        help='Comma-separated list of judge model identifiers (e.g., "openai/gpt-4,google/gemini-1.5-pro-latest")'
    )
    parser.add_argument(
        '--samples-file',
        default="data/judgemark_v3_samples_3_iter.json",
        help='JSON file containing pre-generated samples from various writer models'
    )
    parser.add_argument(
        '--prompts-file',
        default="data/judge_prompts_v3_noref_nocot_x96.json",
        help='JSON file containing the partial judge prompts to be filled with test responses'
    )
    parser.add_argument(
        '--runs-file',
        default="judgemark_v2_runs.json",
        help='Path to store the Judgemark run results'
    )
    parser.add_argument(
        '--run-id',
        help='Resume (or create) a run using this base ID, to be combined with the judge model name'
    )
    parser.add_argument(
        '--threads',
        type=int,
        default=6,
        help='Number of threads to use'
    )
    parser.add_argument(
        '--verbosity',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Set the logging verbosity level'
    )
    parser.add_argument(
        '--num-runs',
        type=int,
        default=1,
        help='Number of benchmark runs to execute'
    )
    parser.add_argument(
        '--save-raw-judge-output',
        action='store_true',
        default=False,
        help='If set, store the raw judge model output in the results JSON (default: false)'
    )
    # --- NEW: scoring range arguments ---
    parser.add_argument(
        '--scoring-min',
        type=float,
        default=0,
        help='Minimum scoring value (default: 0)'
    )
    parser.add_argument(
        '--scoring-max',
        type=float,
        default=10,
        help='Maximum scoring value (default: 10)'
    )
    parser.add_argument(
        '--ensemble-method',
        choices=['average', 'median'],
        default='average',
        help='Method to aggregate scores from an ensemble of judges (default: average).'
    )
    parser.add_argument(
        '--book-club',
        action='store_true',
        help='Enable multi-turn "Book Club" conversational judging. Requires 2+ judge models.'
    )
    return parser.parse_args()

if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Reset sentinel
    should_exit = False
    
    # Parse args
    args = parse_args()
    
    # Setup logging
    verbosity = get_verbosity(args.verbosity)
    setup_logging(verbosity)
    logging.debug("Logging initialized")
    
    # Check that we have an API key for the judge model
    if not API_KEY:
        logging.critical("No OPENAI_API_KEY found in environment variables.")
        raise ValueError("OPENAI_API_KEY not found in environment variables.")
    
    run_ids = []
    for i in range(1, args.num_runs + 1):
        if should_exit:
            break
        logging.info(f"Starting Judgemark-v2 run {i} of {args.num_runs}")
        rid = run_judgemark_v2(
            judge_models=args.judge_models,
            samples_file=args.samples_file,
            prompts_file=args.prompts_file,
            runs_file=args.runs_file,
            num_threads=args.threads,
            run_id=args.run_id,
            save_raw_judge_output=args.save_raw_judge_output,
            scoring_min=args.scoring_min,
            scoring_max=args.scoring_max,
            ensemble_method=args.ensemble_method,
            book_club_mode=args.book_club
        )
        run_ids.append(rid)
    
    # Finally, print summary
    runs = load_json_file(args.runs_file)
    logging.info("\nAll Judgemark-v2 runs completed:")
    print("\nAll Judgemark-v2 runs completed:")
    for rid in run_ids:
        rd = runs.get(rid, {})
        final_score = rd.get("final_judgemark_score", "N/A")
        logging.info(f"Run ID: {rid}, Final Judgemark Score: {final_score}")
        print(f"Run ID: {rid}")
        print(f"Final Judgemark-v2 Score: {final_score}")