# Judgemark V2.1

**Judgemark V2.1** is a benchmark that evaluates how well a language model can judge creative writing. Instead of relying on pairwise preferences, Judgemark V2 prompts the judge model to assign numeric scores for multiple literary criteria (e.g., “Nuanced Characters,” “Overwrought,” “Emotionally Engaging”). It then aggregates those scores, measures how consistent and discriminative they are, and derives a final numeric rating of the judge model’s performance.

The Judgemark leaderboard can be found here: [https://eqbench.com/judgemark-v2.html](https://eqbench.com/judgemark-v2.html)

# V2.1 Updates

To address the issue of the benchmark saturating, a new set of models was chosen as the source of creative writing outputs that the judge is assessing. This selection of *writer models* was selected from the **top ability range of creative writing leaderboard**. The net effect is that v2.1 more strongly assesses the judge's ability to be discriminative of higher quality writing, and to separate the differences between strong writers. This has added more headroom to the test, as the task is now harder.

We also include the ability to ensemble judge models. A model can ensemble with itself, or with other models, with final scoring done by averaging across the ensemble's votes for a given test item. We see a clear uplift in performance from ensembling judges.

Also added in this update is a "book club" mode. In this mode, several rounds of debate occur between the ensemble on the merits & failings of the piece of writing, followed by a discussion on scoring. Finally, each judge gives their own scores, with the entire book club conversation in context. In practice this typically harms the final score significantly -- judges typically are most performant with the fewest context distractors.

Book club is interesting because it assesses two things strongly: 1. How well the models perform on the judging task, given a very long, potentially distracting context, and 2. how well the judges can *productively reason* about the creative writing judging task. These two things are, typically very hard for language models, so book club represents a ramp-up in difficulty.

To note: running book club mode requires a long context window, and is quite expensive to run, especially with larger ensembles. It's off by default.


## Key Features

- **Complex Numeric Scoring**: Requires the judge model to provide 0–10 scores for dozens of criteria, highlighting any shortcomings in following complex instructions.
- **Raw & Calibrated Scores**: The system calculates a “raw” Judgemark score from the judge’s out-of-the-box distribution, and a “calibrated” score after normalizing the distribution for fairer cross-model comparisons.
- **Stability & Separability Metrics**: Goes beyond correlation to measure *how stable* the judge’s rankings are across repeated runs, and *how well* it separates strong from weak creative outputs.
- **Threaded Execution**: Supports multi-threaded item processing, drastically reducing the time required to score multiple creative samples.


## Setup & Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/EQ-bench/Judgemark-v2.git
   cd Judgemark-v2
   ```

2. **Install Python dependencies** (make sure you’re on Python 3.9+):

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables** to include your judge model’s API credentials. For example, if you’re using OpenAI-compatible endpoints:

   ```bash
   # (in .env or system env)
   export OPENAI_API_KEY="sk-..."
   export OPENAI_API_URL="https://openrouter.ai/api/v1/chat/completions"
   ```

## Usage

Run the benchmark via the main script `judgemark_v2.py`. For instance:

```bash
python judgemark_v2.py \
  --judge-models "openai/gpt-4o-mini" \
  --runs-file my_judgemark_runs.json \
  --threads 200 \
  --num-runs 1 \
  --save-raw-judge-output
```

### Command-Line Options

- **`--judge-models`** (required): A comma-delimited list of model ids that will be assessed as an ensemble (e.g. `openai/gpt-4o,anthropic/claude-sonnet-4`). A single judge is fine, and multiple of the same judge can be stacked.
- **`--samples-file`**: Path to the JSON with creative-writing samples to be judged. Default: `data/judgemark_v3_samples_3_iter.json`.
- **`--prompts-file`**: Path to the JSON with partial prompts for the judge. Default: `data/judge_prompts_v3_noref_nocot_x96.json`.
- **`--runs-file`**: The output JSON to store final run results. Default: `judgemark_v2_runs.json`.
- **`--run-id`**: A custom run ID for continuing or naming a run (optional).
- **`--threads`**: Number of threads for parallel scoring. Default: `6`.
- **`--verbosity`**: Log verbosity: one of `[DEBUG, INFO, WARNING, ERROR, CRITICAL]`.
- **`--num-runs`**: Number of times to repeat the entire benchmark. Default: `1`.
- **`--save-raw-judge-output`**: Store the raw text responses from the judge into the results JSON.
- **`--book-club`**: This enables "book club mode" where the judge ensemble debates the merits of the piece, as well as the scoring decisions, before scoring occurs. Warning: this requires high context length support, and it's expensive to run!

## How It Works

1. **Reading In Samples**  
   The script loads `samples_file`, which contains completions to creative writing prompts from multiple “writer models.”

2. **Generating Judge Prompts**  
   For each completion, we load a judge prompt from `prompts_file`. This typically includes instructions like:
   ```
   Please assign numeric scores (0-10) for these criteria:
   - Nuanced Characters
   - Overwrought
   - ...
   [TEST MODEL RESPONSE]
   ...
   ```

3. **Sending Requests to the Judge Model**  
   Each completion + prompt is sent to the `--judge-model` via the functions in `utils/api.py`. We specify a moderate temperature (often `0.5`) and top-k for variability.

4. **Parsing the Judge Output**  
   The script captures lines like `Nuanced Characters: 8` or `Weak Dialogue: 3`, extracts the numeric scores, and aggregates them into a single raw score. Negative criteria (like “Weak Dialogue”) are inverted so 10 = worst.

5. **Storing & Re-Trying**  
   Results are saved in your designated `runs-file`. If an item fails or provides incomplete scores, the script can retry (in subsequent runs) without overwriting previous data.

6. **Final Judgemark Scores**  
   Once all samples are scored:
   - A *raw* Judgemark score is computed from the distribution of assigned scores.  
   - A *calibrated* score is computed after normalizing each judge’s “score spread” to a standard distribution anchored to the mean, 25th & 75th percentile, upper & lower range. Calibration linearly transforms the distribution from these anchor points to match an ideal distribution of 0-10 range, 5 mean, and 25th & 75th percentile 
   - Additional metrics quantify how consistent (stable) and discriminative the judge is.

## Interpreting the Results

The output JSON in your `--runs-file` will contain many details, including per-model breakdowns, iteration-level stats, and final composite scores:

- **`final_judgemark_score`**: The primary benchmark result (based on calibrated distribution). A higher value suggests better correlation with reference preferences, stronger separation between good and weak writing, and higher consistency.
- **`final_judgemark_score_raw`**: A non-calibrated version that shows how well the judge performs “out of the box.”
- **Per-model details**: Found under `results[MODEL_NAME]`, including each snippet’s aggregated raw score and partial criterion scores.

You can also enable **visualization**: the code in `utils/visualization.py` produces bar charts, heatmaps, and scatter plots illustrating how the judge assigned scores across models.

## Contributing

Contributions and bug reports are welcome! If you’d like to add new features—such as custom scoring criteria, improved calibration, or alternative reference sets—feel free to open a PR or file an issue.

## License

This project is licensed under an [MIT License](LICENSE). See the `LICENSE` file for more details.

## Acknowledgments

- **LMSys Chatbot Arena** -- the source for the rankings used in the benchmark for human preference correlation.

---

**Happy Judging!** If you have any questions, reach out via [GitHub Issues](https://github.com/EQ-bench/judgemark-v2/issues) or contact the maintainers.