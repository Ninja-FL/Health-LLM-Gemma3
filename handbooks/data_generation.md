# Data Generation Pipeline

This doc summarizes how `gen_dataset.py` builds supervised instruction–response pairs for finetuning and evaluation. The focus is on PMData, which is the default path when you run the script.

## Entry point
- Script: `gen_dataset.py`
- Defaults at top of file: `MODE="train"` and `DATA="PMData"`.
- Run from repo root:  
  `python gen_dataset.py`

## Common flow (all datasets)
1. Set `DATA` (and `SUBTASK` where applicable).
2. Load raw sensor / survey files from the dataset-specific directory.
3. Compose Alpaca-style triplets: `instruction`, `input`, `output`.
4. Shuffle with seed 123 and split ~50/50 into train vs eval.
5. Serialize:
   - Train slices: `<DATA>_<subtask>_train_{3,10,25}.json`
   - Full train: `<DATA>_<subtask>_train_all.json`
   - Eval: `../eval/data/<data>_<subtask>/step1.json` (relative to the script).

## PMData pipeline (default)
- Root dir expected: `medalpaca/data/pmdata/<participant>/`.
- Files consumed:
  - `pmsys/wellness.csv` — daily self-report: `effective_time_frame`, `fatigue`, `mood`, `readiness`, `sleep_duration_h`, `sleep_quality`, `soreness`, `soreness_area`, `stress`.
  - `fitbit/exercise.json` — activity sessions (time, calories, steps, duration, activityName).
  - `fitbit/resting_heart_rate.json` — resting HR time series.
  - `fitbit/sleep.json` — sleep sessions (startTime, duration).

### Subtask selection
- In the PMData block, set `SUBTASK` to one of: `stress` (default), `readiness`, `sleep_quality`, `fatigue`.
- Label source:
  - `stress` → `stress` column (1–5)
  - `readiness` → `readiness` (0–10)
  - `sleep_quality` → `sleep_quality` (1–5)
  - `fatigue` → `fatigue` (1–5)

### Feature window
- For each wellness row (a timestamped self-report), gather matching Fitbit data from the previous 14 days:
  - Exercise sessions → list of `[timestamp, activity, duration(min), calories, steps]`.
  - Sleep sessions → list of `[timestamp, sleep_duration(min)]`.
  - Resting HR readings → list of `[timestamp, rhr]`.
- Simple averages (`steps_14d`, `calories_14d`, `rhr_14d`, `sleep_dur_14d`) are computed but the prompt currently embeds the raw 14-day lists for steps, calories, RHR, and sleep minutes.

### Prompt construction (per example)
- `instruction`: “You are a personalized healthcare agent trained to predict <SUBTASK> which ranges from <lo> to <hi> based on physiological data and user information.”
- `input`: “The recent 14-days sensor readings show: [Steps]: <list>, [Burned Calories]: <list>, [Resting Heart Rate]: <list>, [SleepMinutes]: <list>, [Mood]: <m> out of 5; What would be the predicted <SUBTASK>?”
- `output`: “The predicted <SUBTASK> level is <label>.”
- Triplet appended to `final_data`.

### Splitting & files
- After shuffling, first half → train; second half → eval (eval capped at 300 examples for PMData tasks).
- Written to repo root:
  - `PMData_<subtask>_train_3.json`
  - `PMData_<subtask>_train_10.json`
  - `PMData_<subtask>_train_25.json`
  - `PMData_<subtask>_train_all.json`
- Eval written to `eval/data/pmdata_<subtask>/step1.json` (relative to repo root).

## Adjustments you might want
- Change `SUBTASK` inside the PMData block to generate different labels.
- Swap the f-string lists for the averaged stats (`steps_14d`, etc.) if you prefer scalar inputs.
- Tweak the look-back window by altering the `timedelta(days=14)` comparisons.
- Add deduplication / NA filtering before `final_data.append` if data quality is an issue.

## Quick sanity checks
- Verify directory structure and filenames match expectations before running.
- Spot-check a generated example in `PMData_<subtask>_train_all.json` to ensure labels and feature spans align. 
