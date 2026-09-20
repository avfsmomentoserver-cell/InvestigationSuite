# Newbie Glossary — What Everything Means (No Math Required)

## Getting Started

### What Is This Platform?

A forecasting tool that watches how people use an app — clicks, form submissions, page views, scrolls — groups those moments into activity bursts (the platform calls these "sessions"), figures out patterns across those bursts, and predicts what happens next. Every prediction dimension is explained in everyday language, not statistics jargon.

### Key Analogy

Think of tracking someone like tracking their gym visits:

- **Each visit** is a "round" (the platform calls this a "session")
- **How long they work out** = duration
- **How often they show up** = gap pattern
- **Whether workouts get longer** = trend
- **Whether some days they sprint and others lounge** = consistency

---

## Section 1: Data Collection Terms

| Column Name | Newbie Name | Plain English Explanation | Real Example |
|---|---|---|---|
| `recorded_at` | When It Happened | The exact moment someone did something. Stored as a timestamp like `"2026-01-05T15:00:00Z"` (January 5th at 3 PM UTC). | `"2026-09-01T12:00:00Z"` |
| `user_id` | Who Did It | The person or device that performed the action. In real data this could be a user ID, an email, or a device identifier. | `"user_12"` or `"device_android_07"` |
| `event_type` | What Kind of Action | The type of thing that happened. Common values are `click`, `submit`, `view`, and `scroll`. Other types may appear depending on what your app records. | `click` |
| `value` | How Much Weight | How important that action counts for. In the standard pipeline every occurrence counts as `1`. You would only set a different value if you wanted certain actions to count as more or less than one occurrence. | `1` (one click = one count) |
| `NaN` / `null` | Missing or Unknown | We don't know when, who, or what — the data point is incomplete. Rows missing `user_id` or `recorded_at` are dropped before any analysis. | That button wasn't logged because of a phone glitch |
| Exact row duplicate | Copied Entry | Two identical rows in the raw CSV (same user, same time, same event). The pipeline drops extras automatically using a keep-first rule. | Two identical clicks at exactly the same millisecond — probably a system glitch |

**Where these live:** These are the columns in your raw input CSV (or similar file). The ETL step (`src/forecasting/etl.py`) reads them first, removes duplicates and nulls, then passes clean data downstream.

---

## Section 2: Grouping Terms — What Sessionize Does

Sessionize takes a flat list of events and turns them into grouped "bursts" of activity. Think of it like splitting your text messages into separate conversations: close together in time means the same conversation; a long silence means a new one started.

| Concept | Newbie Name | Plain English Explanation | Code Reference |
|---|---|---|---|
| `round` / `session` | Activity Burst | A group of actions that happen close together. All actions within a few minutes belong to the same burst. | `src/forecasting/sessionize.py` |
| `gap_minutes` | Break Threshold | The idle time used to draw the line between two bursts. Default is **30 minutes**. If a user does nothing for 30 minutes or more, the next action starts a new burst. | `sessionize(..., gap_minutes=30)` |
| `round_index` | Burst Number | A simple counter per user starting at 1. First burst = 1, second = 2, etc. | Added during sessionize |
| `round_id` | Burst Label | A unique name for each burst, formatted as `{user_id}_{burst_number}`. Used to link events to their burst's summary. | e.g., `"user_12_3"` |
| `round_start` | Burst Start | When the first action in this burst happened. | Computed per burst |
| `round_end` | Burst End | When the last action in this burst happened. | Computed per burst |
| Gap flag (`is_new_round`) | New Burst Marker | An internal signal set when the time since the previous action exceeds `gap_minutes`. Each marker starts a fresh burst. | Internal, discarded after grouping |
| Cumulative sum of gap flags | Counting Bursts | Numbering each burst sequentially so we can track which one comes next per user. | `cumsum()` in `sessionize.py` |

**Default settings:** `gap_minutes=30`. Change it if your users tend to go longer between distinct activity periods.

---

## Section 3: The 8 Prediction Dimensions

These are the heart of the platform. For every recorded burst, the feature engine computes eight numbered pieces of information. They also become the inputs that the trained model uses to make predictions.

### Dimension 1 — Things That Will Happen Next

| Field | Value Type | What It Means |
|---|---|---|
| `event_count_predicted` | Number (e.g., `7.5`) | Based on past bursts, how many actions should we expect the person to perform in their next burst? A value of 7.5 means "somewhere around seven or eight." |

**Source column:** This is the primary prediction target produced by the model at serving time. During training the model learns from `next_event_count`, which is the actual count from the *following* burst.

**Real example:** "Based on what usually happens, this person is likely to do about 7–8 things in their next activity burst."

#### How It Is Computed From Source Events

This prediction comes from a trained model (a LightGBM decision tree ensemble), not from simple arithmetic. Here's how it works:

1. **Events in recent bursts**: Count all rows where `user_id` matches AND all timestamps fall within `[round_start, round_end]`. This gives you the event count for each past burst.
2. **Other signals per burst**: For every burst, also compute duration, gap since previous burst, rolling average gap, trend slope, consistency (CV), per-event-type counts, growth ratio, and stay-home risk score. All of these are computed separately — see Dimensions 3 through 8 below.
3. **Model lookup**: The model was trained on thousands of similar users. It takes all those signals together and passes them through many decision trees. Each tree asks a question like "is the growth ratio above 0?" and follows a path to an answer. The final predicted number is the average across all trees.
4. **Result**: A single number (e.g., `7.5`) shown as `event_count_predicted`. No hidden factors outside what you just read.

The model only uses features derived from **your own activity bursts** — nothing external affects the result.

---

### Dimension 2 — When They Come Back Next

| Field | Value Type | What It Means |
|---|---|---|
| `gap_hours_until_next` | Hours (e.g., `14.3`) | People like this typically return about X hours after their last burst. Here's our best guess for when they'll come back. |

**Source column:** `rolling_mean_gap` — the average time (in seconds, converted to hours) between a user's bursts, calculated using only past bursts (never future ones).

**Real example:** "People like this come back about 14 hours after their last burst. That's our best guess for when they'll return."

#### How It Is Computed From Source Events

This tells us the average gap between consecutive bursts, converted to hours. Here's how we get there:

1. **Events in each burst**: Count all rows where `user_id` matches AND timestamps fall within `[round_start, round_end]`. Note the `round_start` timestamp for each burst.
2. **Gap between bursts**: Sort your bursts by time. For each burst (starting from the second), subtract the previous burst's `round_end` from the current burst's `round_start`. Result is in seconds.
3. **Simple average**: Add up all the gaps (in seconds), divide by the number of gaps. Example: gaps of `7200`, `14400`, `10800` seconds → `(7200 + 14400 + 10800) / 3 = 10800` seconds average.
4. **Convert to hours**: Divide by 3600 → `10800 / 3600 = 3.0` hours. Stored as `rolling_mean_gap` (cumulative average using only past data).
5. **Result**: The number shown as `gap_hours_until_next` (e.g., `14.3`).

For the very first burst, no gap exists yet so the value defaults to `0`.

The computation uses only **gaps within sessions** — nothing outside your activity bursts affects the result.

---

### Dimension 3 — Busier or Quieter?

| Field | Value Type | What It Means |
|---|---|---|
| `trend_direction` | String: `trending_up`, `stable`, or `trending_down` | Is their activity going up (more engaged over time), down (losing interest), or staying about the same? |

**Source columns:** `trend_linear_slope` (a number indicating direction and steepness) and `trend_direction_label` (the human-readable label above).

**How the label is chosen:**
- **trending_up** — each successive burst has noticeably more actions than the last
- **trending_down** — each successive burst has noticeably fewer actions
- **stable** — the changes between bursts are small enough to be normal fluctuation

**Real example:** "Their activity has been climbing over time — they're more engaged with each burst."

#### How It Is Computed From Source Events

This dimension uses linear regression to determine whether burst sizes are going up, down, or staying flat. Here's how:

1. **Count events per burst**: For each burst where `user_id` matches, count all rows with timestamps inside `[round_start, round_end]`.
2. **Number your bursts**: First burst = 1, second = 2, third = 3, etc. (stored as `round_number`).
3. **Draw a best-fit line**: Plot burst number on the x-axis and event count on the y-axis. Find the slope of the straight line that minimizes total distance to all points. Formula: `slope = (n * Σ(x*y) - Σx * Σy) / (n * Σ(x²) - (Σx)²)`, where n is the number of bursts.
4. **Set the significance threshold**: Calculate `half_avg_slope = 0.5 * mean_event_count / (n - 1)`. This is half the average change you'd expect per step.
5. **Compare**: If slope > `half_avg_slope` → `trending_up`. If slope < `-half_avg_slope` → `trending_down`. Otherwise → `stable`.
6. **Fewer than 2 bursts?** Default to `stable` with slope `0.0`.
7. **Result**: One string label shown as `trend_direction`.

The computation is strictly within **your session history** — the threshold adapts to your typical activity level.

---

### Dimension 4 — Do Their Bursts Vary a Lot?

| Field | Value Type | What It Means |
|---|---|---|
| `session_consistency` | String: `low_cv` or `high_cv` | Are their bursts predictable (similar number of actions each time) or wild (sometimes 1 action, sometimes 50)? |

**Source column:** `cv_event_count` — a number measuring spread relative to the average. A value below 0.5 maps to `low_cv` (consistent). Above 0.5 maps to `high_cv` (unpredictable).

Also tracked but not exposed as a label: `cv_duration_seconds` — measures the same consistency idea but for burst length instead of action count.

**Real example:** "When they're active, they reliably do about five to seven actions per burst. Very consistent."

#### How It Is Computed From Source Events

This measures how predictable burst sizes are by comparing spread to average. Here's the math:

1. **Count events per burst**: For each burst, count rows where `user_id` matches and timestamps are within `[round_start, round_end]`.
2. **Calculate average**: Add all burst sizes, divide by the number of bursts → `mean_size`.
3. **Calculate spread**: Compute the standard deviation of all burst sizes. This measures how far each burst deviates from the average.
4. **Normalize**: Divide standard deviation by the mean. Example: average burst size = 10, standard deviation = 3 → CV = 3/10 = 0.3.
5. **Threshold check**: If CV < 0.5 → `low_cv` (consistent bursts). If CV ≥ 0.5 → `high_cv` (unpredictable bursts).
6. **One burst?** With only one data point, standard deviation is zero, so CV defaults to `0` → `low_cv`.
7. **Result**: The label shown as `session_consistency`, plus the raw number stored as `cv_event_count`.

Same logic applies to `cv_duration_seconds`, measuring consistency of burst length instead of count.

The computation is strictly **within your own session history** — cross-user comparisons don't affect the result.

---

### Dimension 5 — Most Likely Activity Type

| Field | Value Type | What It Means |
|---|---|---|
| `top_event_type` | String: e.g., `click`, `submit`, `view`, `scroll`, or `unknown` | When this person is active, what kind of action do they tend to do most? |

**Source columns:** Per-event-type columns (e.g., `click`, `submit`, `view`) created by cross-tabulating event types per burst. At serving time the highest-count non-zero type wins.

**Order of preference:** The system checks in this order: click → submit → view. Whichever has a count greater than zero is reported.

**Real example:** "When they're active, clicking is what they do most. More than half of their actions are clicks."

#### How It Is Computed From Source Events

This finds whichever action type appears most frequently in the current burst. Here's how:

1. **Find current burst events**: Select all rows where `user_id` matches AND timestamps fall within `[round_start, round_end]` for the burst being analyzed.
2. **Count each type**: Tally how many rows have each `event_type` value. Known types include `click`, `submit`, `view`, and `scroll`. Any other value appears as an additional column labeled by its type name.
3. **Create cross-tabulation**: Build a table mapping each burst ID to its count of each event type. Unknown types get their own column automatically.
4. **Pick the winner**: Among the counts, find the type with the highest number. In the serving pipeline, types are checked in priority order: `click` → `submit` → `view`. Whichever has a count greater than zero is reported first.
5. **Fallback**: If no known type is found, show `unknown`.
6. **Result**: One string label shown as `top_event_type` (e.g., `click`, `submit`, `view`, `scroll`, or `unknown`).

Per-event-type counts are also stored as individual columns (e.g., `click`, `submit`) for use by the model in other predictions.

---

### Dimension 6 — Are Sessions Growing?

| Field | Value Type | What It Means |
|---|---|---|
| `growth_ratio` | Number (e.g., `0.25` or `-0.15`) | Compared to their last burst, is this one bigger or smaller? Positive means growing, negative means shrinking. |

**Source column:** `event_count_growth_ratio` — calculated as `(this_burst_size - previous_burst_size) / previous_burst_size`. A value of `0.25` means the current burst was 25% larger than the previous one.

**Note:** If the previous burst had zero actions, the ratio defaults to 0 (no growth assumed).

**Real example:** "This burst had 25% more actions than the last one. Their engagement is expanding."

#### How It Is Computed From Source Events

This computes the relative change in burst size compared to the immediately preceding burst. Here's the exact formula:

1. **Current burst size**: Count all rows where `user_id` matches and timestamps fall within the current burst's `[round_start, round_end]`. Call this `current_count`.
2. **Previous burst size**: Count all rows for the SAME `user_id` in the immediately prior burst (the one whose `round_index` is one less). Call this `previous_count`.
3. **Apply the formula**: `(current_count − previous_count) ÷ previous_count`. Example: previous burst had 5 events, current has 10 → `(10 − 5) / 5 = 1.0` → growing by 100%.
4. **Edge case**: If the previous burst had zero events, the result defaults to `0` (no growth assumed).
5. **Result**: A signed number stored as `event_count_growth_ratio`. Positive = bigger than before. Negative = smaller than before. Zero = same size.

The computation compares only **consecutive bursts for the same user** — nothing else matters.

---

### Dimension 7 — Chance They Stop Coming Back

| Field | Value Type | What It Means |
|---|---|---|
| `stay_home_risk` | Number from `0.0` to `1.0` | Based on how their breaks between bursts are changing, do they seem like they might stop coming back entirely? |

**Scale:**
- `0.0` = very likely to return soon
- `1.0` = very likely stopped completely

**Source columns:** `latest_gap_hours` (hours since their last known activity), `max_gap_seen_hours` (longest silence ever recorded for this user), and `stay_home_risk_score` (the computed score).

**How it works:** If the most recent break is much longer than their typical break pattern, the risk score increases. The score caps at 0 and 1 to stay within a reasonable range. A short adjustment window (ratio of 1.5× their usual gap as the baseline, linearly scaled beyond that) determines when concern kicks in.

**Real example:** "They've been quiet twice as long as usual. There's a 60% chance they've stopped using the app altogether."

#### How It Is Computed From Source Events

This estimates churn risk by comparing the most recent silence to the longest ever seen. Here's the math:

1. **All gaps between bursts**: For the user, compute the time (in hours) between each consecutive pair of bursts. These are stored as `latest_gap_hours` (most recent gap) and `max_gap_seen_hours` (the biggest gap ever recorded).
2. **Longest silence**: Find the maximum of all gap values across the user's entire history. Example: gaps of 1h, 3h, 5h, 2h → `max_gap_seen_hours = 5`.
3. **Latest silence**: Take the gap since the most recent burst. Example: last break was 10 hours → `latest_gap_hours = 10`.
4. **Compute the ratio**: `ratio = latest_gap_hours / max_gap_seen_hours`. Example: `10 / 5 = 2.0`. (If both are zero or missing, ratio defaults to `1.0`.)
5. **Apply the risk formula**: `risk = (ratio − 1.5) / 2.0`. Then clamp the result to the range [0, 1]: `max(0, min(1, risk))`. Example: `(2.0 − 1.5) / 2.0 = 0.25` → stays `0.25`. Another example: ratio = 5 → `(5 − 1.5) / 2.0 = 1.75` → clamped to `1.0`.
6. **Result**: A number from `0.0` (will return soon) to `1.0` (very likely stopped) shown as `stay_home_risk_score`.

The baseline concern kicks in when the latest gap is 1.5× the maximum ever seen. Below that, risk is `0`. Above that, it rises linearly until hitting the cap.

The computation is strictly **within your own session history**.

---

### Dimension 8 — How Sure Are We?

| Field | Value Type | What It Means |
|---|---|---|
| `confidence_range` | Object with `lower_bound` and `upper_bound` | Our best guess is X, but there's a range. In similar situations, the actual result has fallen between the lower and upper bound about most of the time. |

**Source calculation:** `margin = predicted_value * max(0.2, cv_event_count)`. The wider the historical variation in burst sizes, the wider the range. Minimum margin is always 20% of the prediction to prevent falsely precise estimates.

**Real example:** "Our best guess is 7.5 actions. But looking at similar users, the actual count landed somewhere between 4 and 12."

#### How It Is Computed From Source Events

This creates a confidence interval around the predicted event count based on historical variability. Here's the exact calculation:

1. **Get the prediction**: The model outputs a raw predicted number. Call this `raw_pred`. Example: `raw_pred = 7.5`.
2. **Measure historical variation**: Look at `cv_event_count` — the coefficient of variation from Dimension 4. This captures how much burst sizes vary relative to their average.
3. **Choose the margin factor**: Take the larger of two numbers: `0.2` (a minimum 20% safety floor) or your `cv_event_count`. Example: if CV = 0.3 → use `0.3`. If CV = 0.1 → use `0.2` (the floor wins).
4. **Compute the margin**: `margin = raw_pred × margin_factor`. Example: `7.5 × 0.3 = 2.25`.
5. **Build the range**: Subtract the margin from the prediction for the lower bound, add it for the upper bound. Apply a floor of 0 to the lower bound.
   - `lower_bound = max(0, raw_pred − margin)` → `max(0, 7.5 − 2.25) = 5.25`
   - `upper_bound = raw_pred + margin` → `7.5 + 2.25 = 9.75`
6. **Result**: Two numbers shown as `confidence_range.lower_bound` and `confidence_range.upper_bound`, rounded to 2 decimal places.

Larger historical variation produces a wider range; perfectly consistent users get at least a 20% margin on either side.

---

## Section 4: Model Training Terms

| Term (Math Jargon) | Newbie Name | Plain English Explanation | Where It Lives |
|---|---|---|---|
| Training set | Learning Examples | Past burst data we feed into the model so it can learn patterns. Bigger sets generally produce better predictions. | `src/forecasting/train.py` |
| Test set | Practice Exam | Past burst data we hold back to see if the model actually learned or just memorized the training examples. | `src/forecasting/train.py` |
| Cross-validation | Taking Multiple Practice Exams | Splits the data into several chunks, trains on each combination, and tests on what was held out. Averages the results for a fairer quality score. | `src/forecasting/train.py` |
| TimeSeriesSplit | Chronological Split | Always trains on older data, tests on newer data. You wouldn't use tomorrow's exam answers to study for today's test — so shouldn't a model. | `src/forecasting/train.py` |
| MAE (Mean Absolute Error) | Average Wrong By | On average, the model's predictions miss the real answer by this number. Lower is better. An MAE of 2 means predictions are off by about 2 actions on average. | `src/forecasting/train.py` |
| Early stopping | Knowing When to Stop Learning | If the model stops improving on the practice exam after a certain number of rounds, training halts to prevent memorization. Default: 50 unimproved rounds triggers a stop. | `src/forecasting/train.py` |
| `num_boost_round` (default 1000) | Maximum Study Rounds | The ceiling on how many times the model reviews the material before early stopping cuts it short. | `src/forecasting/train.py` |
| Learning rate (default 0.05) | Speed of Learning | A smaller learning rate means the model learns slowly and carefully; a larger one means faster but potentially sloppy learning. | `src/forecasting/train.py` |
| LightGBM | The Learning Engine | A machine learning library that's good at finding patterns in structured numbers. Think of it as a very diligent student specialized in numerical data. | `src/forecasting/train.py` (`import lightgbm as lgb`) |
| `feature_cols` | Clues We Use | The pieces of information the model looks at to make its prediction — essentially every computed dimension except identifiers and the target itself. | `src/forecasting/train.py` |
| `target_col` / `target` | What We're Predicting | The answer we want the model to figure out. In the demo this is `next_event_count` — how many actions happen in the next burst. | `src/forecasting/train.py` |
| `model_out_path` | Saved Brain | The file where the trained model is stored after learning finishes. It contains the model plus the list of feature columns it expects. | Defaults to `models/baseline.joblib` |

---

## Section 5: Pipeline Flow — Step by Step

Here is the end-to-end journey from raw CSV to a running API server returning predictions. Each step maps to a file in `src/forecasting/`.

### Step 0: Generate or Acquire Raw Data

You start with a CSV file (or similar) containing rows with at least `user_id`, `recorded_at`, and optionally `event_type` and `value`. Each row represents one observed action.

**Analogy:** Imagine sorting mail arriving at an office — everything piles up here.

**Demo data:** The demo generates synthetic data with 50 users, 10–40 events each, random timestamps starting from January 1st 2026. Event types are chosen from `click`, `submit`, `view`, `scroll`. Events per user are spaced 1–60 minutes apart with 1–48 hour gaps between users.

---

### Step 1: Data Cleansing (ETL Layer)

**File:** `src/forecasting/etl.py`

The raw data goes through two filters:

1. **Remove copied entries** — if the same row appears multiple times, keep only the first copy
2. **Drop incomplete rows** — rows missing `user_id` or `recorded_at` are removed entirely

After filtering, a log message reports how many rows were dropped and why: `[DEDUP] input_rows=N -> after_dedup=M -> after_na_filter=K -> dropped=D (X duplicates + Y null-key rows)`

**Analogy:** Like throwing away torn envelopes and letters with no return address before you start reading.

---

### Step 2: Sessionize — Group Events Into Bursts

**File:** `src/forecasting/sessionize.py`

Clean events are sorted by user and timestamp. Then consecutive events from the same user are compared: if the gap between them exceeds `gap_minutes` (default 30), a new burst begins. All events within a burst are tagged with a shared `round_id`.

For each burst, three summary fields are computed:
- `round_start`: the earliest timestamp in the burst
- `round_end`: the latest timestamp in the burst
- `event_count`: how many events fall inside

**Analogy:** Like grouping text messages into separate conversations. Messages sent within a minute of each other belong to the same chat; a two-hour silence means a new conversation started.

---

### Step 3: Feature Engineering — Write Notes About Each Burst

**File:** `src/forecasting/features.py`

From the burst-level grouped data, the feature engine computes all 8 prediction dimensions described in Section 3. The output is one row per burst with columns for every dimension:

| What gets computed | Description |
|---|---|
| Base metrics | Action count, duration in seconds |
| Timing metrics | Gaps between bursts, rolling averages |
| Trend analysis | Whether activity is increasing, decreasing, or steady |
| Consistency scores | How variable burst sizes and durations are over time |
| Activity mix | Count of each event type per burst |
| Growth signals | How much bigger or smaller compared to the prior burst |
| Churn signals | Risk score based on whether breaks are getting longer |

**Deduplication guarantee:** Before computing, the engine asserts that each `round_id` belongs to exactly one user. If a collision is found (which shouldn't happen if sessionize worked correctly), it raises a clear error.

**Analogy:** Like writing notes about each conversation — length, number of messages sent, main topics discussed.

---

### Step 4: Target Assignment — Create the Answer Key

**File:** `examples/demo.py` (lines 54–64)

The cleaned-and-featured parquet needs a "what we're predicting" column. In production this comes from your downstream system. In the demo, it simulates a target called `next_event_count` by taking the current burst's event count, multiplying by 0.9 (simulating gradual decline), adding controlled randomness, and clamping to a minimum of 1.

The resulting parquet has shape `(N_rounds, M_columns)` where N is the number of bursts and M includes all feature columns plus the target.

**Analogy:** Writing the answer key for a textbook — each burst now has both the clues (features) and the answer (target) hidden behind it.

---

### Step 5: Model Training — Learn the Patterns

**File:** `src/forecasting/train.py`

The training process:

1. Reads the features parquet file
2. Sorts data chronologically by `round_start`
3. Identifies which columns are clues (`feature_cols`) and which is the answer (`target_col`)
4. Validates no duplicate rows exist in the feature matrix
5. Splits data into folds using chronological ordering (oldest gets trained on, newest gets tested on)
6. For each fold, trains a LightGBM model with early stopping (stops if no improvement for 50 rounds)
7. Records the MAE for each fold
8. Saves the final model (the last fold's model) along with the list of feature column names to `models/baseline.joblib`

If fewer than ~2 samples per fold exist, training fails with a clear error telling you how many samples you need.

**Analogy:** Studying the notes with the answer key. The model tries to find mathematical relationships between the notes and the answers. It takes practice exams (validation folds) to check its understanding. If it stops improving, training stops before it starts blindly memorizing.

---

### Step 6: Serving — Run the API Server

**File:** `src/forecasting/serve.py`

A FastAPI server loads the saved model on startup. You send it a request containing:

- `entity_id`: who you're predicting for
- `history`: a list of past bursts with their features
- `horizon`: how far ahead to predict (default 1 — next burst only)
- `mode`: deterministic (current setting)

The server returns a JSON object with all 8 prediction dimensions plus metadata explaining what each one means. The confidence range is computed dynamically based on the user's historical consistency.

**SDK:** Use the Python SDK at `api/sdk/python/forecast_sdk.py` for a cleaner interface:

```python
from api.sdk.python.forecast_sdk import ForecastSDK
sdk = ForecastSDK("http://localhost:8000")
response = sdk.forecast("user_42", [{"ts": "...", "features": {...}}], horizon=1)
```

**Analogy:** Using the trained model to answer questions in real-time. Like asking a consultant who studied all past cases and gives informed recommendations.

---

### Complete Pipeline Sequence

```
Raw CSV ──→ Dedup & Null Filter ──→ Sessionize ──→ Feature Engine ──→ Parquet File
                                                                                        │
                                                                                        ▼
               SDK Call ◄── JSON Response ◄── Serve API ◄── Trained Model ◄── Train & Save
                  ▲                                                        │
                  │                                                        │
                  └────────────────  History Features  ◄──────────────────┘
```

---

## Quick Reference Card

Use this card to quickly map what you want to know to which prediction dimension answers it.

| What You Want to Know | Which Dimension Answers It | One-Sentence Summary |
|---|---|---|
| How many things will happen next? | `event_count_predicted` | Based on past bursts, here's our best guess for the action count in the next one. |
| When will they come back? | `gap_hours_until_next` | Based on historical timing, here's our estimate of hours until their next burst. |
| Are they getting more or less active? | `trend_direction` | trending_up, stable, or trending_down over their series of bursts. |
| Are their sessions predictable? | `session_consistency` | low_cv means reliable bursts; high_cv means wild swings between them. |
| What do they do most while active? | `top_event_type` | The most frequent action type in their recent bursts (click, submit, view, scroll, or unknown). |
| Is their engagement growing or shrinking? | `growth_ratio` | Positive means bigger than last burst; negative means smaller. |
| Might they stop using the app? | `stay_home_risk` | 0.0 = probably returning soon; 1.0 = very likely stopped completely. |
| How confident is this prediction? | `confidence_range` | A lower and upper bound showing the likely range for the predicted value. |

---

## Frequently Asked Questions

### Can I change the burst threshold?

Yes. Pass `gap_minutes` when calling the sessionize function. Default is 30. Set it higher if your users naturally take longer breaks between distinct activity sessions, lower if they tend to be continuously active.

### What happens if I only have one burst for a user?

The model will still produce a prediction, but trend and consistency dimensions may default to neutral values (e.g., `stable`, `low_cv`). Historical context matters for accurate trend detection.

### What format do timestamps need to be?

ISO 8601 format works: `"2026-01-05T15:00:00Z"`. The pipeline converts them to proper datetime objects internally.

### How large does my dataset need to be for training to work?

At least a handful of bursts — the training code enforces a minimum of roughly 2–5 samples depending on the number of folds. With fewer than ~10 total bursts, consider whether training a model makes sense versus relying on simpler heuristics.

### Where do I find the actual column names for building queries or reports?

The full list lives in the parquet output from `compute_round_features()` in `src/forecasting/features.py`. Key columns include: `event_count`, `duration_seconds`, `gap_since_prev_seconds`, `rolling_mean_gap`, `trend_linear_slope`, `trend_direction_label`, `cv_event_count`, `cv_duration_seconds`, `event_count_growth_ratio`, `latest_gap_hours`, `max_gap_seen_hours`, `stay_home_risk_score`, plus per-event-type columns (`click`, `submit`, `view`, `scroll`).

---

## Appendix: Configuration Summary

| Setting | Default | Where to Change | What It Controls |
|---|---|---|---|
| `gap_minutes` | 30 | `sessionize(df, gap_minutes=X)` | Minutes of silence before starting a new burst |
| Learning rate | 0.05 | `params` dict in `train_baseline()` | How aggressively the model adjusts after each review round |
| Early stopping rounds | 50 | `early_stopping(stopping_rounds=50)` | Rounds without improvement before training stops |
| Maximum boost rounds | 1000 | `num_boost_round=1000` | Hard cap on training iterations |
| CV split count | min(5, max(2, n_samples // 2)) | Auto-computed in `train_baseline()` | How many validation folds to use |
| Confidence floor | 20% | `max(0.2, cv)` in `serve.py` | Minimum margin percentage even for perfectly consistent users |
| Stay-home baseline ratio | 1.5 | `(ratio - 1.5) / 2.0` in `features.py` | How many times their normal gap before concern kicks in |
| Consistency threshold | 0.5 | `"low_cv" if cv < 0.5 else "high_cv"` in `serve.py` | Boundary between predictable and unpredictable burst patterns |
