# Cognitive Decline State Analysis

Clusters U.S. states by how older adults report subjective cognitive decline (SCD), and trains regression models to predict a reported percentage from location, topic, age group, demographic group, and year.

Started as coursework for an AI/ML class.

## Data

**Source:** CDC, *Alzheimer's Disease and Healthy Aging Data* (Behavioral Risk Factor Surveillance System survey), published on data.cdc.gov. U.S. government data, public domain.

**File:** `Alzheimers_Disease_dataset.xlsx` (in this repo, about 38 MB).

The script only uses rows where `Class == "Cognitive Decline"`. That subset has 22,182 rows and four topics:

| Topic | What the percentage measures | Mean |
|---|---|---|
| Subjective cognitive decline or memory loss | Share of adults 45+ reporting SCD | ~12% |
| Functional difficulties associated with SCD | Share of people **with SCD** whose daily functioning is affected | ~39% |
| Need assistance with day-to-day activities because of SCD | Share of people **with SCD** who need help | ~34% |
| Talked with a health care professional about SCD | Share of people **with SCD** who talked to a doctor | ~45% |

Only the first topic is a decline rate for the whole population. The other three are measured among people who already report SCD.

Each row is broken down by:

- **Location:** 50 states, DC, Puerto Rico, and 5 aggregates (Midwest, Northeast, South, West, and the national total)
- **Age group** (`Stratification1`): Overall, 50–64, 65+
- **Demographic** (`Stratification2`): Overall, Male, Female, or a race/ethnicity group
- **Year:** 2015–2022. Most rows cover a single year, and some cover the pooled range 2019–2022.

## How it works

`main.py` runs top to bottom in three stages.

### 1. Cleaning

1. Keep only the cognitive decline rows (22,182).
2. Drop rows with no `Data_Value` (8,046 dropped).
3. Drop footnote columns and the duplicate `Data_Value_Alt` column.
4. Fill a blank `Stratification2` with `"Overall"`.
5. Drop rows with no confidence interval. This leaves **14,039 rows**.
6. Label-encode location, topic, age group, and demographic as integers.
7. Min-max scale `YearStart`, `Data_Value`, and the confidence limits to 0–1.

### 2. Clustering (unsupervised)

- Build a table with one row per location and one column per topic, using rows where age group is `Overall`.
- Scale it and run **K-means with k = 3**.
- Print the average value of each topic per cluster and the first six locations in each cluster.

### 3. Prediction (supervised)

- **Features:** encoded location, topic, age group, demographic, and `YearStart`
- **Target:** `Data_Value` (scaled 0–1)
- **Split:** random 80/20 train/test (`random_state=42`)
- **Models:** linear regression, and a random forest (200 trees, max depth 12)
- **Output:** R², mean absolute error, and random forest feature importances

## Results

Output of `python main.py`:

```
(22182, 31)
dropped 8046 rows with no Data_Value
final shape: (14039, 28)

cluster  Functional difficulties  ...  Talked with health care professional
0        0.4                      ...  0.4
1        0.3                      ...  0.4
2        0.7                      ...  0.6

cluster 0 -> ['Alabama', 'Arizona', 'Arkansas', 'District of Columbia', 'Florida', 'Georgia']
cluster 1 -> ['Alaska', 'California', 'Colorado', 'Connecticut', 'Delaware', 'Hawaii']
cluster 2 -> ['Puerto Rico']

linear regression -> r2: 0.008  mae: 0.14
random forest     -> r2: 0.806  mae: 0.05

Feature importance (random forest)
Topic_enc              0.682
Stratification2_enc    0.125
LocationDesc_enc       0.087
Stratification1_enc    0.053
YearStart              0.053
```

Cluster averages and MAE are on the 0–1 scaled range, not in percentage points. The cluster lists are cut off at six locations each.

### What the results mean

- **Topic explains most of the prediction.** The four topics have very different typical values, so knowing the topic gets you most of the way. Predicting each topic's average alone scores R² of about 0.63. The random forest improves that to 0.81 by also using location, demographic, age, and year.
- **The linear regression result comes from the encoding, not the model.** Label encoding treats categories as ordered numbers (for example, location 0 to 56), which a linear model can't use well. A random forest can still split on these values, so it is less affected.
- **Puerto Rico forms its own cluster** because its values are far from every state's. With min-max scaling, one outlier like this can take a whole cluster.

## Known limitations

These are real issues in the current version, listed so the results are read correctly:

1. **The target mixes four different measures.** The prediction is "the value for this topic," not one cognitive decline rate.
2. (~**Regional and national rows are mixed in with states.** About 4,100 rows are aggregates of other rows, so the same information is counted more than once.~)
  - Aggregate rows removed with latest commit. 
3. **The clustering table averages across demographics and years.** It filters age group to `Overall` but not demographic. Each cell averages about 16 rows (gender, race groups, and years).
4. (~**Overlapping time periods.** Pooled 2019–2022 rows overlap with single-year rows, and the model only sees `YearStart`.~)
 - Kept rows that span the same year (i.e., YearStart == 2022 && YearEnd == 2022 means the row is kept).
5. **Scaling happens before the train/test split, and the target is scaled too.** Error is reported on a 0–1 scale instead of percentage points.
6. **Survey noise.** The median 95% confidence interval in this data is about 16 percentage points wide. A model can't reliably beat that level of uncertainty.
7. **k = 3 was chosen, not tuned.** Silhouette scores for k = 2 and k = 3 are nearly the same.

## Planned improvements

- Filter clustering to the fully `Overall` rows
- Predict one topic at a time, or treat topic explicitly as an input and compare against a topic-average baseline
- Keep the target in percentage points, and fit scalers on training data only
- One-hot encode categories for the linear model
- Hold out whole groups or later years in testing, not just random rows
- Add charts: a cluster map, predicted vs. actual values, and feature importance
- Split `main.py` into functions, add a download script instead of committing the data, and add tests

## Running it

Requires Python 3.11+.

```bash
git clone https://github.com/short-for-calculator/cognitive-decline-state-analysis.git
cd cognitive-decline-state-analysis
pip install -r requirements.txt
python main.py
```

The script reads `Alzheimers_Disease_dataset.xlsx` from the current directory and takes about 20 seconds to run.

## Project structure

```
main.py                          cleaning, clustering, and regression
Alzheimers_Disease_dataset.xlsx  CDC source data
requirements.txt                 pandas, openpyxl, scikit-learn
LICENSE                          MIT
```

## License

MIT. See `LICENSE`. The dataset is U.S. government public-domain data from the CDC.
