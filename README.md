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

- **Location:** 50 states, DC, Puerto Rico.
- **Age group** (`Stratification1`): Overall
- **Demographic** (`Stratification2`): Overall
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
final shape: (9952, 28)
Topic    Functional difficulties associated with subjective cognitive decline or memory loss among older adults  ...  Talked with health care professional about subjective cognitive decline or memory loss
cluster                                                                                                          ...                                                                                        
0                                                     32.0                                                       ...                                               43.0                                     
1                                                     51.9                                                       ...                                               49.2                                     
2                                                     42.2                                                       ...                                               45.1                                     

[3 rows x 4 columns]
cluster 0 -> ['Alaska', 'Arizona', 'California', 'Colorado', 'Connecticut', 'Delaware']
cluster 1 -> ['Illinois', 'North Carolina', 'Puerto Rico']
cluster 2 -> ['Alabama', 'Arkansas', 'District of Columbia', 'Florida', 'Georgia', 'Indiana']
linear regression -> r2: 0.812 mae: 5.19
random forest -> r2: 0.869 mae: 4.11

Process finished with exit code 0
```

Cluster averages and MAE are on the 0–1 scaled range, not in percentage points. The cluster lists are cut off at six locations each.

### What the results mean

- **Topic explains most of the prediction.** The four topics have very different typical values, so knowing the topic gets you most of the way. Predicting each topic's average alone scores R² of about 0.63. The random forest improves that to 0.81 by also using location, demographic, age, and year.
Linear regression’s weak score came from the encoding, not the model. LabelEncoder gave each state an integer (Alabama = 0, Alaska = 1, and so on), so linear regression treated states as if they had a meaningful order. Switching to OneHotEncoder gives each category its own yes/no column (is_Alabama, is_Arizona, …), so the model learns a separate adjustment for each state with no fake ordering. With that one change, linear regression’s R² rose from about 0.01 to 0.81.
- ~**Puerto Rico forms its own cluster** because its values are far from every state's. With min-max scaling, one outlier like this can take a whole cluster.~
  - Puerto Rico used to form its own cluster. After switching to the CDC’s pooled 2019–2022 estimates for the whole population, it now groups with Illinois and North Carolina, the three locations where the largest share of people with SCD report needing help with daily activities. It still stands out for having the lowest SCD rate (about 7%).

## Known limitations

These are real issues in the current version, listed so the results are read correctly:

1. **The target mixes four different measures.** The prediction is "the value for this topic," not one cognitive decline rate.
2. ~~**Regional and national rows are mixed in with states.** About 4,100 rows are aggregates of other rows, so the same information is counted more than once.~~
   - **Fixed:** the national total and the four Census regions are filtered out by `LocationAbbr` before clustering or modeling.
3. ~~**The clustering table averages across demographics and years.** It filters age group to `Overall` but not demographic. Each cell averages about 16 rows (gender, race groups, and years).~~
   - **Fixed:** the clustering table now uses only the CDC's pooled 2019–2022 estimates with both `Stratification1` and `Stratification2` set to `Overall`, so each cell is exactly one value for the whole population. Massachusetts, Montana, and New Jersey have no pooled estimates, so 49 locations are clustered.
4. ~~**Overlapping time periods.** Pooled 2019–2022 rows overlap with single-year rows, and the model only sees `YearStart`.~~
   - **Fixed:** the model now trains only on single-year rows (where `YearStart` equals `YearEnd`). The pooled rows are still used for clustering, where they're the better source.
5. ~~**Scaling happens before the train/test split, and the target is scaled too.** Error is reported on a 0–1 scale instead of percentage points.~~
   - **Fixed:** the scaling step was removed. The random forest doesn't need it, and errors are now reported in percentage points.
6. **Survey noise.** The median 95% confidence interval in this data is about 16 percentage points wide, so prediction errors of a few points are already within the survey's own uncertainty.
7. **k = 3 was chosen, not tuned.** On the pooled table, silhouette scores for k = 2 through 6 all fall between 0.24 and 0.27, so no k clearly wins and the states overlap substantially.

## Planned improvements

- Predict one topic at a time, or treat topic explicitly as an input and compare against a topic-average baseline
- Keep the target in percentage points, and fit scalers on training data only
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
