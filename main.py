# ~ pkg import list ~
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.metrics import silhouette_score
from pathlib import Path

# settings / const variables
TOPIC_LABELS = {
    "Subjective cognitive decline or memory loss among older adults": "SCD",
    "Functional difficulties associated with subjective cognitive decline or memory loss among older adults": "Functional",
    "Need assistance with day-to-day activities because of subjective cognitive decline or memory loss": "Needs help",
    "Talked with health care professional about subjective cognitive decline or memory loss": "Talked with provider"
}
AGGREGATES = ["US", "NRE", "MDW", "SOU", "WEST"]
TEST_YEAR = 2022
DATA_PATH = "Alzheimers_Disease_dataset.xlsx"
RESULTS_DIR = Path("results")
POOL_START, POOL_END = 2019, 2022
K = 3
RANDOM_SEED = 42
CAT_COLS = ["LocationDesc", "Topic", "Stratification1", "Stratification2"]
FEATURES = CAT_COLS + ["YearStart"]
CLUSTERS_CSV = RESULTS_DIR / "state_clusters.csv"

def main():
    pass

df = pd.read_excel(DATA_PATH)

# ~ data normalization and cleaning! ~
# dropping all rows except those mentions 'Cognitive Decline'
cog = df[df['Class'] == 'Cognitive Decline'].copy()
print(cog.shape)

# if row has empty Data_Value == dropped from analysis
n = len(cog)
cog = cog.dropna(subset=['Data_Value'])
print('dropped', n - len(cog), 'rows with no Data_Value')

# columns that provide no added value to analysis
cog = cog.drop(columns=['Data_Value_Footnote_Symbol', 'Data_Value_Footnote', 'Data_Value_Alt'])

# dropping regional aggregate rows
cog = cog[~cog["LocationAbbr"].isin(AGGREGATES)]

# if 'StratificationCategory2' == 'Overall', 'Stratification2' also == 'Overall'
cog['StratificationCategory2'] = cog['StratificationCategory2'].fillna('Overall')
cog['Stratification2'] = cog['Stratification2'].fillna('Overall')

# looking for rows with no confidence interval and dropping them
cog = cog.dropna(subset=['Low_Confidence_Limit', 'High_Confidence_Limit'])
print('final shape:', cog.shape)

# ~ categorization and clustering! ~
"""
originally, the pooling of multiple years was removed to adjust for skewed prediction results. 
removing overlapping studies across strata, clustering now only takes into account the comprehensive studies 
(listed as "Overall") across all unique values that can be inside a strata.
(i.e., Stratification1 covers "50-64 years", "65 years or older" individually and then also "Overall" studies across both).

Clustering now pools years from 2019-2022 across all states.
"""

pivot = cog[(cog['YearStart'] == POOL_START) & (cog['YearEnd'] == POOL_END)
            & (cog['Stratification1'] == 'Overall')
            & (cog['Stratification2'] == 'Overall')
]
pivot = pivot.pivot_table(index='LocationDesc', columns='Topic', values='Data_Value', aggfunc='mean').dropna()

# shortening pivot table column names
pivot = pivot.rename(columns=TOPIC_LABELS)

print('clustering table:', pivot.shape)
pivot_scaled = MinMaxScaler().fit_transform(pivot)

# compare k = 2 through 6 before choosing one
for k in range(2, 7):
    labels = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10).fit_predict(pivot_scaled)
    print('k =', 3, '| silhouette:', round(silhouette_score(pivot_scaled, labels), K))

km = KMeans(n_clusters=K, random_state=RANDOM_SEED, n_init=10)

pivot['cluster'] = km.fit_predict(pivot_scaled)

# due to output restrictions: saving every location's cluster and topic rates to a CSV
RESULTS_DIR.mkdir(exist_ok=True)
pivot.sort_values(["cluster", "LocationDesc"]).round(1).to_csv(CLUSTERS_CSV)
print('saved', len(pivot), 'locations to', CLUSTERS_CSV)

# ~ cluster table ~
print('cluster averages (%):')
print(pivot.groupby('cluster').mean(numeric_only=True).round(1))
print('cluster sizes:', pivot['cluster'].value_counts().sort_index().to_dict())

for c in sorted(pivot['cluster'].unique()):
    print('cluster', c, '->', list(pivot[pivot['cluster'] == c].index[:6]))

# ~ model training and testing ~
"""
model uses single-year rows only, eliminating overlapping studies DURING TRAINING,
model then tests on a year it hasn't seen. created model_df variable to better fork clustering and model from
shared cog df.
"""
model_df = cog[cog["YearStart"] == cog["YearEnd"]].copy()

preprocess = ColumnTransformer([
    ("categories", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ("year", "passthrough", ["YearStart"]),
])

# training on every year prior to the test year
train = model_df[model_df["YearStart"] < TEST_YEAR]
# test on TEST_YEAR only, about 11% of single-year rows
test = model_df[model_df["YearStart"] == TEST_YEAR]

X_train, y_train = train[FEATURES], train["Data_Value"]
X_test, y_test = test[FEATURES], test["Data_Value"]

# presentation prints
print('train rows:', len(train), '| test rows:', len(test))
print(f'models, tested on {TEST_YEAR}:')

# baseline -- predict each topic's average from the training years
topic_means = train.groupby("Topic")["Data_Value"].mean()
base_pred = test["Topic"].map(topic_means)
print('topic baseline -> r2:', round(r2_score(y_test, base_pred), 3), 'mae (pts):', round(mean_absolute_error(y_test, base_pred), 2))

lr = Pipeline([("prep", preprocess), ("model", LinearRegression())])
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
print('linear regression -> r2:', round(r2_score(y_test, lr_pred), 3), 'mae (pts):', round(mean_absolute_error(y_test, lr_pred), 2))

rf = Pipeline([
    ("prep", preprocess),
    ("model", RandomForestRegressor(n_estimators=200, max_depth=12, random_state=RANDOM_SEED, n_jobs=-1)),
]).fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print('random forest -> r2:', round(r2_score(y_test, rf_pred), 3), 'mae (pts):', round(mean_absolute_error(y_test, rf_pred), 2))

# after prediction per-topic MAE: average each model's absolute error within each topic
errors = pd.DataFrame({
    "Topic": test["Topic"].map(TOPIC_LABELS),
    "baseline": (base_pred - y_test).abs(),
    "linear regression": (lr_pred - y_test).abs(),
    "random forest": (rf_pred - y_test).abs(),
})
per_topic = errors.groupby("Topic").mean().round(2)
per_topic["test rows"] = errors["Topic"].value_counts()
print(f'MAE by topic (pts), tested on {TEST_YEAR}:')
print(per_topic.to_string())

"""
permutation importance: shuffle one feature at a time on the 2022 test set and
measure how much MAE increases. a bigger increase means the model relies on it more.
YearStart always scores 0 here, since every test row is from 2022.
"""
result = permutation_importance(
    rf, X_test, y_test,
    n_repeats=10,
    random_state=RANDOM_SEED,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
)
importances = pd.Series(result.importances_mean, index=FEATURES).sort_values(ascending=False)
print('permutation importance (MAE increase, pts):')
# hiding the "dtype: float64" pandas adds when printing the series
print(importances.round(2).to_string())


