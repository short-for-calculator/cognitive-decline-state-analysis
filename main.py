# ~ pkg import list ~
import pandas as pd
import openpyxl
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

df = pd.read_excel('Alzheimers_Disease_dataset.xlsx')

# ~ part c: data normalization and cleaning! ~
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
AGGREGATES = ["US", "NRE", "MDW", "SOU", "WEST"]
cog = cog[~cog["LocationAbbr"].isin(AGGREGATES)]

# if 'StratificationCategory2' == 'Overall', 'Stratification2' also == 'Overall'
cog['StratificationCategory2'] = cog['StratificationCategory2'].fillna('Overall')
cog['Stratification2'] = cog['Stratification2'].fillna('Overall')

# looking for rows with no confidence interval and dropping them
cog = cog.dropna(subset=['Low_Confidence_Limit', 'High_Confidence_Limit'])
print('final shape:', cog.shape)

# turn the category columns into numbers so sklearn can use them - label encoding
cat_cols = ['LocationDesc', 'Topic', 'Stratification1', 'Stratification2']
for c in cat_cols:
    cog[c + '_enc'] = LabelEncoder().fit_transform(cog[c].astype(str))

# scale everything 0-1 so year doesn't dominate percentage or vice versa
num_cols = ['YearStart', 'Data_Value', 'Low_Confidence_Limit', 'High_Confidence_Limit']
cog[num_cols] = MinMaxScaler().fit_transform(cog[num_cols])

# ~ part c: categorization and clustering! ~
"""
originally, the pooling of multiple years was removed to adjust for skewed prediction results. 
removing overlapping studies across strata, clustering now only takes into account the comprehensive studies 
(listed as "Overall") across all unique values that can be inside a strata.
(i.e., Stratification1 covers "50-64 years", "65 years or older" individually and then also "Overall" studies across both).

Clustering now pools years from 2019 & 2022 across all states.
"""
pivot = cog[(cog['YearStart'] == 2019) & (cog['YearEnd'] == 2022)
            & (cog['Stratification1'] == 'Overall')
            & (cog['Stratification2'] == 'Overall')
]
pivot = pivot.pivot_table(index='LocationDesc', columns='Topic', values='Data_Value', aggfunc='mean').dropna()

pivot_scaled = MinMaxScaler().fit_transform(pivot)

km = KMeans(n_clusters=3, random_state=42, n_init=10)
pivot['cluster'] = km.fit_predict(pivot_scaled)

print(pivot.groupby('cluster').mean(numeric_only=True).round(1))
for c in sorted(pivot['cluster'].unique()):
    print('cluster', c, '->', list(pivot[pivot['cluster'] == c].index[:6]))

# ~ part d: prediction ~
"""
model uses single-year rows only, eliminating overlapping studies DURING TRAINING,
model then tests on a year it hasn't seen.
"""
model_df = cog[cog["YearStart"] == cog["YearEnd"]].copy()

features = [c + '_enc' for c in cat_cols] + ['YearStart']
X = cog[features]
y = cog['Data_Value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression().fit(X_train, y_train)
pred = lr.predict(X_test)
print('linear regression -> r2:', round(r2_score(y_test, pred), 3), 'mae:', round(mean_absolute_error(y_test, pred), 2))

rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1).fit(X_train, y_train)
pred = rf.predict(X_test)
print('random forest -> r2:', round(r2_score(y_test, pred), 3), 'mae:', round(mean_absolute_error(y_test, pred), 2))

print(pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False))


