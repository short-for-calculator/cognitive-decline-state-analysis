from main import (
    AGGREGATES,
    POOL_END,
    POOL_START,
    TEST_YEAR,
    clean,
    cluster_rows,
    model_rows,
    split_by_year,
)


# ~ clean() ~
def test_clean_keeps_only_cognitive_decline(raw_df):
    cog = clean(raw_df)
    assert (cog["Class"] == "Cognitive Decline").all()


def test_clean_removes_aggregate_locations(raw_df):
    cog = clean(raw_df)
    assert not cog["LocationAbbr"].isin(AGGREGATES).any()


def test_clean_drops_missing_values(raw_df):
    cog = clean(raw_df)
    assert cog["Data_Value"].notna().all()


def test_clean_fills_blank_stratification2(raw_df):
    cog = clean(raw_df)
    assert cog["Stratification2"].notna().all()
    assert "Overall" in set(cog["Stratification2"])


def test_clean_keeps_expected_row_count(raw_df):
    # 11 raw rows: 4 removed (2 aggregates, 1 missing value, 1 other class)
    assert len(clean(raw_df)) == 7


# ~ clustering branch ~
def test_cluster_rows_are_pooled_overall_only(raw_df):
    rows = cluster_rows(clean(raw_df))
    assert len(rows) == 1
    assert (rows["YearStart"] == POOL_START).all()
    assert (rows["YearEnd"] == POOL_END).all()
    assert (rows["Stratification1"] == "Overall").all()
    assert (rows["Stratification2"] == "Overall").all()


# ~ model branch ~
def test_model_rows_are_single_year_only(raw_df):
    rows = model_rows(clean(raw_df))
    assert (rows["YearStart"] == rows["YearEnd"]).all()
    assert len(rows) == 4


def test_model_rows_keep_subgroups(raw_df):
    rows = model_rows(clean(raw_df))
    assert "Female" in set(rows["Stratification2"])
    assert "65 years or older" in set(rows["Stratification1"])


def test_split_by_year_separates_train_and_test(raw_df):
    train, test = split_by_year(model_rows(clean(raw_df)), TEST_YEAR)
    assert (train["YearStart"] < TEST_YEAR).all()
    assert (test["YearStart"] == TEST_YEAR).all()
    assert len(train) + len(test) == 4