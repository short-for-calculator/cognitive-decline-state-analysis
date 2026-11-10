import pandas as pd
import pytest

SCD = "Subjective cognitive decline or memory loss among older adults"
NEEDS_HELP = "Need assistance with day-to-day activities because of subjective cognitive decline or memory loss"


def make_row(location="Arizona", abbr="AZ", topic=SCD, start=2019, end=2019,
             age="Overall", demographic=None, value=10.0, cls="Cognitive Decline"):
    """One raw row shaped like the CDC spreadsheet, with only the columns main.py uses."""
    return {
        "Class": cls,
        "LocationDesc": location,
        "LocationAbbr": abbr,
        "Topic": topic,
        "YearStart": start,
        "YearEnd": end,
        "Stratification1": age,
        "StratificationCategory2": None if demographic is None else "Race/Gender",
        "Stratification2": demographic,
        "Data_Value": value,
        "Data_Value_Alt": value,
        "Low_Confidence_Limit": None if value is None else value - 2,
        "High_Confidence_Limit": None if value is None else value + 2,
        "Data_Value_Footnote_Symbol": None,
        "Data_Value_Footnote": None,
    }


@pytest.fixture
def raw_df():
    """A tiny raw dataset where each row tests one cleaning or filtering rule."""
    return pd.DataFrame([
        # kept: single-year rows across training years and the test year
        make_row(start=2019, end=2019),
        make_row(start=2021, end=2021, demographic="Female"),
        make_row(start=2022, end=2022, age="65 years or older"),
        make_row(location="Texas", abbr="TX", topic=NEEDS_HELP, start=2015, end=2015),
        # kept: pooled 2019-2022 rows, one Overall and one subgroup
        make_row(start=2019, end=2022),
        make_row(start=2019, end=2022, demographic="Male"),
        make_row(start=2019, end=2022, age="50-64 years"),
        # removed by clean(): aggregates, missing value, wrong class
        make_row(location="United States, DC & Territories", abbr="US"),
        make_row(location="South", abbr="SOU"),
        make_row(value=None),
        make_row(cls="Mental Health"),
    ])