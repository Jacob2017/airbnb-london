import pandas as pd
from random import Random
from datetime import date
import os


def main(use_calendar: bool, use_listings: bool):
    # if not os.path.exists("data/available_dates.csv"):
    #     df = pd.read_csv("data/calendar.csv")
    #     print(len(df))
    #     available_df = df[df["available"] == "t"]
    #     available_df.to_csv("data/available_dates.csv", index=False)
    #     print("filtered to available dates only")
    #     return
    pd.set_option("display.max_columns", None)
    if use_calendar:
        df = pd.read_csv("data/calendar.csv")
        print(len(df))
        df = clean_calendar_data(df)
        print(df.describe())

        r = Random()

        random_rows = [r.randint(0, len(df)) for x in range(25)]
        print(df.iloc[random_rows])
        print(df.dtypes)

    if use_listings:
        df = pd.read_csv("data/listings.csv")
        print(len(df))
        with pd.option_context("display.max_rows", None):
            print(df.dtypes)
        # df = clean_calendar_data(df)
        with pd.option_context("display.max_rows", 25, "display.max_columns", None):
            print(df.describe())

        r = Random()

        random_rows = [r.randint(0, len(df)) for x in range(25)]
        with pd.option_context("display.max_rows", 25, "display.max_columns", None):
            print(df.iloc[random_rows])


def filter_to_available(df) -> pd.DataFrame:
    pass


def clean_calendar_data(df: pd.DataFrame) -> pd.DataFrame:
    df["price_float"] = df["price"].apply(clean_currency).astype(float)
    df["adjusted_price_float"] = df["adjusted_price"].apply(clean_currency).astype(float)
    df["date"] = pd.to_datetime(df["date"])
    try:
        df["date"] = df["date"].dt.date
    except Exception as e:
        print("Data Cleaning Exception:")
        print(e)
    df["available"] = (df["available"] == "t").astype(int)
    return df


def clean_currency(x):
    if isinstance(x, str):
        return x.replace("$", "").replace(",", "")
    return x


if __name__ == "__main__":
    main(False, True)
