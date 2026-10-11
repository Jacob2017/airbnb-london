import pandas as pd


def main():
    df = pd.read_csv("data/calendar.csv")
    print(len(df))
    print(df.describe())


if __name__ == "__main__":
    main()
