import pandas as pd
import numpy as np
from pathlib import Path
import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
from plot_utils import show_2d_hist, show_feature_density
from combine_n_augment_sdr import DATA_FOLDER

report_path = DATA_FOLDER / "augmented_sdrs" / "sdrs_with_layover"

features_path = DATA_FOLDER / "feature_files"
ds_path = DATA_FOLDER / "datasets"
ds_path.mkdir(exist_ok=True, parents=True)
fig_path = DATA_FOLDER.parent / "figs"
fig_path.mkdir(exist_ok=True)


def create_layover_df(layovers_folder, layover_df_path, reports_df):
    print("Creating layover dataframe")
    layovers = []

    def age_at_layover(unix, manu_year):
        ts = pd.to_datetime(unix, unit='s', utc=True)
        years = ts.year - manu_year
        age_month = 12 * years + ts.month
        return age_month

    for file in tqdm.tqdm(list(layovers_folder.iterdir())):
        if file.name.endswith(".csv"):
            l_df = pd.read_csv(file)
            l_df["icao24"] = file.name.split("_")[1].split(".")[0]
            layovers.append(l_df)

    layovers_df = pd.concat(layovers)

    reports_df["Inspection"] = reports_df["StageOfOperationCode"] == "IN"
    df_grouped = reports_df.groupby(["icao24", "LayoverNumber"])
    df_layover_grouped = df_grouped.size().reset_index().rename(columns={0: "Reports"})
    df_layover_grouped = df_layover_grouped.merge(df_grouped["Inspection"].agg(lambda x: x.any()).reset_index(),
                                                  on=["icao24", "LayoverNumber"])
    df_layover_grouped = df_layover_grouped.merge(df_grouped["AircraftOwner"].first().reset_index(),
                                                  on=["icao24", "LayoverNumber"])
    df_layover_grouped = df_layover_grouped.merge(df_grouped["ManufactureYear"].median().reset_index(),
                                                  on=["icao24", "LayoverNumber"])
    df_layover_grouped = df_layover_grouped.merge(df_grouped["AircraftTotalCycles"].median().reset_index(),
                                                  on=["icao24", "LayoverNumber"])
    layovers_df = layovers_df.merge(df_layover_grouped,
                                    on=["icao24", "LayoverNumber"],
                                    how="left")
    layovers_df = layovers_df.merge(airports_df[["ident", "Timezone"]], left_on="LayoverAirport", right_on="ident")
    layovers_df["Inspection"].fillna(False, inplace=True)
    layovers_df["Reports"].fillna(0, inplace=True)
    layovers_df["AgeAtLayover"] = layovers_df.apply(lambda row:
                                                 age_at_layover(row["LayoverStartTime"], row["ManufactureYear"]),
                                                 axis=1)
    layovers_df.to_csv(layover_df_path, index=False)
    return layovers_df


def prepare_dataset(layovers_df, feature_columns):

    def local_time_feature(local_start_time):
        dt_series = pd.to_datetime(local_start_time, unit='s')
        total_hours = dt_series.dt.hour + ((dt_series.dt.minute * 60 + dt_series.dt.second) / (60 * 60))
        feature = total_hours
        return feature

    layover_hours_cap = 7 * 24
    max_layover_age = layovers_df["AgeAtLayover"].max()

    all_features_dfs = []
    stds = []

    for icao, ac_layovers_df in layovers_df.groupby("icao24"):
        if not ac_layovers_df.empty:
            ac_layovers_df = ac_layovers_df.sort_values(by="LayoverStartTime").reset_index(drop=True)
            if "AircraftTotalCycles" in feature_columns:
                if ac_layovers_df[ac_layovers_df["Inspection"]]["AircraftTotalCycles"].notna().sum() < 3:
                    break
                else:
                    flight_number_offset = ac_layovers_df[ac_layovers_df["Inspection"] & ac_layovers_df["AircraftTotalCycles"].notna()]
                    stds.append((flight_number_offset["AircraftTotalCycles"] - flight_number_offset["FlightNumber"]).std())
                    flight_number_offset = (flight_number_offset["AircraftTotalCycles"] - flight_number_offset["FlightNumber"]).median()
                    ac_layovers_df["FlightNumber"] = flight_number_offset + ac_layovers_df["FlightNumber"]
                    stds.append(
                        (flight_number_offset["AircraftTotalCycles"] - flight_number_offset["FlightNumber"]).std())
            if "AgeAtLayover" in feature_columns:
                features_df.loc["AgeAtLayover"] = features_df["AgeAtLayover"] / max_layover_age
            features_df = ac_layovers_df[feature_columns]
            features_df.loc[:, "LayoverLocalStartTime"] = local_time_feature(features_df["LayoverLocalStartTime"])
            features_df.loc[:, "LayoverHours"] = features_df["LayoverHours"].clip(0, layover_hours_cap)
            features_path.mkdir(exist_ok=True)
            features_df.to_csv(features_path / f"{icao}.csv", index=False)
            all_features_dfs.append(features_df)

    all_features_df = pd.concat(all_features_dfs).dropna(axis=0)
    show_2d_hist(all_features_df, "LayoverLocalStartTime", "LayoverHours", save=fig_path / "FeatureDist2D.pdf")
    for col in all_features_df.columns[:-1]:
        show_feature_density(all_features_df, col)
    n = all_features_df.shape[0]
    print(f"Total datapoints: {n}")
    mask = np.full(n, True)
    rand = random.sample(range(n), int(n * 0.1))
    mask[rand] = False
    train_df = all_features_df[mask]
    train_df.to_csv(ds_path / "svm_train_set.csv", index=False)
    print("Stored training file")
    test_df = all_features_df[~mask]
    test_df.to_csv(ds_path / "svm_test_set.csv", index=False)
    print("Stored test file")
    return stds


if __name__ == '__main__':
    df = []
    for report in report_path.iterdir():
        df.append(pd.read_csv(report))

    df = pd.concat(df)
    df['DifficultyDate'] = pd.to_datetime(df['DifficultyDate'])
    airports_df = pd.read_csv(DATA_FOLDER / "augmented_airports.csv")
    layovers_folder = DATA_FOLDER / "layovers" / "aircraft_layovers"
    layover_df_path = DATA_FOLDER / "layovers" / "all_layovers.csv"

    if not layover_df_path.exists():
        layovers_df = create_layover_df(layovers_folder, layover_df_path, df)
    else:
        layovers_df = pd.read_csv(layover_df_path)

    feature_columns = ["LayoverLocalStartTime",
                       "LayoverHours",
                       # "FlightNumber",
                       # "AgeAtLayover",
                       "Inspection"]

    stds = prepare_dataset(layovers_df, feature_columns)

