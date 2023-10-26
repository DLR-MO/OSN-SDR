# SPDX-FileCopyrightText: 2023 Emy Arts <emy.arts@dlr.de>
# SPDX-FileCopyrightText: 2023 German Aerospace Center
#
# SPDX-License-Identifier: MIT

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
tqdm.pandas()
import warnings
from combine_n_augment_sdr import DATA_FOLDER
warnings.filterwarnings("ignore")
#%%

SDR_PATH = DATA_FOLDER / "augmented_sdrs"
STORE_PATH = SDR_PATH /  "sdrs_with_layover"
LAYOVER_STORE_PATH = DATA_FOLDER / "layovers" / "aircraft_layovers"
# STORE_PATH = Path("sdr")
# LAYOVER_STORE_PATH = Path("layover")
OSN_PATH = DATA_FOLDER / "raw_osn_ac_data"
AIRPORTS_DF = pd.read_csv(DATA_FOLDER / "augmented_airports.csv")
STORE_PATH.mkdir(exist_ok=True)
LAYOVER_STORE_PATH.mkdir(parents=True, exist_ok=True)

def get_layover_df(ac_icao):
    store_path = LAYOVER_STORE_PATH / f"layovers_{ac_icao}.csv"
    if store_path.exists():
        return pd.read_csv(store_path)
    path = OSN_PATH / f"flight_list_{ac_icao}.csv"
    if path.exists():
        df = pd.read_csv(path)
        df.loc[:, 'firstseen'] = pd.to_datetime(df['firstseen'], errors='coerce')
        df.loc[:, 'lastseen'] = pd.to_datetime(df['lastseen'], errors='coerce')
        df['midflighttime'] = df['firstseen'] + (df['lastseen'] - df['firstseen'])/2
        # df['midflighttime'] = df[['firstseen', 'lastseen']].mean(axis=1)
        df.sort_values(by="midflighttime", inplace=True)
        df.reset_index(inplace=True, drop=True)
        df.loc[:, 'departure'] = df['departure'].astype(str)
        df.loc[:, 'arrival'] = df['arrival'].astype(str)
        df.loc[:, 'day'] = pd.to_datetime(df['day'], errors='coerce')
        entries = []
        for i, row in df.iterrows():
            if i > 0:
                row_prev = df.iloc[i-1]
                no_nan_airport = not (row['departure'] == 'nan' or row_prev['arrival'] == 'nan')
                if not(no_nan_airport and row['departure'] != row_prev["arrival"]):
                    if row['departure'] == 'nan':
                        layover_ap = row_prev['arrival']
                    else:
                        layover_ap = row['departure']
                    try:
                        tz = AIRPORTS_DF[AIRPORTS_DF['ident'] == layover_ap]['Timezone'].values[0]
                        local_dt = pd.to_datetime(row_prev['lastseen'], utc=True).tz_convert(tz)
                        layover_start = pd.to_datetime(local_dt.date()).timestamp()
                        local_timestamp = local_dt.timestamp()
                    except:
                        local_timestamp = np.nan
                        layover_start = row_prev['day'].timestamp()
                    entries.append({
                        'LayoverAirport': layover_ap,
                        'LayoverStartTime': row_prev['lastseen'].timestamp(),
                        'LayoverEndTime': row['firstseen'].timestamp(),
                        'LayoverStartDay': layover_start,
                        'LayoverLocalStartTime': local_timestamp,
                        'LayoverEndDay': row['day'].timestamp(),
                        'LayoverHours': (row['firstseen'].timestamp() - row_prev['lastseen'].timestamp()) / 3600,
                        'LayoverNumber': len(entries),
                        'FlightNumber': i
                    })
        ac_layover_df = pd.DataFrame(entries)
        ac_layover_df.to_csv(store_path, index=False)
        return ac_layover_df
    else:
        return pd.DataFrame()


def find_layover_at(layover_df, unix_day):
    if unix_day < layover_df['LayoverStartDay'].min():
        return [None for col in layover_df.columns] + [0]
    else:
        masks = [(layover_df['LayoverStartDay'] <= unix_day) & (unix_day <= layover_df['LayoverEndDay']),
                 (layover_df['LayoverStartDay'] < unix_day) & (unix_day <= layover_df['LayoverEndDay']),
                 (layover_df['LayoverStartDay'] <= unix_day) & (unix_day < layover_df['LayoverEndDay']),
                 (layover_df['LayoverStartDay'] < unix_day) & (unix_day < layover_df['LayoverEndDay'])
                 ]
        mask_sums = [sum(mask) for mask in masks]
        if sum(mask_sums) == 0:
            return [None for col in layover_df.columns] + [0]
        elif 1 in mask_sums:
            return layover_df[masks[mask_sums.index(1)]].values[0].tolist() + [1]
        else:
            mask = masks[[i for i, sum in enumerate(mask_sums) if sum > 0][-1]]
            possible_layovers = layover_df[mask]
            return possible_layovers.values[possible_layovers["LayoverHours"].argmax()].tolist() + [len(possible_layovers)]

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--model", help="Give aircraft model name as used sdr folders")

    args = argParser.parse_args()
    model = args.model

    print(model)
    sdr_df = pd.read_csv(SDR_PATH / f"{model}.csv")
    sdr_df["DifficultyDate"] = pd.to_datetime(sdr_df["DifficultyDate"])
    sdr_df['DifficultyDateUNIX'] = sdr_df['DifficultyDate'].map(pd.Timestamp.timestamp).astype(float)


    count = 0
    df_list = []
    total_acs = len(sdr_df['icao24'].unique())
    print(f"{total_acs} total aircraft, {sdr_df.shape[0]} total reports to consider.")
    for icao, sdr_group in sdr_df.groupby('icao24'):
        count = count + 1
        layover_df = get_layover_df(icao)
        if not layover_df.empty and not sdr_group.empty:
            print(icao)
            sdr_group[layover_df.columns.tolist() + ["LayoverCandidates"]] = sdr_group.progress_apply(
                lambda row: find_layover_at(layover_df, row['DifficultyDateUNIX']), result_type='expand', axis=1)
            df_list.append(sdr_group)
            print("layovers:", layover_df.shape[0], "sdrs:",
                  f"{sdr_group['LayoverHours'].dropna().shape[0]}/{sdr_group.shape[0]}\nprocessed {count}/{total_acs} aircraft")
    if len(df_list) > 0:
        df = pd.concat(df_list)
        df.to_csv(STORE_PATH / f"{model}.csv", index=False)
    else:
        print(f"No reports out of {sdr_df.shape[0]} could be connected to layovers")

