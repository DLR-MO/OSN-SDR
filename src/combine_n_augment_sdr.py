# SPDX-FileCopyrightText: 2023 Emy Arts <emy.arts@dlr.de>
# SPDX-FileCopyrightText: 2023 German Aerospace Center
#
# SPDX-License-Identifier: MIT

import numpy as np
from pathlib import Path
import pandas as pd
import argparse
import shutil
from tqdm import tqdm
import pickle
tqdm.pandas()

DATA_FOLDER = Path.cwd()
if DATA_FOLDER.name == 'src':
    DATA_FOLDER = DATA_FOLDER.parent
DATA_FOLDER = DATA_FOLDER / "data"

SDR_COLS = ['DifficultyDate',
            'JASCCode',
            'RegistryNNumber',
            'AircraftMake',
            'AircraftModel',
            'AircraftSerialNumber',
            'Discrepancy',
            'AircraftTotalTime',
            'AircraftTotalCycles',
            'PartLocation',
            'PartCondition',
            'PartMake',
            'PartName',
            'PartNumber',
            'NatureOfConditionA',
            'StageOfOperationCode',
            'HowDiscoveredCode',
            'SubmitterTypeCode'
            ]
META_RENAME = {
    'registration': 'RegistryNNumber',
    'manufacturericao': 'AircraftManufacturer',
    'model': 'AircraftModel',
    'serialnumber': 'AircraftSerialNumber',
    'owner': 'AircraftOwner',
    'built': 'ManufactureYear',
    'engines': 'EngineModel',
    'icao24': 'icao24'
}

META_COLS = list(META_RENAME.values())
def augment_meta(df, row):
    """
    :param df: dataframe for look up
    :param row: [registry_n, model, serial_number]
    :return:
    """

    reg_mask = df['RegistryNNumber'] == row['RegistryNNumber']
    model_mask = df['AircraftModel'] == row['AircraftModel']
    serial_mask = df['AircraftSerialNumber'] == row['AircraftSerialNumber']

    match_1 = reg_mask | model_mask | serial_mask
    if match_1.sum() > 0:
        match_2 = (reg_mask & model_mask) | (reg_mask & serial_mask) | (model_mask & serial_mask)
        if match_2.sum() > 0:
            match_3 = reg_mask & model_mask & serial_mask
            if match_3.sum() > 0:
                return df[match_3].values[0]
            else:
                return df[match_2].values[0]
        else:
            match_1 = reg_mask | serial_mask
            if match_1.sum() > 0:
                return df[match_1].values[0]
            else:
                return [np.nan for i in META_COLS]
    else:
        return [np.nan for i in META_COLS]


def get_sdr_df(path, osn_df, name):
    print(name)
    df = pd.read_html(path)[0]
    df = df[SDR_COLS]
    df.dropna(subset=['RegistryNNumber', 'AircraftSerialNumber'], inplace=True, how='all')
    df['AircraftSerialNumber'] = df['AircraftSerialNumber'].astype(str)
    # df['DifficultyDate'] = pd.to_datetime((df['DifficultyDate']))
    df[META_COLS] = df.progress_apply(lambda row: augment_meta(osn_df, row),
                                      result_type='expand', axis=1)
    df.to_csv(str(temp_data_store_path / name[:-3]) + "csv", index=False)
    return df


if __name__ == '__main__':

    argParser = argparse.ArgumentParser()
    argParser.add_argument("--model", help="Give aircraft model name as used in raw_sdrs folder")

    args = argParser.parse_args()
    model = args.model

    data_download_path = DATA_FOLDER / "raw_sdrs" / model
    data_store_path = DATA_FOLDER / "augmented_sdrs"
    temp_data_store_path = DATA_FOLDER / "temp"
    temp_data_store_path.mkdir(parents=False, exist_ok=True)
    data_store_path.mkdir(parents=False, exist_ok=True)

    osn_df = pd.read_csv(DATA_FOLDER /  "aircraftDatabase.csv")
    osn_df.rename(columns=META_RENAME, inplace=True)
    osn_df = osn_df[META_COLS]
    osn_df.dropna(subset=["RegistryNNumber"], inplace=True)
    osn_df = osn_df[osn_df["RegistryNNumber"].str.startswith('N')]
    if model.startswith('A'):
        osn_df = osn_df[osn_df['AircraftManufacturer'] == 'AIRBUS']
    else:
        osn_df = osn_df[osn_df['AircraftManufacturer'] == 'BOEING']
    osn_df.loc[:, "RegistryNNumber"] = osn_df["RegistryNNumber"].str.lstrip('N')
    osn_df.loc[:, "RegistryNNumber"] = osn_df["RegistryNNumber"].str.replace('-', '')
    osn_df.loc[:, 'AircraftOwner'] = osn_df['AircraftOwner'].str.upper()
    osn_df.loc[:, 'AircraftModel'] = osn_df['AircraftModel'].str.replace('-', '')
    osn_df.loc[:, 'AircraftModel'] = osn_df['AircraftModel'].str.replace(' ', '')
    osn_df.loc[:, 'ManufactureYear'] = osn_df['ManufactureYear'].apply(lambda x: pd.to_datetime(x).year)

    row_select = ['RegistryNNumber', 'AircraftModel', 'AircraftSerialNumber']

    dfs = []
    for path in data_download_path.iterdir():
        if path.is_file() and path.suffix == '.xls':
            df = get_sdr_df(path, osn_df, path.name)
            dfs.append(df)
        elif path.is_dir():
            for sub_path in path.iterdir():
                if sub_path.is_file() and sub_path.suffix == '.xls':
                    df = get_sdr_df(sub_path, osn_df, (path.name + sub_path.name))
                    dfs.append(df)

    complete_df = pd.concat(dfs)
    complete_df['AircraftOwner'] = complete_df['AircraftOwner'].str.replace(' INC', '')
    complete_df['AircraftOwner'] = complete_df['AircraftOwner'].str.replace(' N A ', ' NA ')
    complete_df['AircraftOwner'] = complete_df['AircraftOwner'].str.replace('U S ', 'US ')
    complete_df['AircraftOwner'] = complete_df['AircraftOwner'].str.replace('JET BLUE', 'JETBLUE')
    complete_df['AircraftOwner'] = complete_df['AircraftOwner'].str.replace(' CORPORATION', '')
    complete_df['AircraftOwner'] = complete_df['AircraftOwner'].str.replace(' CORP', '')
    complete_df['AircraftOwner'] = complete_df['AircraftOwner'].str.replace(' CO ', ' ')
    complete_df['AircraftOwner'] = complete_df['AircraftOwner'].str.replace(' COMPANY ', ' ')

    complete_df.to_csv(data_store_path / f"{model}.csv", index=False)

    with open(DATA_FOLDER / "icao_list.pkl", 'wb') as f:
        pickle.dump(list(complete_df['icao24'].unique()), f)
    shutil.rmtree(temp_data_store_path)
