import pandas as pd
from pyopensky.impala import Impala
import pickle
from combine_n_augment_sdr import DATA_FOLDER
from datetime import datetime

with open(DATA_FOLDER / "icao_list.pkl", 'rb') as f:
    icao_codes = pickle.load(f)


opensky = Impala()
for icao in icao_codes:
    print(icao)
    flight_list = opensky.flightlist(icao24 = icao,
                                   start=int(pd.to_datetime("2016/1/1").timestamp()),
                                   stop=int(datetime.now().timestamp()))
    flight_list.to_csv(DATA_FOLDER / "raw_osn_ac_data" / f"flight_list_{icao}.csv", index=False)

#
# for code in icao_codes:
#     query = f"SELECT * FROM flights_data4 WHERE icao24 = '{code}';"
#     flight_list = opensky.rawquery(query)
#     flight_list.to_csv(f"data2/flight_list_{code}.csv", index=False)