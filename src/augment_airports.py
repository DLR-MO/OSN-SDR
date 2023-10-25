import pandas as pd
from timezonefinder import TimezoneFinder
from combine_n_augment_sdr import DATA_FOLDER

airports_df = pd.read_csv(DATA_FOLDER / "airports.csv")
tz_finder = TimezoneFinder()
airports_df["Timezone"] = airports_df.apply(lambda row:
                                            tz_finder.timezone_at(lng=row["longitude_deg"], lat=row["latitude_deg"]),
                                            axis=1)
airports_df.to_csv(DATA_FOLDER / "augmented_airports.csv")
