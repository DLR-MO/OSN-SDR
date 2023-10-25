# OSN-SDR
Combining service difficulty reports with OpenSky network data

## Dowloading data

To get started download the service difficulty reports of your model of interest at https://sdrs.faa.gov/ and put them under:
/data/raw_sdrs/[model_name]

Then dowload the most recent OpenSky Aircraft database at https://opensky-network.org/datasets/metadata/ and put it in  under:
/data/

Finally from https://ourairports.com/data/ download the "airports.csv" file to obtain a list of airports and their coordinates.
add this file under /data/

## Running the Code

### Augmenting SDRs with OSN AircraftDB
This script will go through all files in the folder with your model name, combine them together and augment them with information from the OSN aircraft database.

``
python combine_n_augment_sdrs.py --model [model_name]
``

### Downloading flights of aircraft with reports
To download the flights from the OpenSky network the pyopensky library is used.
Please follow the installation instructions at https://github.com/open-aviation/pyopensky.
Note: this script takes a long time to execute and downloads large amounts of data.

``
python dowload_flights.py
``

### Add airport timezones
To add the timezone information to each airport run

``
python augment_airports.py
``

### Identify Layovers and add them to the SDRs

``
python combine_sdr_osn.py --model [model_name]
``

### Prepare normalised dataset
Prepare a standardised datasets based on the desired features from the layover information.
A training and test set are created with a 90%-10% split.

``
python prepare_datasets.py
``
