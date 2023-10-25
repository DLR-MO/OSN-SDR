# OSN-SDR
Combining service difficulty reports with OpenSky network data

## Dowloading data

To get started download the service difficulty reports of your model of interest at https://sdrs.faa.gov/ and put them under:
/data/raw_sdrs/{model_name}

Then dowload the most recent OpenSky Aircraft database at https://opensky-network.org/datasets/metadata/ and put it in  under:
/data/

## Running the Code

### Augmenting SDRs with OSN AircraftDB
This script will go through all files in the folder with your model name, combine them together and augment them with information from the OSN aircraft database.

``
python combine_n_augment_sdrs.py --model {model_name}
``

