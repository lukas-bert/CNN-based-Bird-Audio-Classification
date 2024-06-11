# imports
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import time
from maad import util
import warnings
# suppress all warnings
warnings.filterwarnings("ignore")

# parse arguments (storage dir)
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('path_storage', help = 'Path to save dataset to')
parser.add_argument('dataset_name', help = 'Name of the dataset. Will create subfolder with dataset_name')
parser.add_argument('-o', '--overwrite', action='store_true',
                    help='overwrite storage folder if existing')

args = parser.parse_args()

# Configurations 
class Config():
    def __init__(self, min_length = 5, 
                 max_length = 300, 
                 min_quality = 'B', 
                 recording_type = "song",
                 min_sr = 24000,
                 max_other_birds = 2,
                 min_files = 10,
                 max_files = 30,
                 max_nb_classes = 10
                   ):
        
        self.min_length = min_length
        self.max_length = max_length
        self.min_quality = min_quality
        self.recording_type = recording_type
        self.min_sr = min_sr
        self.max_other_birds = max_other_birds
        self.min_files = min_files
        self.max_files = max_files
        self.max_nb_classes = max_nb_classes

cfg = Config()

# Function to convert recording time format to seconds
def convert_to_seconds(time_str):
    parts = list(map(int, time_str.split(':')))
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1]*60 + parts[2]
    else:
        return float('NaN')

# List of bird species in europe:

data = [ # 100 common bird species in central europe
    ['Turdus merula', 'Common Blackbird', 'Amsel'],
    ['Erithacus rubecula', 'European Robin', 'Rotkehlchen'],
    ['Passer domesticus', 'House Sparrow', 'Haussperling'],
    ['Parus major', 'Great Tit', 'Kohlmeise'],
    ['Cyanistes caeruleus', 'Blue Tit', 'Blaumeise'],
    ['Sylvia atricapilla', 'Eurasian Blackcap', 'Mönchsgrasmücke'],
    ['Fringilla coelebs', 'Common Chaffinch', 'Buchfink'],
    ['Pica pica', 'Eurasian Magpie', 'Elster'],
    ['Sturnus vulgaris', 'European Starling', 'Star'],
    ['Columba palumbus', 'Common Wood Pigeon', 'Ringeltaube'],
    ['Passer montanus', 'Eurasian Tree Sparrow', 'Feldsperling'],
    ['Troglodytes troglodytes', 'Eurasian Wren', 'Zaunkönig'],
    ['Chloris chloris', 'European Greenfinch', 'Grünfink'],
    ['Dendrocopos major', 'Great Spotted Woodpecker', 'Großer Buntspecht'],
    ['Sitta europaea', 'Eurasian Nuthatch', 'Kleiber'],
    ['Turdus philomelos', 'Song Thrush', 'Singdrossel'],
    ['Carduelis carduelis', 'European Goldfinch', 'Stieglitz'],
    ['Hirundo rustica', 'Barn Swallow', 'Rauchschwalbe'],
    ['Phylloscopus collybita', 'Common Chiffchaff', 'Zilpzalp'],
    ['Streptopelia decaocto', 'Eurasian Collared Dove', 'Türkentaube'],
    ['Apus apus', 'Common Swift', 'Mauersegler'],
    ['Lophophanes cristatus', 'European Crested Tit', 'Haubenmeise'],
    ['Garrulus glandarius', 'Eurasian Jay', 'Eichelhäher'],
    ['Aegithalos caudatus', 'Long-tailed Tit', 'Schwanzmeise'],
    ['Pyrrhula pyrrhula', 'Eurasian Bullfinch', 'Gimpel'],
    ['Prunella modularis', 'Dunnock', 'Heckenbraunelle'],
    ['Sylvia communis', 'Common Whitethroat', 'Dorngrasmücke'],
    ['Phylloscopus trochilus', 'Willow Warbler', 'Fitis'],
    ['Spinus spinus', 'Eurasian Siskin', 'Erlenzeisig'],
    ['Turdus iliacus', 'Redwing', 'Rotdrossel'],
    ['Corvus corone', 'Carrion Crow', 'Rabenkrähe'],
    ['Corvus frugilegus', 'Rook', 'Saatkrähe'],
    ['Corvus monedula', 'Eurasian Jackdaw', 'Dohle'],
    ['Strix aluco', 'Tawny Owl', 'Waldkauz'],
    ['Athene noctua', 'Little Owl', 'Steinkauz'],
    ['Buteo buteo', 'Common Buzzard', 'Mäusebussard'],
    ['Accipiter nisus', 'Eurasian Sparrowhawk', 'Sperber'],
    ['Falco tinnunculus', 'Common Kestrel', 'Turmfalke'],
    ['Anas platyrhynchos', 'Mallard', 'Stockente'],
    ['Anser anser', 'Greylag Goose', 'Graugans'],
    ['Cygnus olor', 'Mute Swan', 'Höckerschwan'],
    ['Motacilla alba', 'White Wagtail', 'Bachstelze'],
    ['Anthus pratensis', 'Meadow Pipit', 'Wiesenpieper'],
    ['Regulus regulus', 'Goldcrest', 'Wintergoldhähnchen'],
    ['Phoenicurus ochruros', 'Black Redstart', 'Hausrotschwanz'],
    ['Luscinia megarhynchos', 'Common Nightingale', 'Nachtigall'],
    ['Oriolus oriolus', 'Eurasian Golden Oriole', 'Pirol'],
    ['Emberiza citrinella', 'Yellowhammer', 'Goldammer'],
    ['Turdus viscivorus', 'Mistle Thrush', 'Misteldrossel'],
    ['Cuculus canorus', 'Common Cuckoo', 'Kuckuck'],
    ['Falco peregrinus', 'Peregrine Falcon', 'Wanderfalke'],
    ['Falco subbuteo', 'Eurasian Hobby', 'Baumfalke'],
    ['Asio otus', 'Long-eared Owl', 'Waldohreule'],
    ['Bubo bubo', 'Eurasian Eagle-Owl', 'Uhu'],
    ['Alcedo atthis', 'Common Kingfisher', 'Eisvogel'],
    ['Columba livia', 'Rock Pigeon', 'Stadttaube'],
    ['Carduelis spinus', 'Eurasian Siskin', 'Erlenzeisig'],
    ['Plegadis falcinellus', 'Glossy Ibis', 'Sichler'],
    ['Phoenicurus phoenicurus', 'Common Redstart', 'Gartenrotschwanz'],
    ['Larus ridibundus', 'Black-headed Gull', 'Lachmöwe'],
    ['Larus canus', 'Common Gull', 'Sturmmöwe'],
    ['Larus argentatus', 'European Herring Gull', 'Silbermöwe'],
    ['Chroicocephalus genei', 'Slender-billed Gull', 'Dünnschnabelmöwe'],
    ['Pernis apivorus', 'European Honey Buzzard', 'Wespenbussard'],
    ['Milvus milvus', 'Red Kite', 'Rotmilan'],
    ['Milvus migrans', 'Black Kite', 'Schwarzmilan'],
    ['Accipiter gentilis', 'Northern Goshawk', 'Habicht'],
    ['Aquila chrysaetos', 'Golden Eagle', 'Steinadler'],
    ['Haliaeetus albicilla', 'White-tailed Eagle', 'Seeadler'],
    ['Pandion haliaetus', 'Osprey', 'Fischadler'],
    ['Circus cyaneus', 'Northern Harrier', 'Kornweihe'],
    ['Circus aeruginosus', 'Western Marsh Harrier', 'Rohrweihe'],
    ['Circus pygargus', 'Montagu s Harrier', 'Wiesenweihe'],
    ['Pluvialis apricaria', 'European Golden Plover', 'Goldregenpfeifer'],
    ['Vanellus vanellus', 'Northern Lapwing', 'Kiebitz'],
    ['Gallinago gallinago', 'Common Snipe', 'Bekassine'],
    ['Tringa totanus', 'Common Redshank', 'Rotschenkel'],
    ['Actitis hypoleucos', 'Common Sandpiper', 'Flussuferläufer'],
    ['Philomachus pugnax', 'Ruff', 'Kampfläufer'],
    ['Numenius arquata', 'Eurasian Curlew', 'Großer Brachvogel'],
    ['Limosa limosa', 'Black-tailed Godwit', 'Uferschnepfe'],
    ['Scolopax rusticola', 'Eurasian Woodcock', 'Waldschnepfe'],
    ['Charadrius dubius', 'Little Ringed Plover', 'Flussregenpfeifer'],
    ['Charadrius hiaticula', 'Common Ringed Plover', 'Sandregenpfeifer'],
    ['Larus marinus', 'Great Black-backed Gull', 'Mantelmöwe'],
    ['Sterna hirundo', 'Common Tern', 'Flussseeschwalbe'],
    ['Chlidonias niger', 'Black Tern', 'Trauerseeschwalbe'],
    ['Chlidonias leucopterus', 'White-winged Tern', 'Weißflügelseeschwalbe'],
    ['Ciconia ciconia', 'White Stork', 'Weißstorch'],
    ['Ciconia nigra', 'Black Stork', 'Schwarzstorch'],
    ['Ardea cinerea', 'Grey Heron', 'Graureiher'],
    ['Ardea alba', 'Great Egret', 'Silberreiher'],
    ['Ardeola ralloides', 'Squacco Heron', 'Rallenreiher'],
    ['Nycticorax nycticorax', 'Black-crowned Night Heron', 'Nachtreiher'],
    ['Ixobrychus minutus', 'Little Bittern', 'Zwergdommel'],
    ['Botaurus stellaris', 'Eurasian Bittern', 'Rohrdommel'],
    ['Pelecanus crispus', 'Dalmatian Pelican', 'Krauskopfpelikan'],
    ['Pelecanus onocrotalus', 'Great White Pelican', 'Rosapelikan']
]

df_species = pd.DataFrame(data,columns =['scientific name', 'english name', 'german name'])

gen = []
sp = []
for name in df_species['scientific name']:
    gen.append(name.rpartition(' ')[0])
    sp.append(name.rpartition(' ')[2])

min_quality_ = {"A": "B",
                "B": "C",
                "C": "D",
                "D": "E"}

# Query xeno canto API
df_query = pd.DataFrame() # save querries as dataframe
df_query['param1'] = gen # select only wanted species
df_query['param2'] = sp
df_query['param3'] = f'type:{cfg.recording_type}' # type of recording should include "song"
#df_query['param4'] ='area:europe' # only recordings from europe
df_query['param5'] = f'len:{cfg.min_length}-{cfg.max_length}' # only recordings longer than 5 seconds, shorter than 5 minutes
df_query['param6'] = f'q:">{min_quality_[cfg.min_quality]}"' # only recordings of quality 'A' or 'B'

# Get recordings metadata corresponding to the query

print("Reading metadata from xeno-canto.org. \n This can take a few minutes.")

df_dataset= util.xc_multi_query(df_query,
                                 format_time = False,
                                 format_date = False,
                                 verbose=True)

print(f"Total files: {len(df_dataset)}")

# Drop low sample rate recordings 
df_dataset["smp"] = df_dataset["smp"].astype(int)
print(f"Dropping {len(df_dataset[df_dataset['smp'] < cfg.min_sr])} recordings with sample rate sr < {cfg.min_sr}.")
df_dataset = df_dataset[df_dataset["smp"] >= cfg.min_sr]

# Calculate number of other species in background:
df_dataset["nr other"] = df_dataset["also"].apply(lambda x: len(x))

# Drop recordings with too many other species in background
print(f"Dropping {len(df_dataset[df_dataset['nr other'] > cfg.max_other_birds])} recordings with more than {cfg.max_other_birds} other species in background.")
df_dataset = df_dataset[df_dataset["nr other"] <= cfg.max_other_birds]

# Drop species with too low available recordings
counts = df_dataset.groupby("en")["en"].count().reset_index(name='count').sort_values("count", ascending = False)[:cfg.max_nb_classes]
remaining_species = counts[counts["count"] > cfg.min_files]["en"].tolist()

df_dataset = df_dataset[df_dataset["en"].apply(lambda x: x in remaining_species)]

print(f"Dropping species with less than {cfg.min_files} recordings. \n",
      f"{len(remaining_species)} species remaining. ({len(df_dataset)} files remaining).")

df_dataset['length_seconds'] = df_dataset['length'].apply(convert_to_seconds)

# Download files
print("Proceeding to download files...")

df_dataset = util.xc_selection(df_dataset,
                               max_nb_files = cfg.max_files,
                               #max_length='05:00', # redundant
                               #min_length='00:10',
                               min_quality = cfg.min_quality,
                               verbose = True )

print(f"Downloading {len(df_dataset)} files... This may take a few minutes.")
print(f"Overwriting existing folder = {args.overwrite}")

df_dataset = util.xc_download(df_dataset,
                 rootdir = args.path_storage,
                 dataset_name = args.dataset_name,
                 overwrite = args.overwrite,
                 save_csv= True,
                 verbose = True)

df_path = os.path.join(args.path_storage, f"{args.dataset_name}.csv")
print(f"Saving dataframe. ({len(df_dataset)} entries in total.)")
df_dataset.to_csv(df_path, index = False)
print("Finisehd.")
