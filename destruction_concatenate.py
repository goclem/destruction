from destruction_utilities import *
from pathlib import Path

CITIES = ['aleppo', 'damascus', 'daraa', 'deir-ez-zor', 'hama', 'homs', 'idlib', 'raqqa']

def concat(cities, suffix, shape, path = '../data'):
    dest = f"{path}/all/others"
    path = Path(dest)
    path.mkdir(parents=True, exist_ok=True)
    delete_zarr_if_exists('all', suffix)
    zarr.save(f"{dest}/all_{suffix}.zarr", np.empty((0,*shape)))
    for city in CITIES:
        print(f'------ Starting concatenation operation for {city} / suffix: {suffix}')
        path = f'../data/{city}/others/{city}_{suffix}.zarr'
        z = zarr.open(path)
        f = zarr.open(f"{dest}/all_{suffix}.zarr", mode='a')
        idx_array= make_tuple_pair(z.shape[0], 7500)
        for idx_range in idx_array:
            print(f"--------- {idx_range}")
            k = z[0:1000][:]
            f.append(k)

concat(CITIES, 'images_conv_train_balanced', (128,128,3))
concat(CITIES, 'images_conv_valid', (128,128,3))
concat(CITIES, 'images_conv_test', (128,128,3))
concat(CITIES, 'labels_conv_train_balanced', (1,1,1))
concat(CITIES, 'labels_conv_valid', (1,1,1))
concat(CITIES, 'labels_conv_test', (1,1,1))