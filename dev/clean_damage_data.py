
import geopandas as gpd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input_file_path", help="Path to the input raw damage .gpkg")

args = parser.parse_args()
INPUT_PATH = args.input_file_path

df = gpd.read_file(INPUT_PATH)

sensor_date_columns = [col for col in df.columns if 'SensDt' in col]
damage_class_columns = [col for col in df.columns if 'DmgCls' in col and 'GrpDmgCls' not in col ]

for i, sensor_date_col in enumerate(sensor_date_columns):
    if i==0:
        allDates = df[sensor_date_col]
    else:
        allDates = allDates.append(df[sensor_date_col])
        
sensor_date_values = allDates.unique()
sensor_date_values = sensor_date_values[sensor_date_values != np.array(None)]
        
for i, damage_class_col in enumerate(damage_class_columns):
    if i==0:
        allClasses = df[damage_class_col]
    else:
        allClasses = allClasses.append(df[damage_class_col])
        
class_values = allClasses.unique()
print("All classes: {}".format(class_values.unique()))


for i, date in enumerate(sensor_date_values):
    df[date] = df[damage_class_columns[i]]
    
df = df[[*sensor_date_values, 'geometry']]
df.to_file(INPUT_PATH.split(".gpkg")[0]+"__processed.gpkg", driver="GPKG")
