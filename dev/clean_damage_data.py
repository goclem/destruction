
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

new_df = []
for i, row in df.iterrows():

    row_entry = {}
    row_entry['geometry'] = row['geometry']
    
    for j, sensor_date_col in enumerate(sensor_date_columns):
        if(row[sensor_date_col] != None):
            row_entry[row[sensor_date_col]] = row[damage_class_columns[j]]
    
    new_df.append(row_entry)


df = gpd.GeoDataFrame(new_df)
print("Unique_values: {}".format(df[df.columns[1]].unique()))
df.to_file(INPUT_PATH.split(".gpkg")[0]+"__processed.gpkg", driver="GPKG")
