import geopandas as gpd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input_file_path", help="Path to the input raw damage .gpkg")

args = parser.parse_args()
INPUT_PATH = args.input_file_path

# INPUT_PATH = "/Users/arogyak/projects/mwd/data/daraa/others/daraa_settlement.gpkg"
df = gpd.read_file(INPUT_PATH)
df = df.filter(['geometry'])
df.to_file(INPUT_PATH.split(".gpkg")[0]+"__processed.gpkg", driver="GPKG")
