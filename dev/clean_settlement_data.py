import geopandas as gpd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--suffix", help="Suffix to add to output filename (default: '')")


args = parser.parse_args()
INPUT_PATH = args.input_file_path
SUFFIX = ""

if args.suffix:
    SUFFIX = args.suffix


# INPUT_PATH = "/Users/arogyak/projects/mwd/data/daraa/others/daraa_settlement.gpkg"
df = gpd.read_file(INPUT_PATH)
df = df.filter(['geometry'])
df.to_file(INPUT_PATH.split(".gpkg")[0] + SUFFIX + ".gpkg", driver="GPKG")
