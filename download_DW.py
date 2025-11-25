import argparse
from pathlib import Path

import ee
import eemont  # noqa: F401
import geemap
import geopandas as gpd
import pandas as pd
from tqdm import tqdm

from utils import (
    calc_dw_aggregate_v2,
    calculate_data_area,
    chunk_list,
    create_dw_classes_mask,
    drop_z_from_gdf,
    make_date_window,
    weekly_dates,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process Dynamic World data for inundation analysis."
    )
    parser.add_argument(
        "--input-vector",
        type=Path,
        required=True,
        help="Path to input vector file - must be polygon vector",
    )
    parser.add_argument(
        "--ee-project",
        type=str,
        required=True,
        help="Earth Engine project ID (required)",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=7,
        help="Step size in days (default: 7)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=7,
        help="Window size in days (default: 7)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/output_v2/annual"),
        help="Output directory (default: data/output_v2/annual)",
    )
    parser.add_argument(
        "--name-attribute",
        type=str,
        default="Name",
        help="Name attribute column (default: Name)",
    )
    parser.add_argument(
        "--year-start",
        type=int,
        default=2017,
        help="Start year (default: 2017)",
    )
    parser.add_argument(
        "--year-end",
        type=int,
        default=2025,
        help="End year (default: 2025)",
    )
    parser.add_argument(
        "--season-start",
        type=str,
        default="05-01",
        help="Season start date MM-DD (default: 05-01)",
    )
    parser.add_argument(
        "--season-end",
        type=str,
        default="10-31",
        help="Season end date MM-DD (default: 10-31)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # setup input args
    INPUT_VECTOR = args.input_vector
    STEPSIZE = args.step_size
    WINDOWSIZE = args.window_size
    OUTPUT_DIR = args.output_dir
    NAME_ATTRIBUTE = args.name_attribute
    START_YEAR = args.year_start
    END_YEAR = args.year_end
    SEASON_START = args.season_start
    SEASON_END = args.season_end
    EE_PROJECT = args.ee_project

    # initialize ee project
    geemap.ee_initialize(project=EE_PROJECT)

    # create output dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # read vector data
    gdf = gpd.read_file(INPUT_VECTOR)

    # create gee feature collection
    fc = geemap.gdf_to_ee(drop_z_from_gdf(gdf[:]))

    # config
    band_names = [
        "water",
        "trees",
        "grass",
        "flooded_vegetation",
        "crops",
        "shrub_and_scrub",
        "built",
        "bare",
        "snow_and_ice",
    ]
    feature_index_name = NAME_ATTRIBUTE
    reducer = ee.Reducer.sum()
    CRS = "EPSG:3572"
    SCALE = 10
    reducer_dict = {
        "reducer": reducer,
        "collection": fc.select(feature_index_name),
        "crs": CRS,
        "scale": SCALE,
        "bands": band_names,
    }

    for year in tqdm(range(START_YEAR, END_YEAR + 1)):
        print(f"Processing year: {year}")
        try:
            df_list = []
            date_list = weekly_dates(
                f"{year}-{SEASON_START}", end_date=f"{year}-{SEASON_END}", step=STEPSIZE
            )
            date_chunks = chunk_list(date_list, 4)

            # for dates in tqdm(date_chunks):
            for dates in tqdm(date_chunks[:]):
                imlist = []
                for date in tqdm(dates, desc="Processing dates"):
                    try:
                        window = make_date_window(date, WINDOWSIZE, mode="each")
                        start_date = window["start_date"]
                        end_date = window["end_date"]
                        # calculate class mode values
                        # single date
                        im = calc_dw_aggregate_v2(
                            start_date=start_date,
                            end_date=end_date,
                            timestamp_date=date,
                            polygons=fc,
                        )
                        if im is None:
                            print(f"No data for date: {date}")
                            continue
                        # append to list
                        im_classes = create_dw_classes_mask(ee.Image(im))
                        imlist.append(im_classes)
                    except Exception as e:
                        continue

                #  make a collection from class images
                ic_classes = ee.ImageCollection(imlist)

                # run extraction
                fc_out = ic_classes.getTimeSeriesByRegions(**reducer_dict)

                # convert FeatureCollection with values to pandas
                df_out = geemap.ee_to_df(fc_out)

                df_list.append(df_out)

            df_merged = pd.concat(df_list, ignore_index=True)

            # calculate data areas
            ds = df_merged.set_index([NAME_ATTRIBUTE, "date"]).to_xarray()
            df_final = calculate_data_area(ds).to_dataframe().reset_index()

            # save output
            df_final.to_csv(OUTPUT_DIR / f"output_{year}_{WINDOWSIZE}dayAgg.csv")
            df_final.to_parquet(OUTPUT_DIR / f"output_{year}_{WINDOWSIZE}dayAgg.parquet")

        except Exception as e:
            print(e)
            print(f"Error processing year: {year}")
            continue


if __name__ == "__main__":
    main()
