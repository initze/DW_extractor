# Tools to extract Dynamic World data for dynamic temporal windows

## Installation

* uv recommended
`uv sync`

## Dependencies

* earthengine-api
* geemap
* eemont
* geopandas
* tqdm
* numpy
* matplotlib
* seaborn

## Usage

* download Class areas [ha] of dynamic world for each item of your vector file
  * every 7 days
  * dominant class per pixel in 7 day window (date +- 3 days)
  * Year 2024, from 01 August to 15 September

```bash
uv run download_DW.py --input-vector YOUR-FILE.gpkg --ee-project YOUR-EEPROJECT --step-size 7 --window-size 7 --output-dir data/output_v3_3daystep/annual --name-attribute Name --year-start 2024 --year-end 2024 --season-start 08-01 --season-end 09-15
```
