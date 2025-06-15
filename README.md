# smap-loss-functions

Work with SMAP loss functions describing soil moisture loss vs. time due to evapotranspiration and runoff.

This package fits, plots, and projects soil moisture based on data from the NASA **Soil Moisture Active Passive (SMAP)** mission
using an approach developed by Koster et al
(2017; [doi:10.1175/jhm-d-16-0285.1](https://journals.ametsoc.org/view/journals/hydr/18/3/jhm-d-16-0285_1.xml)). SMAP
is designed to measure the amount of water in the top 5 cm of soil everywhere on Earth's surface every 2-3 days.
SMAP loss functions as defined in the paper are simple functions to describe the dynamics in soil moisture driven by
precipitation, evapotranspiration, and runoff, enabling short-term forecasts of soil drydown.

## Features

This package provides a script with subcommands to:
1. Choose a rectangular region of the geospatial grid used by SMAP to cover a region-of-interest
2. Export soil moisture from a SMAP HDF5 file to a raster template created in step (1).
3. Convert GPM IMERG data to a GeoTIFF on the same grid, enabling further calculations.
3. Dump data from EASE GeoTIFFs (2 and 3) to text files.
4. Write SMAP and IMERG text data (3) to an SQLite database.
5. Fit SMAP loss functions to SMAP and precipitation data (in 4).
6. Forecast SMAP soil moisture over 5 days with zero precipitation (based on 5).
7. Plot a loss function from a single grid cell---either the function, or a simulation against data.

## Limitations

The basic functionality of the package has only been applied to SMAP L3 9 km data (spl3smp_e) and SMAP L2 36 km data (spl2smp) and daily GPM IMERG daily 0.1 degree grid data.

## Installation

To install dependencies:

```bash
pip install numpy scipy netCDF4 h5py pyproj matplotlib gdal ease_lonlat
```

For development, clone the repository and install in editable mode:

```bash
git clone [https://github.com/alex-cobb/smap-loss-functions.git](https://github.com/alex-cobb/smap-loss-functions.git)
cd smap_loss_functions
pip install -e .
```

## Usage

This package provides a command-line interface (CLI). You can run the main script and its various subcommands directly from your terminal.

To see subcommands and global options, run:

```bash
smap-loss-functions --help
```

Each subcommand has its own help message describing its arguments and options:

```bash
smap-loss-functions <subcommand_name> --help
```

## Mini-tutorial
Here is a sketch of how the subcommands in this package can used in combination with other tools to analyze and visualize SMAP data.  This example will use SMAP L3 9 km data (`spl3smp_e`) with GPM IMERG daily data over the island of Sumatra.

First, identify the region-of-interest:
```bash
echo '{"srs": "EPSG:4326", "w": 95, "n": 6, "e": 106.1, "s": -5.85}' > roi.json
```
Now determine the subset of the the EASE 2.0 global 9 km grid used by SMAP:
```bash
smap-loss-functions choose-ease-grid roi.json smap_grid.json --grid-name EASE2_G9km \
  --cols smap_cols.tif --rows smap_rows.tif
```
This command writes the grid details to `smap_grid.json` and creates two raster files that show the EASE 2.0 grid cells covering the region of interest.  You can quickly view them with QGIS:
```bash
qgis smap_cols.tif
```

Next, extract SMAP data from an HDF5 file obtained from NASA (for example, from EarthData Search) into GeoTIFF rasters using the index rasters created in the last step:
```bash
smap-loss-functions smap-to-raster-template smap_cols.tif smap_rows.tif SMAP_L3_SM_P_E_20190723_R19240_001_HEGOUT.h5 SMAP_L3_SM_P_E_20190723_R19240_001_HEGOUT.tif
```
Repeat this for all the relevant SMAP data files and write all those file names to a file (for example, `smap_tif_names.txt`; `ls -1 SMAP_L3_SM_P_E_*.tif > smap_tif_names.txt`), then dump all their data to a text file with:
```bash
smap-loss-functions dump-ease-raster-data smap_cols.tif smap_rows.tif -f smap_tif_names.txt -o smap_data.txt
```
At this stage, you can also see the SMAP data easily, for example with `qgis SMAP_L3_SM_P_E_20190723_R19240_001_HEGOUT.tif`.  Or, to easily view data from consecutive files,
convert them to pseudocolor images first using the `write-colormap` and gdaldem from [GDAL](https://gdal.org/):
```bash
smap-loss-functions write-colormap plasma_r 0.0 1.0 -o smap_colormap.txt
gdaldem color-relief -of PNG -alpha SMAP_L3_SM_P_E_20190723_R19240_001_HEGOUT.tif smap_colormap.txt SMAP_L3_SM_P_E_20190723_R19240_001_HEGOUT.png
```
... and then view them or animate them.

Now prepare the IMERG data.  To convert GPM IMERG precipitation NetCDF4 data to GeoTIFF format, use
```bash
smap-loss-functions imerg-to-geotiff 3B-DAY.MS.MRG.3IMERG.20200101-S000000-E235959.V07B.nc4 3B-DAY.MS.MRG.3IMERG.20200101-S000000-E235959.V07B.tif
```
The resulting GeoTIFF (view it! `qgis 3B-DAY.MS.MRG.3IMERG.20200101-S000000-E235959.V07B.tif`) will be in geographic coordinates (EPSG:4326).  To transform it to the same grid as the SMAP data (by averaging), use
```bash
gdalwarp -t_srs EPSG:6933 -te 9161176.799999997 -747663.9191981069 10242142.8 765688.4791804515 -ts 120 168 -te_srs EPSG:6933 -r average 3B-DAY.MS.MRG.3IMERG.20200101-S000000-E235959.V07B.tif 3B-DAY.MS.MRG.3IMERG.20200101-S000000-E235959.V07B_ease.tif
```
The arguments for the extent (`-te`) and numbers of rows and columns (`-ts`) are obtainable from the raster grid description generated earlier (`smap_grid.json`).  This command will carry through metadata, including the datetime range of data collection (view with `gdalinfo 3B-DAY.MS.MRG.3IMERG.20200101-S000000-E235959.V07B_ease.tif`).  The warped and unwarped GPM IMERG data can also be converted to pseudocolor images for viewing using the same procedure as described above (`write-colormap`).

As with the SMAP data, write all the warped GPM IMERG filenames to a file (here, `imerg_ease_tif_names.txt`; for example, `ls -1 3B-DAY.MS.MRG.3IMERG.*_ease.tif > imerg_ease_tif_names.txt`) and dump all the data as text:
```bash
smap-loss-functions dump-ease-raster-data smap_cols.tif smap_rows.tif -f imerg_ease_tif_names.txt -o imerg_data.txt
```

Now, write all the SMAP and IMERG data to an SQLite database for further processing:
```bash
smap-loss-functions write-smap-db smap_data.txt imerg_data.txt smap_data.sqlite3
```
Then fit the loss functions, writing the loss function parameters for each grid cell to an SQLite database called `loss_functions.sqlite3`:
```bash
smap-loss-functions fit-loss-functions smap_data.sqlite3 loss_functions.sqlite3
```
The loss functions can now be visualized using `smap-loss-functions plot-loss-function`.  To see the fit between SMAP data and a simulation using the loss function parameters from grid column 2948, row 734:
```bash
smap-loss-functions plot-loss-function loss_functions.sqlite3 2948 734 --against-smap-db smap_data.sqlite3
```
Or, to see just the loss function:
```bash
smap-loss-functions plot-loss-function loss_functions.sqlite3 2948 734
```
To find valid cell columns and rows for your dataset, refer to the index rasters (`smap_col.tif` and `smap_row.tif` in this example), for example by using `gdalinfo -mm` or by inspecting them in `qgis`.  Or, enter an invalid column and row and look at the valid ranges in the error message.

The fitted loss functions can now be used for forecasts that assume zero rain (writing here to an output database called `smap_forecast.sqlite3`):
```bash
smap-loss-functions forecast-smap loss_functions.sqlite3 smap_data.sqlite3 smap_forecast.sqlite3
```
The forecasts will extend 5 days from the last valid value in each grid cell.

Here, `smap_data.sqlite3` can be any appropriately formatted database providing initial conditions for the forecasts.  The database only needs to contain a `smap_data` table defined as follows:
```sql
CREATE TABLE smap_data (
  start_datetime timestamp NOT NULL,
  thru_datetime timestamp NOT NULL,
  ease_col integer NOT NULL,
  ease_row integer NOT NULL,
  soil_moisture real NOT NULL,
  PRIMARY KEY (start_datetime, ease_col, ease_row)
);
```

GeoTIFFs of soil moistures from the last 5 days in the forecast database can be produced with
```bash
smap-loss-functions write-forecast-geotiffs smap_forecast.sqlite3 smap_cols.tif smap_rows.tif SMAP_forecast_{}.tif
```
This command will write SMAP GeoTIFFs with names `SMAP_forecast_<date>.tif` for the last forecast datetime (five days after the last observation used to produce the forecast) and each of the preceding four days at the same time.  Because each forecast begins with observational data, this means that the first "forecast" is really a nowcast, showing a combination of SMAP data from the last observation and forecasts, using the fit loss functions, in cells with no observational data on that date.

## Differences between this example and the methods in Koster et al (2017)

1. The example above assumes the SMAP L3 9 km product; Koster et al (2017) worked with L2 36 km data.
2. In Koster et al (2017), the parameters `nd` in equation 3 is set to one day (24 h) such that the maximum
   infiltration rate is "the rate [that] if it were to be applied over a full
   day... would exceed the current soil water deficit".  Because the time step for simulations is 1 h,
   this implies that soil moisture cannot reach full saturation (Wmax).  In
   some settings, this may not be realistic and so the default for this parameter is 1.0, which simply
   limits infiltration to prevent the soil moisture from exceeding Wmax.  The original behavior from Koster
   et al (2017) can be recovered by setting max_infiltration_h to 24.
3. Koster et al (2017) converted precipitation to the SMAP grid using `areal weighting` (p. 838).  The example
   above uses the GDAL tool `gdalwarp` with `-r average` which probably has different behavior.

## Running Tests

To run the tests, navigate to the root directory of the project and execute pytest:

```bash
pytest
```

## References

The calculations in this package are based on the following paper:

  * Koster, R. D., Reichle, R. H., & Mahanama, S. P. P. (2017). A Data-Driven Approach for Daily Real-Time Estimates and Forecasts of Near-Surface Soil Moisture. Journal of Hydrometeorology, 18(3), 837–843. [https://doi.org/10.1175/jhm-d-16-0285.1](https://journals.ametsoc.org/view/journals/hydr/18/3/jhm-d-16-0285_1.xml)

SMAP L3 soil moisture data can be found via this DOI:

  * O'Neill, P. E., Chan, S., Njoku, E. G., Jackson, T., Bindlish, R. & Chaubell,
    J. (2023). SMAP L2 Radiometer Half-Orbit 36 km EASE-Grid Soil Moisture. (SPL2SMP,
    Version 9). Boulder, Colorado USA. NASA National Snow and Ice Data Center
    Distributed Active Archive
    Center. [doi:10.5067/K7Y2D8QQVZ4L](https://doi.org/10.5067/K7Y2D8QQVZ4L).

The package scripts facilitate work with daily precipitation data from GPM IMERG:

  * Huffman, G.J., E.F. Stocker, D.T. Bolvin, E.J. Nelkin, Jackson Tan (2023), GPM IMERG
    Final Precipitation L3 1 day 0.1 degree x 0.1 degree V07, Edited by Andrey
    Savtchenko, Greenbelt, MD, Goddard Earth Sciences Data and Information Services
    Center (GES
    DISC). [doi:10.5067/GPM/IMERGDF/DAY/07](https://doi.org/10.5067/GPM/IMERGDF/DAY/07).

## Acknowledgments

This code was initially developed by Alex Cobb, Raymond Samalo and Liu Junzhe with
support from the National Research Foundation Singapore through the Singapore-MIT
Alliance for Research and Technology’s Center for Environmental Sensing and Modeling
interdisciplinary research program and from the Office for Space Technology and Industry
(OSTIn), Singapore's national space office, through its Space Technology Development
Programme (Grant No. S22-02007-STDP).

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the `LICENSE` file for
details.
