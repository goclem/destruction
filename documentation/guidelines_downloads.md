# Guidelines for downloading rasters

Updated 2022-05-09 

## Download

1. Go to the [allmapsoft website](http://www.allmapsoft.com/geid/) and download the latest version of google earth images downloader (updated weekly) and install.
2. Once installed go to `Help > Register` and introduce the following license code `JRZXC-NIIMW-JAKNT-BGGUS-VUXHW`.
3. To start a new task, select a new task name in `Task name`. Format the name as `image_YYYY_MM_DD`.
4. In `Maps type` select `Google Earth Satellite Imagery`.
5. Introduce the date of interest in the format `YYYY_MM_DD`. **Important**: The software will select the closest available tiles to the selected date. That means that for some regions in the area of interest some tiles can be from a different date than in other regions. Check that the whole urban area is from the same date in Google Earth.
6. Select zoom level from 19 to 19.
7. Define longitude and latitudes of interest. To do so you need to find the area of interest in Google Earth, or alternatively check the log files of the previous downloads in the folder `GEID/logs`.
8. Select destination folder.
9. Start.
10. Once the download task has finished go to `Tools > Map Combiner`.
11. Select the .geid project that you want to combine in a `.tif` file.
12. Start

## Format

Make sure that:

- The raster dimensions (height and width) are multiples of 128.
- The raster dimensions across dates are exactly the same.
- The raster is formatted as 8-bits unsigned integers.
- The raster is named `city/image_YYYY_MM_DD`

## Utilities

### Homs

*Coordinates*

- Left longitude  input= 36.6646
- Right longitude input= 36.7746
- Top latitude    input= 34.7738
- Bottom latitude input= 34.6771

*Downloads*

- 2013-09-26#1, found and download image of 2013-10-31
- 2014-04-21#2, found image for 2016/05/30

### Logs

*Coordinates*
 
- Left longitude  input= 36.612
- Right longitude input= 36.669
- Top latitude    input= 35.9506
- Bottom latitude input= 35.9056

*Downloads*

- 2013-09-15#1, found image of 2014-02-07,
- 2014-05-02#2, found and downloaded image of 2014-05-31,
- 2015-04-06#3, found and downloaded image of 2015-04-17,
- 2016-08-01#4, found and download image of 2016-08-01,

### Daraa

*Coordinates*

- Left longitude  input= 36.0689
- Right longitude input= 36.1388
- Top latitude    input= 32.66
- Bottom latitude input= 32.597

*Downloads*

- 2013-09-07#1, found and downloaded image of 2013-11-10,
- 2014-05-01#2, found and downloaded image of 2014-05-01,
- 2015-06-04#3, found image of 2016-02-25 (no imagery for 2015),
- 2016-04-19#4, found image of 2016-04-19,

### Raqqa

*Coordinates*

- Left longitude  input= 38.9344
- Right longitude input= 39.0604
- Top latitude    input= 35.9766
- Bottom latitude input= 35.9302

*Downloads*

- 2013-10-22#1, found image of 2014-03-21 (!),
- 2014-02-12#2, found image of 2014-03-21,
- 2015-05-29#3, found image of 2016-07-01 (!),

### Deir

*Coordinates*

- Left longitude  input= 40.1038
- Right longitude input= 40.1663
- Top latitude    input= 35.3602
- Bottom latitude input= 35.3009

*Downloads*

- 2013-10-24#1, found and downloaded image of 2013-10-24,
- 2014-05-13#2, found image of 2014-09-16,
- 2015-05-10#3, found image of 2016-04-17 (no images in 2015),
- 2016-05-10, found image of 2016-05-25 (!),
- 2016-05-25#4, found image of 2016-05-25,

### Hama

*Coordinates*

- Left longitude  input= 36.7196
- Right longitude input= 36.8023
- Top latitude    input= 35.1714
- Bottom latitude input= 35.0893

*Downloads*

- 2013-09-26#1, found image of 2013-10-31,
- 2014-03-05#2, found image of 2014-04-03,
- 2016-06-30, found image of 2016-06-30,
- 2016-07-06#3, found image of 2016-07-29.

### Damascus

*Coordinates*

- Left longitude  input= 36.2735
- Right longitude input= 36.5125
- Top latitude    input= 33.5763
- Bottom latitude input= 33.4380

*Downloads*

- 2012-10-10, found images of 2012-09-03, 2012-08-30
- 2013_02_21, found images of 2013-1-3, 2013-2-21
- 2015-09-26#1, found images of 2015-4-17, 2015-1-10, and 2014-12-28 (!)
- 2016-04-02#2, found images of 2016-1-12, 2016-2-29, 2014-12-28 (very few). 




