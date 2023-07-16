# Lightning

This is the historical lightining dataset we will use.

https://fire.ak.blm.gov/content/maps/aicc/Data/Data%20(zipped%20Text%20Files)/Historical_Lightning_as_txt.zip

https://fire.ak.blm.gov/content/maps/aicc/Data/Data%20(zipped%20Shapefiles)/CurrentYearLightning_SHP.zip

TODO:
- visualization to get the time where most lightnings occurs
- visualization get get which months most lightnings occurs
- need to convert or reproject to align 5km lightining data with WRF data
- repeat table 5 from paper lighting

## Variables to Extract from WRF

Four groups of variables were summarized from literature (Reap 1991, Burrows et al 2005, Sousa et al 2013, Blouin et al 2016), including stability indices, cloud properties, weather conditions at multiple pressure levels (500 hPa, 700 hPa, 850 hPa, and 1000 hPa), and two lightning parameterizations from WRF (table 4).

BT | Brightness temperature
CFtotal | Total cloud cover fraction
CFlow | Low-level cloud cover fraction
DZ700-850 | Thickness between any two pressure layers
GPZ500 | Geopotential height at multiple pressure levels
GPZ700 | Geopotential height at multiple pressure levels
GPZ850 | Geopotential height at multiple pressure levels
Helicity | Helicity
LCL | Lifted Condensation Level
PLI | Parcel Lifted Index (to 500 hPa)
PW | Precipitable water
Rain | Total precipitation
RH500 | Relative humidity at surface and multiple pressure levels
RH700 | Relative humidity at surface and multiple pressure levels
RH2 | Relative humidity at surface and multiple pressure levels
RH850 | Relative humidity at surface and multiple pressure levels
RH800 | Relative humidity at surface and multiple pressure levels
SHOW | Showalter Index
SLP | Sea level pressure
Td500  | Dewpoint temperature at surface and multiple pressure levels
Td2 | Dewpoint temperature at surface and multiple pressure levels
T2 | Air temperature at surface and multiple pressure levels
T750 | Air temperature at surface and multiple pressure levels
T500 | Air temperature at surface and multiple pressure levels
T850 | Air temperature at surface and multiple pressure levels
TT | Temperature-dewpoint spread at multiple pressure levels
W500 | Vertical velocity at multiple pressure levels



