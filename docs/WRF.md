# WRF

## Docker Container

1. Download NVIDIA HPC SDK container
```
singularity build --sandbox nvhpc-22.11-devel-cuda_multi-ubuntu20.04 docker://nvcr.io/nvidia/nvhpc:22.11-devel-cuda_multi-ubuntu20.04
```

## WRF Installation

1. Download and uncompress
```
wget https://github.com/wrf-model/WRF/releases/download/v4.4.2/v4.4.2.tar.gz
tar -zxvf v4.4.2.tar.gz
```

## WPS Installation

1. Download and uncompress
```
wget https://github.com/wrf-model/WPS/archive/refs/tags/v4.4.tar.gz
tar -zxvf v4.4.tar.gz
```

## Libraries Compatibility

```
wget https://www2.mmm.ucar.edu/wrf/OnLineTutorial/compile_tutorial/tar_files/Fortran_C_NETCDF_MPI_tests.tar
```

## Variables Documentation

 looks good - Helicity, SLP, 'GPZ500', 'TD500', 'RH500',
 'TD2', 'LCL', 'PW', 'RH2', 'RAINTotal'
 TT, TC500
 wrong - PLI
 maybe
   CFTotoal (wrong, local is 0, ours is higher)
   CFlow (wrong, local is 0, ours is higher)
   CFMed  (wrong, local is 0, ours is higher)
   CFHigh (wrong, local is 0, ours is higher)
   'GPZ750' no-data problems, kind of similar
   'GPZ700' no local data to compare to
   'RH700' no local data to compare to
   'GPZ850' no local data to compare to
   'SHOW' wrong because of no data
   'T500' no local data to compare to
    'RH800' wrong because of no data
    'T2' numbers look far away - our T2 is in Kelvin
    'RH850' no local data to compare to
    'T850' no local data to compare to
   'W500'
   'WA500' -looks good
  'DZ700_850' - looks good
 'BT' missing