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