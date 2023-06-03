---
title: Alaska Wildfire Occurrence
emoji: 🔥
colorFrom: green
colorTo: red
sdk: docker
pinned: false
license: apache-2.0
app_port: 7860
---

# wildfire-occurrence

Wildfire occurrence modeling using Terrestrial Ecosystem Models and Artificial Intelligence

[CG Lightning Probability Forecast](https://jordancaraballo-alaska-wildfire-occurrence.hf.space/)

## Objectives

- Probabilistic wildfire occurrence model
- Model both occurrence, spread and risk of fire
- Create data pipeline between UAF TEM and NCCS/SMCE resources
- 30m local Alaska models, 1km circumpolar models
- Integration of precipitation, temperature and lightning datasets

## Datasets

1. Daily Fire Ignition Points

```bash
```

2. Daily Area Burned

The dataset comes from https://daac.ornl.gov/cgi-bin/dsviewer.pl?ds_id=1559 for 2001-2019. This dataset
will be extended for 2020-2025. Dataset is located under /explore/nobackup/projects/ilab/projects/LobodaTFO/data/raw_data/ABoVE_DoB.

```bash
python DAACDataDownload.py -dir /explore/nobackup/projects/ilab/projects/LobodaTFO/data/raw_data/ABoVE_DoB -f URL_FROM_ORDER
```

3. Annual Fuel Composition

```bash
```

4. Human Accesibility

```bash
```

5. Topographic Influence

```bash
```

All datasets described above will be delivered in the 1 km modeling grid for tundra ecoregions.

## Containers

### Python Container

```bash
module load singularity
singularity build --sandbox /lscratch/$USER/container/wildfire-occurrence docker://nasanccs/wildfire-occurrence:latest
```

## Extracting variables from WRF

```bash
singularity shell --nv -B /explore/nobackup/projects/ilab,/explore/nobackup/projects/3sl,$NOBACKUP,/lscratch,/explore/nobackup/people /lscratch/jacaraba/container/wildfire-occurrence/
python wrf_analysis.py 
```

## Dataset Generation and Training

```bash
singularity exec --env PYTHONPATH="/explore/nobackup/people/jacaraba/development/wildfire-occurrence" --nv -B /explore/nobackup/projects/ilab,/explore/nobackup/projects/3sl,$NOBACKUP,/lscratch,/explore/nobackup/people /lscratch/jacaraba/container/wildfire-occurrence python /explore/nobackup/people/jacaraba/development/wildfire-occurrence/wildfire_occurrence/model/lightning/lightning_model.py
```

(base) [jacaraba@gpu021 ~]$ singularity exec --env PYTHONPATH="/explore/nobackup/people/jacaraba/development/wildfire-occurrence" --nv -B /explore/nobackup/projects/ilab,/explore/nobackup/projects/3sl,$NOBACKUP,/lscratch,/explore/nobackup/people /lscratch/jacaraba/container/wildfire-occurrence python /explore/nobackup/people/jacaraba/development/wildfire-occurrence/wildfire_occurrence/model/lightning/lightning_model.py 


## Contributors

- Jordan Alexis Caraballo-Vega, jordan.a.caraballo-vega@nasa.gov