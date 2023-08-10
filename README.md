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

[![DOI](https://zenodo.org/badge/545456432.svg)](https://zenodo.org/badge/latestdoi/545456432)

[CG Lightning Probability Forecast](https://huggingface.co/spaces/jordancaraballo/alaska-wildfire-occurrence)

## Tutorial Exercises

| Lecture Topic | Interactive Link | 
|---|---|
| **Python Spatial Visualization** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nasa-nccs-hpda/wildfire-occurrence/blob/main/notebooks/intern/LightningVisualization.ipynb) |
| **Python ALDN Validation** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nasa-nccs-hpda/wildfire-occurrence/blob/main/notebooks/intern/ALDN-Validation.ipynb) |

## Objectives

- Probabilistic wildfire occurrence model
- Model both occurrence, spread and risk of fire
- Create data pipeline between UAF TEM and NCCS/SMCE resources
- 30m local Alaska models, 1km circumpolar models
- Integration of precipitation, temperature and lightning datasets

## Containers

### Python Container

```bash
module load singularity
singularity build --sandbox /lscratch/$USER/container/wildfire-occurrence docker://nasanccs/wildfire-occurrence:latest
```

## Quickstart

### Running WRF

```bash
conda activate ilab-pytorch; PYTHONPATH="/explore/nobackup/people/$USER/development/wildfire-occurrence" python /explore/nobackup/people/$USER/development/wildfire-occurrence/wildfire_occurrence/view/wrf_pipeline_cli.py -c /explore/nobackup/people/$USER/development/wildfire-occurrence/wildfire_occurrence/templates/config.yaml --start-date 2023-06-06 --forecast-lenght 10 --pipeline-step all
```

## Extracting variables from WRF

Running this script to extract variables from WRF and perform lightning inference

```bash
singularity shell --nv -B /explore/nobackup/projects/ilab,/explore/nobackup/projects/3sl,$NOBACKUP,/lscratch,/explore/nobackup/people /lscratch/jacaraba/container/wildfire-occurrence/
python wrf_analysis.py 
```

## Generate output variables

```bash
singularity exec --env PYTHONPATH="/explore/nobackup/people/jacaraba/development/wildfire-occurrence" --nv -B /explore/nobackup/projects/ilab,/explore/nobackup/projects/3sl,$NOBACKUP,/lscratch,/explore/nobackup/people /lscratch/jacaraba/container/wildfire-occurrence python /explore/nobackup/people/$USER/development/wildfire-occurrence/wildfire_occurrence/view/wrf_pipeline_cli.py -c /explore/nobackup/people/$USER/development/wildfire-occurrence/wildfire_occurrence/templates/config.yaml --start-date 2023-06-29 --forecast-lenght 10 --pipeline-step postprocess
```

## Generate output variables

```bash
singularity exec --env PYTHONPATH="/explore/nobackup/people/jacaraba/development/wildfire-occurrence" --nv -B /explore/nobackup/projects/ilab,/explore/nobackup/projects/3sl,$NOBACKUP,/lscratch,/explore/nobackup/people /lscratch/jacaraba/container/wildfire-occurrence python /explore/nobackup/people/$USER/development/wildfire-occurrence/wildfire_occurrence/view/lightning_pipeline_cli.py -c /explore/nobackup/people/$USER/development/wildfire-occurrence/wildfire_occurrence/templates/config.yaml --pipeline-step preprocess
```

## Dataset Generation and Training

```bash
singularity exec --env PYTHONPATH="/explore/nobackup/people/jacaraba/development/wildfire-occurrence" --nv -B /explore/nobackup/projects/ilab,/explore/nobackup/projects/3sl,$NOBACKUP,/lscratch,/explore/nobackup/people /lscratch/jacaraba/container/wildfire-occurrence python /explore/nobackup/people/jacaraba/development/wildfire-occurrence/wildfire_occurrence/model/lightning/lightning_model.py
```

Full Data Pipeline Command

```bash
singularity exec --env PYTHONPATH="/explore/nobackup/people/jacaraba/development/wildfire-occurrence" --nv -B /explore/nobackup/projects/ilab,/explore/nobackup/projects/3sl,$NOBACKUP,/lscratch,/explore/nobackup/people /lscratch/jacaraba/container/wildfire-occurrence python /explore/nobackup/people/jacaraba/development/wildfire-occurrence/wildfire_occurrence/model/lightning/lightning_model.py 
```

## Contributors

- Jordan Alexis Caraballo-Vega, jordan.a.caraballo-vega@nasa.gov
