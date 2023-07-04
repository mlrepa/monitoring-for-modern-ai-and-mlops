# Enhanced Machine Learning Model Monitoring with MLFlow and Evidently AI

This example shows steps to integrate Evidently and [MLFlow](https://mlflow.org/) into your ML prototypes and production pipelines.

![Evidently.ai + MLFlow](static/banner.png "Dashboard preview")

--------
Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── config             <- Configs directory
    ├── data
    │   ├── features       <- Features for model training and inference.
    │   ├── raw            <- The original, immutable data dump.
    │   └── reference      <- Reference datasets for monitoring.
    ├── fastapi            <- FastAPI application
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    ├── reports             <- Monitoring report files
    │
    ├── src                <- Source code for use in this project.
    │   ├── monitoring     <- Common code for monitoring 
    │   │
    │   ├── pipelines      <- Source code for all pipelines
    │   │
    │   ├── scripts        <- Helper scripts
    │   │
    │   ├── utils          <- Utility functions and classes 
    ├── static             <- Assets for docs 
    └── streamlit_app      <- Streamlit application
     


--------

## :woman_technologist: Installation

### 1. Fork / Clone this repository

Get the tutorial example code:

```bash
git clone git@github.com:evidentlyai/evidently.git
cd evidently/examples/integrations/mlflow_integration
```


### 2. Create a virtual environment

- This example requires Python 3.9 or above 

```bash
python3 -m venv .venv
echo "export PYTHONPATH=$PWD" >> .venv/bin/activate
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements-dev.txt
```


### 3. Setup Jupyter Notebooks (optional)

In case you are also interested in Evidently Dashboard visualization in Jupyter install jupyter nbextention:
```bash 
jupyter nbextension install --sys-prefix --symlink --overwrite --py evidently`
```

And activate it:
```bash 
jupyter nbextension enable evidently --py --sys-prefix
```
More details: https://docs.evidentlyai.com/install-evidently 


### 4 - Download data

This is a preparation step. Load data from [https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset) to the `data/` directory

```bash 
python src/pipelines/load_data.py              
```

## :tv: Pipelines and Monitoring dashboards


### 1 - Generate predictions


We prepared a script to generate predictions for the model. The scripts simulate a requests to the model and save predictions to PostgreSQL database.

```bash 
python src/scripts/simulate.py > simulation.log 2>&1
```

**Note**:
- `> simulation.log` means that all simulation script logs will be wrote in the file `simulation.log`
- `2>&1` means that ***all*** logs (including errors and non-stdout logs) will be redirect to `simulation.log`

### 2 - Monitoring reports

To generate and view monitoring reports open following endpoints:
- *model performance*: http://0.0.0.0:5000/monitor-model
- *target drift*: http://0.0.0.0:5000/monitor-target

<details><summary>Notes</summary>

- you can build report on different size of prediction data using parameter *`window_size`*, for instance:
    - http://0.0.0.0:5000/monitor-model?window_size=300
    - http://0.0.0.0:5000/monitor-target?window_size=100
- default value of *`window_size`* is *3000*

</details>

### 3 - Preview monitoring reports via Streamlit UI (optional)

Streamlit application implements convenient interface to build and render monitoring reports.

To render report:
- open [Streamlit application](http://localhost:8501)
- input required window size (options; 3000 by default)
- click on of two buttons (***Model performance*** or ***Target drift***) and wait report rendering


## :checkered_flag: Stop cluster

```bash
docker compose down
```

<details>
<summary>Notes</summary>

- To clear cluster one needs to remove `Docker` volume containing monitoring (`Postgres`) database
- It may be useful to run this tutorial from scratch
- Run the command:
  
```bash
docker compose down -v
```

</details>


## Acknowledgments

The dataset used in the example is from: https://www.kaggle.com/c/bike-sharing-demand/data?select=train.csv
Fanaee-T, Hadi, and Gama, Joao, 'Event labeling combining ensemble detectors and background knowledge', Progress in Artificial Intelligence (2013): pp. 1-15, Springer Berlin Heidelberg
More information about the dataset can be found in UCI machine learning repository: https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset