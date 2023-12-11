# Get Started with ML Monitoring


## :woman_technologist: Installation

### 1. Fork / Clone this repository

Get the tutorial example code:

```bash
git clone https://gitlab.com/risomaschool/tutorials-raif/monitoring-1-get-started.git
cd monitoring-1-get-started
```


### 2. Create a virtual environment

- This example requires Python 3.9 or above 

```bash
python3 -m venv .venv
echo "export PYTHONPATH=$PWD" >> .venv/bin/activate
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 3 - Download data

This is a preparation step. Load data from [https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset) to the `data/` directory

```bash 
python src/load_data.py              
```

## :tv: Monitoring Examples

| No. | Monitoring Example | Description |
|---|---|---|
| 1. | **1-getting-started-tutorial.ipynb** | Get started with Evidently Monitoring |
| 2. | **2-monitor-model.ipynb**| Model Monitoring with Evidently and MLFlow |
| 3. | **3-data-drift.ipynb**| Evaluate Data Drift for train/test datasets with Evidently and MLFlow |
| 4. | **4-great-expectatinos.ipynb** | Tutorial for Great Expectations |


## :checkered_flag: View experiments and monitoring reports in MLflow UI

```bash
mlflow ui
``` 
And then navigate to [http://localhost:5000](http://localhost:5000) in your browser


## Acknowledgments

The dataset used in the example is downloaded from: https://www.kaggle.com/c/bike-sharing-demand/data?select=train.csv
- Fanaee-T, Hadi, and Gama, Joao, 'Event labeling combining ensemble detectors and background knowledge', Progress in Artificial Intelligence (2013): pp. 1-15, Springer Berlin Heidelberg
- More information about the dataset can be found in UCI machine learning repository: https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset