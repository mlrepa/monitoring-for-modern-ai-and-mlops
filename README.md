![Comprehensive ML Monitoring for Modern AI and MLOps](docs/images/monitoring-banner-1.png)

# Comprehensive ML Monitoring for Modern AI and MLOps

## 👩‍💻 Installation

### 1. Fork / Clone this repository

Get the tutorial example code:

```bash
git clone https://github.com/mlrepa/monitoring-for-modern-ai-and-mlops.git
cd monitoring-1-get-started
```

### 2️⃣ Create and Activate Virtual Environment

It's highly recommended to use a virtual environment to manage project dependencies.

Install `uv`:

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh
# Or consult official uv documentation for other installation methods
   ```

Create a virtual environment using `uv`:

```bash
uv venv .venv --python 3.12

# Activate the virtual environment
# On macOS and Linux:
source .venv/bin/activate
# On Windows:
# .\.venv\Scripts\activate
```

> 👉 **Note:** The `uv venv .venv` command creates a virtual environment in a folder named `.venv` within your project directory.

### 3️⃣ Install Dependencies

With the virtual environment activated, install the required Python packages:

```bash
# First, install the project in editable mode
uv pip install -e .

# Then install development dependencies
uv pip install --group dev .
```

> 👉 **Note:** Development dependencies are defined in `pyproject.toml` under `[dependency-groups]` and include tools for code quality (linters, formatters) and testing.

### 4️⃣ Download data

This is a preparation step. Load data from [https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset) to the `data/` directory

```bash
python src/load_data.py              
```

## 📺 Monitoring Examples

| No. | Monitoring Example | Description |
|---|---|---|
| 1. | **1-getting-started-tutorial.ipynb** | Get started with Evidently Monitoring |
| 2. | **2-monitor-model.ipynb**| Model Monitoring with Evidently and MLFlow |
| 3. | **3-great-expectatinos.ipynb** | Tutorial for Great Expectations |
| 4. | **4-monitoring-with-grafana.ipynb** | Monitoring with Grafana |

## 🏁 Run MLflow to visualize experiments and monitoring reports

```bash
mlflow ui
```

And then navigate to [http://localhost:5000](http://localhost:5000) in your browser

## Run Grafana to visualise monitoring metrics

```bash
docker compose up
```

The following services will be started:

- *monitoring-db*: monitoring database (PostgreSQL) - for collecting model metrics
- *grafana*: a service (tool) for visualizing metrics

Navigate to Grafana UI on [http://localhost:3000](http://localhost:3000) in your browser. The default credentials are `admin/admin`

## Acknowledgments

The dataset used in the example is downloaded from: https://www.kaggle.com/c/bike-sharing-demand/data?select=train.csv
- Fanaee-T, Hadi, and Gama, Joao, 'Event labeling combining ensemble detectors and background knowledge', Progress in Artificial Intelligence (2013): pp. 1-15, Springer Berlin Heidelberg
- More information about the dataset can be found in UCI machine learning repository: https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset
