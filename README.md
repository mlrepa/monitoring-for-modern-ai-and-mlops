# Monitoring for Modern AI and MLOps Tutorial

![Comprehensive ML Monitoring for Modern AI and MLOps](static/images/monitoring-banner-1.png)

This tutorial provides a comprehensive guide to ML monitoring for modern AI and MLOps, demonstrating how to keep your machine learning models and data reliable in production. It covers essential monitoring concepts and provides practical examples using open-source tools: **Evidently AI** for data and model quality, **MLflow** for agent tracing and evaluation, and **Grafana** for metric visualization.

## Overview

The tutorial covers:

- The essential tasks and strategies for effective ML monitoring in an MLOps context, including different monitoring layers and paradigms.
- Setting up model performance and data quality monitoring with Evidently AI (using its modern API).
- Setting up dedicated monitoring for LLM-powered systems and AI Agents with Evidently AI.
- Tracing and evaluating AI Agents with MLflow.
- Visualizing crucial monitoring metrics with Grafana dashboards.

## Prerequisites

- Python 3.9+
- `uv` (Python package installer)
- Docker and Docker Compose
- Git
- Basic understanding of Python, pandas, and machine learning concepts.

## Quick Start: Installation & Setup

1. **Clone the repository:**

    ```bash
    git clone https://github.com/mlrepa/monitoring-for-modern-ai-and-mlops.git
    cd monitoring-for-modern-ai-and-mlops
    ```

2. **Create and activate a virtual environment:**

    ```bash
    uv venv .venv --python 3.12

    # Activate the virtual environment
    # On macOS and Linux:
    source .venv/bin/activate
    # On Windows:
    # .\.venv\Scripts\activate
    ```

    > 👉 **Note:** The `uv venv .venv` command creates a virtual environment in a folder named `.venv` within your project directory.

3. **Install Dependencies:**

    With the virtual environment activated, install the required Python packages:

    ```bash
    # Install the project with all dependencies
    uv sync

    # Install development dependencies (optional)
    uv sync --extra dev
    ```

4. **Set up Jupyter Kernel:**

    Create and register a Jupyter kernel for this project to ensure the notebooks run with the correct environment:

    ```bash
    # Create a kernel for this project
    python -m ipykernel install --user --name=ml-monitoring --display-name="ML Monitoring Tutorial"
    ```

    Now you can run the notebooks with Jupyter Lab:

    ```bash
    jupyter lab
    ```

    > 👉 **Important:** When opening notebooks in Jupyter Lab, make sure to select the "ML Monitoring Tutorial" kernel from the kernel dropdown menu to ensure all dependencies are available.

## Project Structure

```text
.
├── data/              # Data directory (contains raw data if loaded via script)
├── models/            # Placeholder for trained models if needed
├── notebooks/         # Jupyter notebooks for tutorial steps
│   ├── 1-evidently-getting-started.ipynb    # Core Evidently AI for ML models/data
│   ├── 2-evidenlty-rag-metrics.ipynb      # Evidently AI for LLM/RAG systems
│   ├── 3-grafana-getting-started.ipynb    # Initial Grafana setup
│   └── 4-monitor-model-with-grafana.ipynb # Logging metrics to DB for Grafana
├── src/               # Python source code
│   └── load_data.py   # Script to download and prepare data
├── docker-compose.yaml # Docker Compose file for PostgreSQL and Grafana
├── pyproject.toml     # Project dependencies and metadata
└── README.md          # This README file
```

## Getting Started with the Tutorial

To follow the tutorial step-by-step:

1. **Load data:**
    Load data from [https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset) to the `data/` directory.

    ```bash
    python src/load_data.py
    ```

2. **Start the monitoring stack:**
    This command will launch PostgreSQL (for metric storage) and Grafana (for visualization).

    ```bash
    docker compose up -d
    ```

    The following services will be started:

    - `monitoring-db`: PostgreSQL database for collecting model metrics.
    - `grafana`: Grafana service for visualizing metrics.

    Navigate to the Grafana UI on [http://localhost:3000](http://localhost:3000) in your browser. The default credentials are `admin/admin`.

3. **Open `tutorial.md` and run the notebooks in order:**
    Launch Jupyter Lab and follow the `tutorial.md` (the main guide for this project), running the corresponding Jupyter notebooks as indicated in each section.

    Make sure to select the "ML Monitoring Tutorial" kernel when opening each notebook. The notebooks to follow are:

    - `1-evidently-getting-started.ipynb`: Introduction to Evidently AI for traditional ML models and data.
    - `2-evidenlty-rag-metrics.ipynb`: Using Evidently AI for LLM-powered systems and Agents.
    - `3-evidently-grafana-ml.ipynb`: Logging model quality metrics to PostgreSQL and visualizing them in Grafana.

    > 👉 **Note on MLflow section:** The MLflow section in the `tutorial.md` provides conceptual guidance and code snippets, drawing from an external example for tracing and evaluating AI agents. It does not require a specific notebook from *this* repository to run.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset: [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset)
- Evidently AI: [https://www.evidentlyai.com/](https://www.evidentlyai.com/)
- Grafana: [https://grafana.com/](https://grafana.com/)
- MLflow: [https://mlflow.org/](https://mlflow.org/)
