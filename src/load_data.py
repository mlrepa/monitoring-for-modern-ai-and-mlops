import datetime
import io
import logging
import zipfile
from pathlib import Path
from typing import NoReturn

import pandas as pd
import requests
from requests.exceptions import RequestException

DATASET_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
DATA_DIR = "data"
FILENAME = "raw_data.csv"


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_data(destination_path: str) -> None:
    """
    Download the Bike Sharing dataset from UCI machine learning repository.

    Args:
        destination_path: Path where the downloaded data will be saved

    Raises:
        RequestException: If the download fails
        OSError: If there are issues with file operations

    More information about the dataset can be found in UCI machine learning repository:
    https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset

    Acknowledgement: Fanaee-T, Hadi, and Gama, Joao, 'Event labeling combining ensemble detectors
    and background knowledge', Progress in Artificial Intelligence (2013): pp. 1-15, Springer Berlin Heidelberg
    """
    try:
        logger.info(f"Downloading dataset from {DATASET_URL}")
        response = requests.get(DATASET_URL)
        response.raise_for_status()
        content = response.content

        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            raw_data = pd.read_csv(
                archive.open("hour.csv"),
                header=0,
                sep=',',
                parse_dates=['dteday']
            )

        raw_data.index = raw_data.apply(
            lambda row: datetime.datetime.combine(
                row.dteday.date(),
                datetime.time(row.hr)
            ),
            axis=1
        )

        # Ensure the directory exists
        Path(destination_path).parent.mkdir(parents=True, exist_ok=True)
        raw_data.to_csv(destination_path, index=False)
        logger.info(f"Data successfully downloaded to: {destination_path}")

    except RequestException as e:
        logger.error(f"Failed to download data: {str(e)}")
        raise
    except OSError as e:
        logger.error(f"Failed to save data: {str(e)}")
        raise


if __name__ == "__main__":

    download_data(f"{DATA_DIR}/{FILENAME}")
