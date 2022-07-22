import click
import mlflow
from flash.core.data.utils import download_data
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

EXPERIMENT_NAME = "dl_model_chapter04"
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://hpc.if.uz.zgora.pl:12303"
os.environ["MLFLOW_TRACKING_URI"] = "http://hpc.if.uz.zgora.pl:12302"

mlflow.set_experiment(EXPERIMENT_NAME)
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
logger.info("pipeline experiment_id: %s", experiment.experiment_id)


@click.command(help="This program downloads data for finetuning a deep learning model for sentimental classification.")
@click.option("--download_url", default="https://pl-flash-data.s3.amazonaws.com/imdb.zip",
              help="This is the remote url where the data will be downloaded")
@click.option("--local_folder", default="./data", help="This is a local data folder.")
@click.option("--pipeline_run_name", default="chapter06", help="This is a mlflow run name.")
def task(download_url, local_folder, pipeline_run_name):
    with mlflow.start_run(run_name=pipeline_run_name) as mlrun:
        logger.info("Downloading data from  %s", download_url)
        download_data(download_url, local_folder)
        mlflow.log_param("download_url", download_url)
        mlflow.log_param("local_folder", local_folder)
        mlflow.log_param("mlflow run id", mlrun.info.run_id)
        mlflow.set_tag('pipeline_step', __file__)
        mlflow.log_artifacts(local_folder, artifact_path="data")

    logger.info("finished downloading data to %s", local_folder)


if __name__ == '__main__':
    task()
