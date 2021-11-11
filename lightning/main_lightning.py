import logging
import os.path
from datetime import datetime
import tqdm
import argparse
import yaml

import pytorch_lightning
from pytorch_lightning import Trainer, seed_everything

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback

from lightning.cloudCT_data_module import CloudCTDataModule
from lightning.cloudCT_module import CloudCTModule
from lightning.utils import create_and_configure_logger
from lightning.params import RAY_HYPER_PARAMS, CONSTS, NON_RAY_HYPER_PARAMS, NUM_AVAILABLE_GPU, RESULTS_PATH, USE_RAY


class CloudCT_ProgressBar(pytorch_lightning.callbacks.ProgressBar):

    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.set_description('running train ...')
        return bar

    def on_train_start(self, trainer, pl_module):
        print('Starting train!')

    def on_train_end(self, trainer, pl_module):
        print('Train ended.')

    def on_validation_start(self, trainer, pl_module):
        print('Starting validation!')

    def on_validation_end(self, trainer, pl_module):
        print('Validation ended.')

    def on_test_start(self, trainer, pl_module):
        print('Starting test!')

    def on_test_end(self, trainer, pl_module):
        print('Test ended.')


def main(args, ray_params=None, consts=None):

    config_path = args.conf
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Define Model
    model = CloudCTModule(config).double()

    # Define Data
    cloudCT_dm = CloudCTDataModule(config=config, train_csv_path=config['dataset']["train_csv"], test_csv_path=config['dataset']["test_csv"],
                               batch_size=config['training']['batch_size'], num_workers=config['training']['num_workers'],)

    # Define minimization parameter TODO: figure out what is this
    metrics = {"loss": f"loss/{config['training']['loss_type']}_val",
               "accuracy": "acc/accuracy_val"}
    callbacks = [CloudCT_ProgressBar(), TuneReportCheckpointCallback(metrics, filename="checkpoint", on="validation_end")]

    # Setup the pytorch-lighting trainer and run the model
    trainer = Trainer(max_epochs=config['training']['epochs'],
                      callbacks=callbacks, gpus=1)

    cloudCT_dm.setup('fit')
    trainer.fit(model=model, datamodule=cloudCT_dm)

    # # test
    # cloudCT_dm.setup('test')
    # trainer.test(model=model, datamodule=cloudCT_dm.test_dataloader())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Seq2seq')
    parser.add_argument('-c', '--conf', help='path to configuration file', required=True)

    # group = parser.add_mutually_exclusive_group()
    # group.add_argument('--train', action='store_true', help='Train')
    # group.add_argument('--predict_on_test_set', action='store_true', help='Predict on test set')
    # group.add_argument('--predict', action='store_true', help='Predict on single file')
    #
    # parser.add_argument('--filename', help='path to file')

    args = parser.parse_args()

    logger = create_and_configure_logger(
        log_name=f"{os.path.join(os.getcwd(), 'lightning_logs/')}_{datetime.now().strftime('%Y-%m-%d %H_%M_%S')}.log", level=logging.INFO)

    if USE_RAY:
        reporter = CLIReporter(
            metric_columns=["loss", "accuracy", "training_iteration"],
            max_progress_rows=50,
            print_intermediate_tables=True)

        analysis = tune.run(
            tune.with_parameters(main(args), consts=CONSTS),
            config=RAY_HYPER_PARAMS,
            local_dir=RESULTS_PATH,  # where to save the results
            fail_fast=True,  # if one run fails - stop all runs
            metric="accuracy",
            mode="max",
            progress_reporter=reporter,
            log_to_file=True,
            resources_per_trial={"cpu": 6, "gpu": NUM_AVAILABLE_GPU})

        logger.info(f"best_trial {analysis.best_trial}")
        logger.info(f"best_config {analysis.best_config}")
        logger.info(f"best_logdir {analysis.best_logdir}")
        logger.info(f"best_checkpoint {analysis.best_checkpoint}")
        logger.info(f"best_result {analysis.best_result}")
        results_df = analysis.dataframe(metric="accuracy", mode="max", )
        results_df.to_csv(os.path.join(analysis._experiment_dir, f'output_table.csv'))

        # main(args)
    else:
        main(args, ray_params=NON_RAY_HYPER_PARAMS, consts=CONSTS)
