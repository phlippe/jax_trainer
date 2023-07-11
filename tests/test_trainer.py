import json
import os
import pathlib
import shutil
from glob import glob

import optax
import yaml
from absl import logging
from absl.testing import absltest
from ml_collections import ConfigDict

from datasets import build_dataset_module
from experiments.base.trainer import TrainerModule
from experiments.classifier.img_classifier import ImgClassifierTrainer


class TestBuildTrainer(absltest.TestCase):
    # Test if constructing various optimizers work

    def test_build_trainer(self):
        config = yaml.safe_load(pathlib.Path("tests/config/cifar10_classifier.yaml").read_text())
        config = ConfigDict(config)
        dataset = build_dataset_module(config.dataset)
        exmp_input = next(iter(dataset.train_loader))
        trainer = ImgClassifierTrainer(
            trainer_config=config.trainer,
            model_config=config.model,
            optimizer_config=config.optimizer,
            exmp_input=exmp_input,
        )
        eval_metrics = trainer.train_model(
            train_loader=dataset.train_loader,
            val_loader=dataset.val_loader,
            test_loader=dataset.test_loader,
            num_epochs=trainer.trainer_config.train_epochs,
        )
        self.assertGreater(eval_metrics[5]["val/acc"], 0.5)
        self.assertGreater(eval_metrics["test"]["test/acc"], 0.5)
        logging.get_absl_handler().flush()
        logging.get_absl_handler().close()

    def test_loading_trainer(self):
        orig_config = yaml.safe_load(
            pathlib.Path("tests/config/cifar10_classifier.yaml").read_text()
        )
        orig_config = ConfigDict(orig_config)

        ckpt_folder = sorted(glob("tests/checkpoints/BuildTrainerTest/checkpoints/*"))[-1]
        trainer = ImgClassifierTrainer.load_from_checkpoint(
            ckpt_folder, exmp_input_file="tests/checkpoints/BuildTrainerTest/exmp_input.pkl"
        )

        self.assertTrue(
            trainer.trainer_config.logger.log_dir == orig_config.trainer.logger.log_dir
        )
        self.assertTrue(trainer.trainer_config.train_epochs == orig_config.trainer.train_epochs)

        dataset = build_dataset_module(orig_config.dataset)
        test_metrics = trainer.test_model(dataset.test_loader)
        self.assertGreater(test_metrics["test/acc"], 0.5)

        with open("tests/checkpoints/BuildTrainerTest/metrics/test_epoch_0005.json", "rb") as f:
            orig_test_metric = json.load(f)
        self.assertAlmostEqual(test_metrics["test/acc"], orig_test_metric["test/acc"], places=4)

    @classmethod
    def tearDownClass(cls):
        # Remove checkpoint
        shutil.rmtree("tests/checkpoints/BuildTrainerTest")
