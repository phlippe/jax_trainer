from experiments.base.trainer import TrainerModule
from experiments.classifier.img_classifier import ImgClassifierTrainer
from datasets import build_dataset_module
import optax
from ml_collections import ConfigDict
from absl.testing import absltest
import yaml
import pathlib


class TestBuildTrainer(absltest.TestCase):
    # Test if constructing various optimizers work

    def test_build_trainer(self):
        config = yaml.safe_load(pathlib.Path('tests/config/cifar10_classifier.yaml').read_text())
        config = ConfigDict(config)
        dataset = build_dataset_module(config.dataset)
        exmp_input = next(iter(dataset.train_loader))
        trainer = ImgClassifierTrainer(trainer_config=config.trainer,
                                       model_config=config.model,
                                       optimizer_config=config.optimizer,
                                       exmp_input=exmp_input)
        eval_metrics = trainer.train_model(train_loader=dataset.train_loader, 
                                           val_loader=dataset.val_loader, 
                                           test_loader=dataset.test_loader,
                                           num_epochs=5)