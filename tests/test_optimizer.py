from experiments.base.optimizer_constructor import build_optimizer
import optax
from ml_collections import ConfigDict
from absl.testing import absltest


class TestBuildOptimizer(absltest.TestCase):
    # Test if constructing various optimizers work

    def test_build_optimizer_sgd(self):
        optimizer_config = {
            'name': 'sgd',
            'lr': 0.001,
            'momentum': 0.9,
            'nesterov': True
        }
        optimizer_config = ConfigDict(optimizer_config)
        optimizer, _ = build_optimizer(optimizer_config)
        self.assertTrue(isinstance(optimizer, optax.GradientTransformation))

    def test_build_optimizer_adam(self):
        optimizer_config = {
            'name': 'adam',
            'lr': 0.001,
            'beta1': 0.9
        }
        optimizer_config = ConfigDict(optimizer_config)
        optimizer, _ = build_optimizer(optimizer_config)
        self.assertTrue(isinstance(optimizer, optax.GradientTransformation))

    def test_build_optimizer_adamw(self):
        optimizer_config = {
            'name': 'adamw',
            'lr': 0.001,
            'beta1': 0.9,
            'weight_decay': 0.01
        }
        optimizer_config = ConfigDict(optimizer_config)
        optimizer, _ = build_optimizer(optimizer_config)
        self.assertTrue(isinstance(optimizer, optax.GradientTransformation))

    def test_build_optimizer_schedule_constant(self):
        optimizer_config = {
            'name': 'adam',
            'lr': 0.001,
            'beta1': 0.9,
            'scheduler': {
                'name': 'constant'
            }
        }
        optimizer_config = ConfigDict(optimizer_config)
        optimizer, _ = build_optimizer(optimizer_config)
        self.assertTrue(isinstance(optimizer, optax.GradientTransformation))

    def test_build_optimizer_schedule_cosine_decay(self):
        optimizer_config = {
            'name': 'adam',
            'lr': 0.001,
            'beta1': 0.9,
            'scheduler': {
                'name': 'cosine_decay',
                'alpha': 0.1,
                'decay_steps': 1000
            }
        }
        optimizer_config = ConfigDict(optimizer_config)
        optimizer, _ = build_optimizer(optimizer_config)
        self.assertTrue(isinstance(optimizer, optax.GradientTransformation))

    def test_build_optimizer_schedule_exponential_decay(self):
        optimizer_config = {
            'name': 'adam',
            'lr': 0.001,
            'beta1': 0.9,
            'scheduler': {
                'name': 'exponential_decay',
                'decay_rate': 0.1,
                'transition_steps': 1,
                'staircase': False
            }
        }
        optimizer_config = ConfigDict(optimizer_config)
        optimizer, _ = build_optimizer(optimizer_config)
        self.assertTrue(isinstance(optimizer, optax.GradientTransformation))
    
    def test_build_optimizer_schedule_warmup_cosine_decay(self):
        optimizer_config = {
            'name': 'adam',
            'lr': 0.001,
            'beta1': 0.9,
            'scheduler': {
                'name': 'warmup_cosine_decay',
                'alpha': 0.1,
                'decay_steps': 1000,
                'warmup_steps': 100
            }
        }
        optimizer_config = ConfigDict(optimizer_config)
        optimizer, _ = build_optimizer(optimizer_config)
        self.assertTrue(isinstance(optimizer, optax.GradientTransformation))

    def test_build_optimizer_gradient_clipping(self):
        optimizer_config = {
            'name': 'adam',
            'lr': 0.001,
            'beta1': 0.9,
            'scheduler': {
                'name': 'constant'
            },
            'grad_clip_norm': 1.0,
            'grad_clip_value': 0.1
        }
        optimizer_config = ConfigDict(optimizer_config)
        optimizer, _ = build_optimizer(optimizer_config)
        self.assertTrue(isinstance(optimizer, optax.GradientTransformation))

    def test_build_optimizer_weight_decay(self):
        optimizer_config = {
            'name': 'adam',
            'lr': 0.001,
            'beta1': 0.9,
            'weight_decay': 0.01
        }
        optimizer_config = ConfigDict(optimizer_config)
        optimizer, _ = build_optimizer(optimizer_config)
        self.assertTrue(isinstance(optimizer, optax.GradientTransformation))


if __name__ == '__main__':
    absltest.main()