import optax
from ml_collections import ConfigDict


def build_optimizer(optimizer_config: ConfigDict,
                    num_epochs: int = 0,
                    num_train_steps_per_epoch: int = 0):
    """Build optimizer from config.

    Args:
        optimizer_config (ConfigDict): ConfigDict for optimizer.

    Returns:
        optax.GradientTransformation: Optimizer.
    """
    # Build optimizer class
    optimizer_name = optimizer_config.name
    optimizer_name = optimizer_name.lower()
    opt_class = None
    if optimizer_name == 'adam':
        opt_class = lambda sched: optax.adam(sched,
                                             b1=optimizer_config.get('beta1', 0.9),
                                             b2=optimizer_config.get('beta2', 0.999),
                                             eps=optimizer_config.get('eps', 1e-8))
    elif optimizer_name == 'adamw':
        opt_class = lambda sched: optax.adamw(sched,
                                              b1=optimizer_config.get('beta1', 0.9),
                                              b2=optimizer_config.get('beta2', 0.999),
                                              eps=optimizer_config.get('eps', 1e-8),
                                              weight_decay=optimizer_config.get('weight_decay', 0.0))
    elif optimizer_name == 'sgd':
        opt_class = lambda sched: optax.sgd(sched,
                                            momentum=optimizer_config.get('momentum', 0.0),
                                            nesterov=optimizer_config.get('nesterov', False))
    else:
        raise ValueError(f'Unknown optimizer {optimizer_name}.')
    
    # Build learning rate schedule
    lr = float(optimizer_config.lr)
    scheduler_config = optimizer_config.get('scheduler', ConfigDict())
    schedule_name = scheduler_config.get('name', None)
    decay_steps = scheduler_config.get('decay_steps', 
                                       num_epochs * num_train_steps_per_epoch)
    lr_schedule = None
    if schedule_name is None or schedule_name == 'constant':
        lr_schedule = optax.constant_schedule(lr)
    elif schedule_name == 'cosine_decay':
        assert decay_steps > 0, 'decay_steps must be positive'
        lr_schedule = optax.cosine_decay_schedule(
            init_value=lr,
            decay_steps=decay_steps,
            alpha=scheduler_config.get('alpha', 0.0),
        )
    elif schedule_name == 'exponential_decay':
        lr_schedule = optax.exponential_decay(
            init_value=lr,
            decay_rate=scheduler_config.decay_rate,
            transition_steps=scheduler_config.get('transition_steps', 1),
            staircase=scheduler_config.get('staircase', False),
        )
    elif schedule_name == 'warmup_cosine_decay':
        assert decay_steps > 0, 'decay_steps must be positive'
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=lr,
            decay_steps=decay_steps,
            warmup_steps=scheduler_config.warmup_steps,
            end_value=scheduler_config.get('end_value', 0.0),
        )
    else:
        raise ValueError(f'Unknown learning rate schedule {schedule_name}.')
    
    # Gradient transformation
    grad_trans = []
    if optimizer_config.get('grad_clip_norm', None) is not None:
        grad_trans.append(
            optax.clip_by_global_norm(optimizer_config.grad_clip_norm)
        )
    if optimizer_config.get('grad_clip_value', None) is not None:
        grad_trans.append(
            optax.clip(optimizer_config.grad_clip_value)
        )
    if optimizer_config.get('weight_decay', 0.0) > 0.0 and not optimizer_name in ['adamw']:
        grad_trans.append(
            optax.add_decayed_weights(optimizer_config.weight_decay)
        )
    
    # Put everything together
    optimizer = optax.chain(
        *grad_trans,
        opt_class(lr_schedule)
    )
    return optimizer, lr_schedule