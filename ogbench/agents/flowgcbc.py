import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from functools import partial

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActorVectorField, UnconditionalEmbedding


class FlowGCBCAgent(flax.struct.PyTreeNode):
    """Implicit flow Q-learning (IFQL) agent.

    IFQL is the flow variant of implicit diffusion Q-learning (IDQL).
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()


    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the behavioral flow-matching actor loss."""
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng, cfg_rng = jax.random.split(rng, 4)

        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        unconditional_embed = self.network.select('unconditional_embed')(params=grad_params) # (1, goal_dim)
        do_cfg = jax.random.bernoulli(cfg_rng, p=0.1, shape=(batch_size,))
        goals = jnp.where(do_cfg[:, None], unconditional_embed, batch['actor_goals'])

        pred = self.network.select('actor_flow')(batch['observations'], x_t, t, goals=goals, params=grad_params)
        actor_loss = jnp.mean((pred - vel) ** 2)

        return actor_loss, {
            'actor_loss': actor_loss,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = actor_loss
        return loss, info

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        return self.replace(network=new_network, rng=new_rng), info

    @partial(jax.jit, static_argnames=['flow_steps'])
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        cfg=None,
        use_gaussian=False,
        temperature=None,
        flow_steps=None,
    ):
        """Sample actions from the actor."""
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_flow_encoder')(observations)

        action_seed, noise_seed = jax.random.split(seed)
        actions = jax.random.normal(
            action_seed,
            (*observations.shape[:-1], self.config['action_dim']),
        )

        if cfg is None:
            cfg = self.config['cfg']
        if flow_steps is None:
            flow_steps = self.config['flow_steps']
        unk_embed = self.network.select('unconditional_embed')()[0]
        for i in range(flow_steps):
            t = jnp.full((*observations.shape[:-1], 1), i / flow_steps)

            vel_unk = self.network.select('actor_flow')(observations, actions, t, goals=unk_embed, is_encoded=True)
            vel_cond = self.network.select('actor_flow')(observations, actions, t, goals=goals, is_encoded=True)
            vel = vel_unk + cfg * (vel_cond - vel_unk)

            actions = actions + vel / flow_steps
        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        example_batch,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            ex_goals: Example batch of goals.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_observations = example_batch['observations']
        ex_actions = example_batch['actions']
        ex_goals = example_batch['value_goals']
        ex_times = ex_actions[..., :1]
        action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['actor_flow'] = encoder_module()

        # Define networks.
        actor_flow_def = GCActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_flow'),
        )

        uc_embed_def = UnconditionalEmbedding(
            goal_dim=ex_goals.shape[-1],
        )

        network_info = dict(
            actor_flow=(actor_flow_def, (ex_observations, ex_actions, ex_times, ex_goals)),
            unconditional_embed=(uc_embed_def, ()),
        )
        if encoders.get('actor_flow') is not None:
            # Add actor_flow_encoder to ModuleDict to make it separately callable.
            network_info['actor_flow_encoder'] = (encoders.get('actor_flow'), (ex_observations,))
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adamw(learning_rate=config['lr'], weight_decay=config['weight_decay'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        config['action_dim'] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='flowgcbc',  # Agent name.
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,  # Learning rate.
            weight_decay=0.0,  # Weight decay.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            flow_steps=16,  # Number of flow steps.
            cfg=1.0,  # Flow control factor.
            discrete=False,
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            dataset_class='GCDataset',
            value_p_curgoal=0.0,  # Unused (defined for compatibility with GCDataset).
            value_p_trajgoal=1.0,  # Unused (defined for compatibility with GCDataset).
            value_p_randomgoal=0.0,  # Unused (defined for compatibility with GCDataset).
            value_geom_sample=False,  # Unused (defined for compatibility with GCDataset).
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor
            gc_negative=True,
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),
        )
    )
    return config