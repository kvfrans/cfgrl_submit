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
from utils.networks import GCActorVectorField, GCValue, IsPositiveEmbedding


class FlowIQLAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def value_loss(self, batch, grad_params):
        """Compute the IQL value loss."""
        q1, q2 = self.network.select('target_critic')(batch['observations'], actions=batch['actions'])
        q = jnp.minimum(q1, q2)
        v = self.network.select('value')(batch['observations'], params=grad_params)
        value_loss = self.expectile_loss(q - v, q - v, self.config['expectile']).mean()

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def critic_loss(self, batch, grad_params):
        """Compute the IQL critic loss."""
        next_v = self.network.select('value')(batch['next_observations'])
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v

        q1, q2 = self.network.select('critic')(
            batch['observations'], actions=batch['actions'], params=grad_params
        )
        critic_loss = ((q1 - q) ** 2 + (q2 - q) ** 2).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the behavioral flow-matching actor loss."""
        v = self.network.select('value')(batch['observations'])
        q1, q2 = self.network.select('critic')(batch['observations'], actions=batch['actions'])
        q = jnp.minimum(q1, q2)
        adv = q - v
        is_positive = (adv >= self.config['adv_threshold']).astype(jnp.int8) + 1 # [1, 2]

        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng, cfg_rng = jax.random.split(rng, 4)

        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        do_cfg = jax.random.bernoulli(cfg_rng, p=0.1, shape=(batch_size,))
        is_positive_label = jnp.where(do_cfg, 0, is_positive)
        is_positive_emb = self.network.select('is_positive_embedding')(is_positive_label, params=grad_params)

        pred = self.network.select('actor_flow')(batch['observations'], x_t, t, is_positive_emb, params=grad_params)
        actor_loss = jnp.mean((pred - vel) ** 2)

        return actor_loss, {
            'actor_loss': actor_loss,
            'adv_mean': adv.mean(),
            'adv_max': adv.max(),
            'adv_min': adv.min(),
            'ratio': is_positive.mean(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        critic_loss, critic_info = self.critic_loss(batch, grad_params)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = value_loss + critic_loss + actor_loss
        return loss, info
    
    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')

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
        assert goals is None
        action_seed, noise_seed = jax.random.split(seed)
        actions = jax.random.normal(
            action_seed,
            (*observations.shape[:-1], self.config['action_dim']),
        )

        if cfg is None:
            cfg = self.config['cfg']
        if flow_steps is None:
            flow_steps = self.config['flow_steps']

        unc_embed = self.network.select('is_positive_embedding')(jnp.zeros(1, dtype=jnp.int8))[0]
        pos_embed = self.network.select('is_positive_embedding')(jnp.ones(1, dtype=jnp.int8) * 2)[0]
        for i in range(flow_steps):
            t = jnp.full((*observations.shape[:-1], 1), i / flow_steps)

            vel_unk = self.network.select('actor_flow')(observations, actions, t, unc_embed)
            vel_cond = self.network.select('actor_flow')(observations, actions, t, pos_embed)
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
        ex_times = ex_actions[..., :1]
        action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['actor_flow'] = encoder_module()

        value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=True,
            ensemble=False,
            gc_encoder=encoders.get('value'),
        )

        critic_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=True,
            ensemble=True,
            gc_encoder=encoders.get('critic'),
        )

        # Define networks.
        actor_flow_def = GCActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            # encoder=encoders.get('actor_flow'),
        )

        is_positive_embed_def = IsPositiveEmbedding(
            emb_dim=32,
        )
        pos_label = jnp.ones(ex_observations.shape[:-1] + (is_positive_embed_def.emb_dim,), dtype=jnp.int8)

        network_info = dict(
            value=(value_def, (ex_observations,)),
            critic=(critic_def, (ex_observations, None, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, None, ex_actions)),
            actor_flow=(actor_flow_def, (ex_observations, ex_actions, ex_times, pos_label)),
            is_positive_embedding=(is_positive_embed_def, (pos_label,)),
        )
        # if encoders.get('actor_flow') is not None:
        #     # Add actor_flow_encoder to ModuleDict to make it separately callable.
        #     network_info['actor_flow_encoder'] = (encoders.get('actor_flow'), (ex_observations,))
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adamw(learning_rate=config['lr'], weight_decay=config['weight_decay'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network_params['modules_target_critic'] = network_params['modules_critic']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        config['action_dim'] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='flowiql',  # Agent name.
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,  # Learning rate.
            weight_decay=0.0,  # Weight decay.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            tau=0.005,  # Target network update rate.
            expectile=0.9,  # IQL expectile.
            discount=0.99,  # Discount factor.
            flow_steps=16,  # Number of flow steps.
            cfg=1.0,  # Flow control factor.
            adv_threshold=0.0,  # Advantage threshold for positive sampling.
            discrete=False,
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            dataset_class='Dataset',
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),
        )
    )
    return config