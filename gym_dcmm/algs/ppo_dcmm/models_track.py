import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, units, input_size):
        super(MLP, self).__init__()
        layers = []
        for output_size in units:
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ELU())
            input_size = output_size
        self.mlp = nn.Sequential(*layers)

        # orthogonal init of weights
        # hidden layers scale np.sqrt(2)
        self.init_weights(self.mlp, [np.sqrt(2)] * len(units))

    def forward(self, x):
        return self.mlp(x)

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

class CNNBase(nn.Module):
    def __init__(self, input_shape, hidden_size=256):
        super(CNNBase, self).__init__()
        # Input shape is (C, H, W)
        # Assuming input is depth image (1, 112, 112) or similar
        
        self.main = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Compute output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            output = self.main(dummy_input)
            self.cnn_output_size = output.shape[1]
            
        self.linear = nn.Linear(self.cnn_output_size, hidden_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.main(x)
        x = self.linear(x)
        x = self.relu(x)
        return x

class ActorCritic(nn.Module):
    def __init__(self, kwargs):
        nn.Module.__init__(self)
        separate_value_mlp = kwargs.pop('separate_value_mlp')
        self.separate_value_mlp = separate_value_mlp

        actions_num = kwargs.pop('actions_num')
        input_shape = kwargs.pop('input_shape') # This is now a dict or tuple?
        # We expect input_shape to be a dict with 'vector' and 'image' keys, or we handle it in forward
        # For now, let's assume input_shape passed here is the vector shape, and we hardcode/config the image shape
        
        self.units = kwargs.pop('actor_units')
        
        self.use_vision = kwargs.get('use_vision', True)
        
        # Vector input size
        if isinstance(input_shape, dict):
             mlp_input_shape = input_shape['vector'][0]
             img_input_shape = input_shape['image']
        else:
             # Fallback for legacy/tuple input
             mlp_input_shape = input_shape[0]
             img_input_shape = None
             self.use_vision = False

        # CNN Feature Extractor
        if self.use_vision:
            self.cnn = CNNBase(img_input_shape, hidden_size=256)
            combined_input_size = mlp_input_shape + 256
        else:
            self.cnn = None
            combined_input_size = mlp_input_shape
        
        out_size = self.units[-1]

        self.actor_mlp = MLP(units=self.units, input_size=combined_input_size)
        if self.separate_value_mlp:
            self.value_mlp = MLP(units=self.units, input_size=combined_input_size)
        self.value = torch.nn.Linear(out_size, 1)
        self.mu = torch.nn.Linear(out_size, actions_num)
        self.sigma = nn.Parameter(
            torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                fan_out = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)
        nn.init.constant_(self.sigma, 0)

        # policy output layer with scale 0.01
        # value output layer with scale 1
        torch.nn.init.orthogonal_(self.mu.weight, gain=0.01)
        torch.nn.init.orthogonal_(self.value.weight, gain=1.0)
    
    def save_actor(self, actor_mlp_path, actor_head_path):
        """
        Save actor and critic model parameters to files.

        actor_path: Path to save actor model parameters.
        """
        torch.save(self.actor_mlp.state_dict(), actor_mlp_path)
        torch.save(self.mu.state_dict(), actor_head_path)
        # Also save CNN?
        # torch.save(self.cnn.state_dict(), cnn_path)

    @torch.no_grad()
    def act(self, obs_dict):
        # used specifically to collection samples during training
        # it contains exploration so needs to sample from distribution
        mu, logstd, value = self._actor_critic(obs_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        selected_action = distr.sample()
        result = {
            'neglogpacs': -distr.log_prob(selected_action).sum(1),
            'values': value,
            'actions': selected_action,
            'mus': mu,
            'sigmas': sigma,
        }
        return result

    @torch.no_grad()
    def act_inference(self, obs_dict):
        # used for testing
        mu, logstd, value = self._actor_critic(obs_dict)
        return mu

    def _actor_critic(self, obs_dict):
        # obs_dict['obs'] is expected to be a dict or contain both parts
        # But PPO storage usually flattens things.
        # We need to check how PPO passes data.
        # In PPO_Track.model_act, it passes {'obs': processed_obs}
        # processed_obs comes from running_mean_std.
        
        # If we changed obs to be a dict in PPO, then processed_obs is a dict?
        # We need to ensure PPO passes 'vector' and 'image' correctly.
        
        if isinstance(obs_dict['obs'], dict):
            vector_obs = obs_dict['obs']['vector']
            if self.use_vision:
                image_obs = obs_dict['obs']['image']
        else:
            # Fallback if obs is just tensor (legacy)
            vector_obs = obs_dict['obs']
            image_obs = None

        if self.use_vision and image_obs is not None:
            # Process Image
            img_features = self.cnn(image_obs)
            # Concatenate
            x = torch.cat([vector_obs, img_features], dim=1)
        else:
            x = vector_obs

        x_actor = self.actor_mlp(x)
        mu = self.mu(x_actor)
        
        if self.separate_value_mlp:
            x_value = self.value_mlp(x)
        else:
            x_value = x_actor # Share features if not separate
            
        value = self.value(x_value)

        sigma = self.sigma
        # Normalize to (-1,1)
        mu = torch.tanh(mu)
        return mu, mu * 0 + sigma, value

    def forward(self, input_dict):
        prev_actions = input_dict.get('prev_actions', None)
        mu, logstd, value = self._actor_critic(input_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        entropy = distr.entropy().sum(dim=-1)
        prev_neglogp = -distr.log_prob(prev_actions).sum(1)
        result = {
            'prev_neglogp': torch.squeeze(prev_neglogp),
            'values': value,
            'entropy': entropy,
            'mus': mu,
            'sigmas': sigma,
        }
        return result