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


class DepthCNN(nn.Module):
    """
    CNN for processing depth images.
    Input: (B, 1, H, W)
    Output: (B, 256)
    """
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        # Calculate output size for 84x84 input
        # 84 -> (84-8)/4 + 1 = 20
        # 20 -> (20-4)/2 + 1 = 9
        # 9 -> (9-3)/1 + 1 = 7
        # Output: 32 * 7 * 7 = 1568
        
        self.linear = nn.Sequential(
            nn.Linear(32 * 7 * 7, 256), 
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.main(x)
        x = self.linear(x)
        return x


class ActorCritic(nn.Module):
    def __init__(self, kwargs):
        nn.Module.__init__(self)
        separate_value_mlp = kwargs.pop('separate_value_mlp')
        self.separate_value_mlp = separate_value_mlp

        actions_num = kwargs.pop('actions_num')
        input_shape = kwargs.pop('input_shape')
        self.units = kwargs.pop('actor_units')
        
        # New args for splitting
        self.state_dim = kwargs.pop('state_dim', 35)
        self.depth_pixels = kwargs.pop('depth_pixels', 84*84)
        self.img_size = kwargs.pop('img_size', 84)
        
        out_size = self.units[-1]
        
        # Actor MLP (Takes State only)
        # Note: We keep the name actor_mlp_c for compatibility with PPO script if needed,
        # but PPO script calls model(batch_dict) so it doesn't access actor_mlp_c directly usually,
        # except for loading weights.
        # Catching Actor
        self.actor_mlp_c = MLP(units=self.units, input_size=self.state_dim)
        
        # Tracking Actor (Legacy / Placeholder)
        # We initialize it to avoid errors if PPO tries to access it, but it might not be used correctly
        # if we don't load weights.
        self.actor_mlp_t = MLP(units=self.units, input_size=self.state_dim) # Placeholder size

        # Critic (Takes State + Depth)
        # 1. Vision Encoder
        self.critic_cnn = DepthCNN()
        
        # 2. State Encoder (MLP)
        self.value_mlp = MLP(units=self.units, input_size=self.state_dim)
        
        # 3. Fusion Head
        # Concatenates State Features (out_size) + Visual Features (256)
        self.value_head = nn.Linear(out_size + 256, 1)
        
        self.mu_t = torch.nn.Linear(out_size, actions_num-12) # Placeholder
        self.mu_c = torch.nn.Linear(out_size, actions_num) # Full action dim (20)
        
        self.sigma_t = nn.Parameter(
            torch.zeros(actions_num-12, requires_grad=True, dtype=torch.float32), requires_grad=True)
        self.sigma_c = nn.Parameter(
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
        nn.init.constant_(self.sigma_t, 0)
        nn.init.constant_(self.sigma_c, 0)
        # policy output layer with scale 0.01
        # value output layer with scale 1
        torch.nn.init.orthogonal_(self.mu_t.weight, gain=0.01)
        torch.nn.init.orthogonal_(self.mu_c.weight, gain=0.01)
        torch.nn.init.orthogonal_(self.value_head.weight, gain=1.0)

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
        # obs_dict['obs'] is the flattened tensor (B, state_dim + depth_pixels)
        obs = obs_dict['obs']
        
        # 1. Split
        state = obs[:, :self.state_dim]
        depth_flat = obs[:, self.state_dim:]
        
        # 2. Process Depth
        # Reshape to (B, 1, H, W) and Normalize (0-255 -> 0-1)
        depth_img = depth_flat.view(-1, 1, self.img_size, self.img_size)
        depth_img = depth_img.float() / 255.0
        
        # 3. Actor Path (State Only)
        x_c = self.actor_mlp_c(state)
        mu_c = self.mu_c(x_c)
        
        # 4. Critic Path (State + Depth)
        # State Features
        state_feat = self.value_mlp(state)
        # Visual Features
        vis_feat = self.critic_cnn(depth_img)
        # Fusion
        combined = torch.cat([state_feat, vis_feat], dim=1)
        value = self.value_head(combined)

        sigma_c = self.sigma_c
        # Normalize to (-1,1)
        mu_c = torch.tanh(mu_c)

        # Note: We are only using the Catching Actor (mu_c) as per instructions.
        # Tracking Actor is ignored/placeholder.
        mu = mu_c 
        sigma = sigma_c

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