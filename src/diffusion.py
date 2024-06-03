import numpy as np
import torch
import torch.nn as nn
from unet import ScoreNet
# diffusion.py contains 2 different diffusion models with 
# different loss functions


# Diffusion Model that implements loss functiond defined 
# at the end of Question 3 in ../theory/writeup.pdf`
class Diffusion(nn.Module):
    def __init__(self, model, n_steps, device, min_beta, max_beta):
        super().__init__()
        
        self.n_steps = n_steps
        # All beta, alpha, alpha_bar are padded with an extra element in the 0th index
        self.beta = torch.linspace(min_beta, max_beta, n_steps)
        self.beta = torch.hstack([torch.Tensor([0]), self.beta]).to(device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, 0)
        self.device = device
        self.mu = model.to(device)
    
    def forward_process(self, x_0, t):
        """
        Given an $x_0, t$ outputs $x_{t-1}, x_{t}$. It first computes $x_{t-1}$ then $q(x_t|x_{t-1})$.
        """
        x_t1 = (x_0 * torch.sqrt(self.alpha_bar[t-1]) 
                + torch.randn_like(x_0,requires_grad=False) * torch.sqrt(1-self.alpha_bar[t-1]))
        x_t = (torch.sqrt(self.alpha[t]) * x_t1
                + torch.randn_like(x_0, requires_grad=False) * torch.sqrt(1-self.alpha[t]))

        return x_t1, x_t
    
    def predict_mu(self, xt, t):
        """
        Utilizes the `ScoreNet` network to determine $\mu(x_t, t; \theta)$
        """
        t = t.reshape(-1, 1)
        return self.mu(xt, t)
    
    def forward_direct(self, x_0):
        """
        Given $x_0$ directly computes $x_t$ and returns the randomly sampled
        $t$ as well. $t$ ranges from $1$ to $N-steps$.
        """
        n = x_0.shape[0]
        t = torch.randint(1, self.n_steps+1, size=(n,1,1,1),requires_grad=False).to(self.device)
        randn = torch.randn_like(x_0,requires_grad=False)
        x_t = x_0 * torch.sqrt(self.alpha_bar[t]) + randn * torch.sqrt(1-self.alpha_bar[t])
        return x_t, t
    
    def to_T(self, x_0):
        """
        Converts samples of x_0 directly to x_T, which should be Gaussian noise
        """
        n = x_0.shape[0]
        t = (self.n_steps * torch.ones(n, 1, 1, 1).int()).to(self.device)
        x_t = x_0 * torch.sqrt(self.alpha_bar[t]) + torch.randn_like(x_0,requires_grad=False) * torch.sqrt(1-self.alpha_bar[t])
        return x_t
    
    def forward_high_var(self, x0):
        """
        Does the forward pass of the model from Part 1
        """
        n = x0.shape[0]
        ts = torch.randint(1, self.n_steps+1, size=(n,1,1,1),requires_grad=False).to(self.device)
        xt_minus, xt = self.forward_process(x0, ts)
        
        return xt_minus, xt, self.predict_mu(xt, ts), ts
    
        
    def mse_loss_high_var(self, xtminus, mu_t, t):
        """
        Loss function for Part 1. 
        """
        t = t.squeeze()
        diff = xtminus - mu_t
        numerator = torch.mean(diff*diff,dim=(1,2,3))
        return numerator / (2 * (1 - self.alpha[t.reshape(-1)]))
    
    def decode(self, x_T):
        """
        Given an $x_T$, that is, pure Gaussian noise, this runs through the reverse 
        process removing noise
        """
        with torch.no_grad():
            x_i = x_T
            for t in range(self.n_steps, 0, -1):
                expanded_t = (t * torch.ones(x_i.shape[0], 1, 1, 1).int()).to(self.device)
                mu = self.predict_mu(x_i, expanded_t)
                z = torch.randn_like(x_i).to(self.device)
                std = torch.sqrt(1 - self.alpha[expanded_t])
                x_i = mu + std * z
            return x_i
    
    def sample(self, num_samples):
        """
        Generates random noise, runs through the reverse process, and outputs the results
        Satisfies the requirements of Part 1 (h)
        """
        x = torch.randn((num_samples,1,28,28),requires_grad=False).to(self.device)
        return self.decode(x)


# Opt_Diffusion implements a different loss function: the 
# last equation in ../theory/writeup.pdf
# This loss function tends to work better than the other one
class Opt_Diffusion(nn.Module):
    def __init__(self, model, n_steps, device, min_beta, max_beta):
        super().__init__()
        
        self.n_steps = n_steps
        self.beta = torch.linspace(min_beta, max_beta, n_steps)
        self.beta = torch.hstack([torch.Tensor([0]), self.beta]).to(device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, 0)
        self.device = device
        self.mu = model.to(device)
    
    def predict_eps(self, xt, t):
        """
        Utilizes the neural network to determine $e(x_t, t; \theta)$
        """
        t = t.reshape(-1, 1)
        return self.mu.forward(xt, t)
        
    def forward_direct(self, x_0):
        """
        Given $x_0$ directly computes $x_t$ and returns the randomly sampled
        $t$ as well. $t$ ranges from $1$ to $N-steps$.
        """
        # Through implementing a lower- loss function; no longer need to sample
        # x_(t-1). This function finds x_t
        n = x_0.shape[0]
        t = torch.randint(1, self.n_steps+1, size=(n,1,1,1),requires_grad=False).to(self.device)
        noise = torch.randn_like(x_0,requires_grad=False)
        x_t = x_0 * torch.sqrt(self.alpha_bar[t]) +  noise * torch.sqrt(1-self.alpha_bar[t])
        return x_t, t, noise
    
    def to_T(self, x_0):
        """
        Converts samples of x_0 directly to x_T. Should be Gaussian noise
        """
        n = x_0.shape[0]
        t = (self.n_steps * torch.ones(n, 1, 1, 1, requires_grad=False).int()).to(self.device)
        x_t = x_0 * torch.sqrt(self.alpha_bar[t]) + torch.randn_like(x_0,requires_grad=False) * torch.sqrt(1-self.alpha_bar[t])
        return x_t
            
    def loss(self, x_0, x_t, t, true_eps): 
        """
        Loss function for Part 2. The network now predicts the noise instead of $\mu$.
        """
        left_term = torch.div(1 - self.alpha[t], 2 * self.alpha[t] * (1 - self.alpha_bar[t]))
        predicted_eps = self.predict_eps(x_t, t)
        diff = true_eps - predicted_eps
        numerator = torch.sum(diff * diff, dim=(1,2,3))
        return left_term * numerator
        
    def decode(self, x):
        """
        Given an $x_T$, that is, pure Gaussian noise, this runs through the reverse 
        process removing noise
        """
        with torch.no_grad():
            for t in range(self.n_steps, 0, -1):
                expanded_t = (t * torch.ones(x.shape[0], 1, 1, 1, requires_grad=False).int()).to(self.device)
                eps = self.predict_eps(x, expanded_t)
                mu_coef = 1 / torch.sqrt(self.alpha[expanded_t])
                mu_term = x - ((1 - self.alpha[expanded_t]) / torch.sqrt(1 - self.alpha_bar[expanded_t])) * eps
                mu = mu_coef * mu_term
                z = torch.randn_like(x, requires_grad=False).to(self.device)
                std = torch.sqrt(1 - self.alpha[expanded_t])
                x = mu + std * z
            x = torch.clamp(x, max=1)
            return x
    
    def sample(self, num_samples):
        """
        Generates random noise, runs through the reverse process, and outputs the results
        """

        x = torch.randn((num_samples,1,28,28),requires_grad=False).to(self.device)
        return self.decode(x)
