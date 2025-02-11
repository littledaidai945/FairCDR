import enum
import math
import numpy as np
import torch as th
import torch.nn.functional as F
import torch.nn as nn

import os
class ModelMeanType(enum.Enum):
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon

class GaussianDiffusion(nn.Module):
    def __init__(self, config,mean_type, noise_schedule, noise_scale, noise_min, noise_max,\
            steps, device, history_num_per_term=10, beta_fixed=True):
        self.mean_type = mean_type
        self.noise_schedule = noise_schedule
        self.noise_scale = noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.steps = steps
        self.device = device
        self.emb_size=config["diffusion"]["emb_size"]
        self.time_type=config["diffusion"]["time_type"]
        self.norm=config["diffusion"]["norm"]
        self.reweight=config["diffusion"]["reweight"]
        self.history_num_per_term = history_num_per_term
        self.Lt_history = th.zeros(steps, history_num_per_term, dtype=th.float64).to(device)
        self.Lt_count = th.zeros(steps, dtype=int).to(device)
        self.config=config
        self.save_dir = "./save/{}_{}".format(config["data"]["name"],config["target_data"]["name"])

        if config["train_type"]=="train":      
            target_all_user_embeddings=th.load(os.path.join(self.save_dir, "target_all_user_embeddings.pt")).to(device)
            target_all_item_embeddings_para=th.load(os.path.join(self.save_dir, "target_item_embeddings_para.pt")).to(device)
            self.target_all_embeddings = th.cat((target_all_user_embeddings,target_all_item_embeddings_para), dim=0)

            source_all_user_embeddings=th.load(os.path.join(self.save_dir, "source_all_user_embeddings.pt")).to(device)
            self.source_all_item_embeddings=th.load(os.path.join(self.save_dir, "source_all_item_embeddings.pt"))
            self.source_all_item_embeddings=[tensor.to(device) for tensor in self.source_all_item_embeddings]
            source_all_item_embeddings_para=th.load(os.path.join(self.save_dir, "source_item_embeddings_para.pt")).to(device)
            self.source_all_embeddings = th.cat((source_all_user_embeddings,source_all_item_embeddings_para), dim=0)
        if noise_scale != 0.:
            self.betas = th.tensor(self.get_betas(), dtype=th.float64).to(self.device)
            if beta_fixed:
                self.betas[0] = 0.00001  # Deep Unsupervised Learning using Noneequilibrium Thermodynamics 2.4.1
        
            assert len(self.betas.shape) == 1, "betas must be 1-D"
            assert len(self.betas) == self.steps, "num of betas must equal to diffusion steps"
            assert (self.betas > 0).all() and (self.betas <= 1).all(), "betas out of range"

            self.calculate_for_diffusion()

        super(GaussianDiffusion, self).__init__()
    
    def get_betas(self):
        
        if self.noise_schedule == "linear" or self.noise_schedule == "linear-var":
            start = self.noise_scale * self.noise_min
            end = self.noise_scale * self.noise_max
            if self.noise_schedule == "linear":
                return np.linspace(start, end, self.steps, dtype=np.float64)
            else:
                return betas_from_linear_variance(self.steps, np.linspace(start, end, self.steps, dtype=np.float64))
        elif self.noise_schedule == "cosine":
            return betas_for_alpha_bar(
            self.steps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        )
        elif self.noise_schedule == "binomial":  # Deep Unsupervised Learning using Noneequilibrium Thermodynamics 2.4.1
            ts = np.arange(self.steps)
            betas = [1 / (self.steps - t + 1) for t in ts]
            return betas
        else:
            raise NotImplementedError(f"unknown beta schedule: {self.noise_schedule}!")
    
    def calculate_for_diffusion(self):
        alphas = 1.0 - self.betas
        self.alphas_cumprod = th.cumprod(alphas, axis=0).to(self.device)
        self.alphas_cumprod_prev = th.cat([th.tensor([1.0]).to(self.device), self.alphas_cumprod[:-1]]).to(self.device)  # alpha_{t-1}
        self.alphas_cumprod_next = th.cat([self.alphas_cumprod[1:], th.tensor([0.0]).to(self.device)]).to(self.device)  # alpha_{t+1}
        assert self.alphas_cumprod_prev.shape == (self.steps,)

        self.sqrt_alphas_cumprod = th.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = th.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = th.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = th.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = th.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        self.posterior_log_variance_clipped = th.log(
            th.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
            self.betas * th.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * th.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )
    
    def p_sample(self, model, x_start, steps, sampling_noise=False):
        assert steps <= self.steps, "Too much steps in inference."
        if steps == 0:
            x_t = x_start
        else:
            t = th.tensor([steps - 1] * x_start.shape[0]).to(x_start.device)
            x_t = self.q_sample(x_start, t)

        indices = list(range(self.steps))[::-1]

        if self.noise_scale == 0.:
            for i in indices:
                t = th.tensor([i] * x_t.shape[0]).to(x_start.device)
                x_t = model(x_t, t)
            return x_t

        for i in indices:
            t = th.tensor([i] * x_t.shape[0]).to(x_start.device)
            out = self.p_mean_variance(model, x_t, t)
            if sampling_noise:
                noise = th.randn_like(x_t)
                nonzero_mask = (
                    (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
                )  # no noise when t == 0
                x_t = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
            else:
                x_t = out["mean"]
            del out
        return x_t
    
    def mutual_p_sample(self, model, steps, sampling_noise=False):
        x_start = [tensor.detach() for tensor in self.source_all_item_embeddings]
        is_list = isinstance(x_start, list)
        
        if not is_list:
            x_start = [x_start]
        
        assert steps <= self.steps, "Too much steps in inference."
        x_t_list = []
        
        for x in x_start:
            if steps == 0:
                x_t = x
            else:
                t = th.tensor([steps - 1] * x.shape[0]).to(x.device)
                x_t = self.q_sample(x, t)
            
            indices = list(range(self.steps))[::-1]
            with th.no_grad():
                if self.noise_scale == 0.:
                    for i in indices:
                        t = th.tensor([i] * x_t.shape[0]).to(x.device)
                        x_t = model(x_t, t)
                else:
                    for i in indices:
                        t = th.tensor([i] * x_t.shape[0]).to(x.device)
                        out = self.p_mean_variance(model, x_t, t)
                        if sampling_noise:
                            noise = th.randn_like(x_t)
                            nonzero_mask = (
                                (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
                            )  # no noise when t == 0
                            x_t = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
                        else:
                            x_t = out["mean"]
                        del out
            x_t_list.append(x_t)
        
        return x_t_list if is_list else x_t_list[0]
    
    def mutual_p_sample_para(self, model, steps, sampling_noise=False):
        x_start = self.source_all_embeddings.detach()
        is_list = isinstance(x_start, list)
        if not is_list:
            x_start = [x_start]
        
        assert steps <= self.steps, "Too much steps in inference."
        
        x_t_list = []
        
        for x in x_start:
            if steps == 0:
                x_t = x
            else:
                t = th.tensor([steps - 1] * x.shape[0]).to(x.device)
                x_t = self.q_sample(x, t)
            
            indices = list(range(self.steps))[::-1]
            with th.no_grad():
                if self.noise_scale == 0.:
                    for i in indices:
                        t = th.tensor([i] * x_t.shape[0]).to(x.device)
                        x_t = model(x_t, t)
                else:
                    for i in indices:
                        t = th.tensor([i] * x_t.shape[0]).to(x.device)
                        out = self.p_mean_variance(model, x_t, t)
                        if sampling_noise:
                            noise = th.randn_like(x_t)
                            nonzero_mask = (
                                (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
                            )  # no noise when t == 0
                            x_t = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
                        else:
                            x_t = out["mean"]
                        del out
            x_t_list.append(x_t)
        return x_t_list if is_list else x_t_list[0]
    
    def training_losses(self, model, x_start, reweight=False):
        batch_size, device = x_start.size(0), x_start.device
        ts, pt = self.sample_timesteps(batch_size, device, 'importance')
        noise = th.randn_like(x_start)
        if self.noise_scale != 0.:
            x_t = self.q_sample(x_start, ts, noise)
        else:
            x_t = x_start

        terms = {}
        model_output = model(x_t, ts)
        target = {
            ModelMeanType.START_X: x_start,
            ModelMeanType.EPSILON: noise,
        }[self.mean_type]

        assert model_output.shape == target.shape == x_start.shape

        mse = mean_flat((target - model_output) ** 2)

        if reweight == True:
            if self.mean_type == ModelMeanType.START_X:
                weight = self.SNR(ts - 1) - self.SNR(ts)
                weight = th.where((ts == 0), 1.0, weight)
                loss = mse
            elif self.mean_type == ModelMeanType.EPSILON:
                weight = (1 - self.alphas_cumprod[ts]) / ((1-self.alphas_cumprod_prev[ts])**2 * (1-self.betas[ts]))
                weight = th.where((ts == 0), 1.0, weight)
                likelihood = mean_flat((x_start - self._predict_xstart_from_eps(x_t, ts, model_output))**2 / 2.0)
                loss = th.where((ts == 0), likelihood, mse)
        else:
            weight = th.tensor([1.0] * len(target)).to(device)

        terms["loss"] = weight * loss
        terms["loss"] /= pt
        return terms

    def sample_timesteps(self, batch_size, device, method='uniform', uniform_prob=0.001):
        if method == 'importance':  # importance sampling
            if not (self.Lt_count == self.history_num_per_term).all():
                return self.sample_timesteps(batch_size, device, method='uniform')
            
            Lt_sqrt = th.sqrt(th.mean(self.Lt_history ** 2, axis=-1))
            pt_all = Lt_sqrt / th.sum(Lt_sqrt)
            pt_all *= 1- uniform_prob
            pt_all += uniform_prob / len(pt_all)

            assert pt_all.sum(-1) - 1. < 1e-5

            t = th.multinomial(pt_all, num_samples=batch_size, replacement=True)
            pt = pt_all.gather(dim=0, index=t) * len(pt_all)

            return t, pt
        
        elif method == 'uniform':  # uniform sampling
            t = th.randint(0, self.steps, (batch_size,), device=device).long()
            pt = th.ones_like(t).float()

            return t, pt
            
        else:
            raise ValueError
    
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )
    
    def q_posterior_mean_variance(self, x_start, x_t, t):
        assert x_start.shape == x_t.shape
        posterior_mean = (
            self._extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def p_mean_variance(self, model, x, t):
        B, C = x.shape[:2]
        assert t.shape == (B, )
        model_output = model(x, t)

        model_variance = self.posterior_variance
        model_log_variance = self.posterior_log_variance_clipped

        model_variance = self._extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)
        
        if self.mean_type == ModelMeanType.START_X:
            pred_xstart = model_output
        elif self.mean_type == ModelMeanType.EPSILON:
            pred_xstart = self._predict_xstart_from_eps(x, t, eps=model_output)
        else:
            raise NotImplementedError(self.mean_type)
        
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        del model_output 
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    
    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            self._extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )
    
    def SNR(self, t):
        self.alphas_cumprod = self.alphas_cumprod.to(t.device)
        return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])
    
    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
       
        arr = arr.to(timesteps.device)
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)

    def progressive_diffusion_and_denoising(self,model, x_start, steps):
        x = x_start 
        results = []

        for n in range(1, steps + 1): 
            t = th.tensor([t - 1] * x.shape[0]).to(x.device)  
            x_noisy = self.q_sample(x_start, t) 
            indices = list(range(n))[::-1]
            for i in indices:
                t = th.tensor([i] * x.shape[0]).to(x.device) 
                out = self.p_mean_variance(model, x_noisy, t)
                x_noisy = out["mean"] 
            results.append(x_noisy)

        return results
    
    def rbf_kernel(self, x, y, sigma=1.0, chunk_size=4096):
        device = x.device
        batch_size_x = x.size(0)
        batch_size_y = y.size(0)
        x_norm = (x ** 2).sum(dim=1, keepdim=True)  
        y_norm = (y ** 2).sum(dim=1, keepdim=True) 

        mean_kernel_value = 0.0
        total_elements = 0

        for i in range(0, batch_size_x, chunk_size):
            x_chunk = x[i:i + chunk_size]
            x_chunk_norm = x_norm[i:i + chunk_size]

            for j in range(0, batch_size_y, chunk_size):
                y_chunk = y[j:j + chunk_size]
                y_chunk_norm = y_norm[j:j + chunk_size]
                dist_chunk = x_chunk_norm - 2 * th.matmul(x_chunk, y_chunk.T) + y_chunk_norm.T
                kernel_chunk = th.exp(-dist_chunk / (2 * sigma ** 2)) 

                mean_kernel_value += kernel_chunk.sum()
                total_elements += kernel_chunk.numel()

        mean_kernel_value /= total_elements
        return mean_kernel_value
    
    def compute_mmd(self,x, y, sigma=1.0):
        k_xx = self.rbf_kernel(x, x, sigma) 
        k_yy = self.rbf_kernel(y, y, sigma)
        k_xy = self.rbf_kernel(x, y, sigma)

        mmd_loss = k_xx+ k_yy - 2 * k_xy
        return mmd_loss
    
    def load_result_memmap(self,index, device):
        user_result = self.user_results[index].to(device)
        item_result = self.item_results[index].to(device)
        all_result=th.cat([item_result, user_result], dim=0)
        del user_result
        del item_result
        return all_result


    def mutual_training_losses(self, model,device,reweight=False):
        target_all_embeddings = self.target_all_embeddings.detach()
        target_all_embeddings.requires_grad_(False)
        x_start=target_all_embeddings
        batch_size= target_all_embeddings.size(0)
        ts, pt = self.sample_timesteps(batch_size, device, 'importance')
        noise = th.randn_like(x_start)
        if self.noise_scale != 0.:
            x_t = self.q_sample(x_start, ts, noise)
        else:
            x_t = x_start

        terms = {}
        model_output = model(x_t, ts)
        target = {
            ModelMeanType.START_X: x_start,
            ModelMeanType.EPSILON: noise,
        }[self.mean_type]

        assert model_output.shape == target.shape == x_start.shape

        mse = mean_flat((target - model_output) ** 2)

        if reweight == True:
            if self.mean_type == ModelMeanType.START_X:
                weight = self.SNR(ts - 1) - self.SNR(ts)
                weight = th.where((ts == 0), 1.0, weight)
                loss = mse
            elif self.mean_type == ModelMeanType.EPSILON:
                weight = (1 - self.alphas_cumprod[ts]) / ((1-self.alphas_cumprod_prev[ts])**2 * (1-self.betas[ts]))
                weight = th.where((ts == 0), 1.0, weight)
                likelihood = mean_flat((x_start - self._predict_xstart_from_eps(x_t, ts, model_output))**2 / 2.0)
                loss = th.where((ts == 0), likelihood, mse)
        else:
            weight = th.tensor([1.0] * len(target)).to(device)

        terms["loss"] = weight * loss
        
        # update Lt_history & Lt_count
        for t, loss in zip(ts, terms["loss"]):
            if self.Lt_count[t] == self.history_num_per_term:
                Lt_history_old = self.Lt_history.clone()
                self.Lt_history[t, :-1] = Lt_history_old[t, 1:]
                self.Lt_history[t, -1] = loss.detach()
            else:
                try:
                    self.Lt_history[t, self.Lt_count[t]] = loss.detach()
                    self.Lt_count[t] += 1
                except:
                    print(t)
                    print(self.Lt_count[t])
                    print(loss)
                    raise ValueError

        terms["loss"] /= pt
        return terms

def betas_from_linear_variance(steps, variance, max_beta=0.999):
    alpha_bar = 1 - variance
    betas = []
    betas.append(1 - alpha_bar[0])
    for i in range(1, steps):
        betas.append(min(1 - alpha_bar[i] / alpha_bar[i - 1], max_beta))
    return np.array(betas)

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
  
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def normal_kl(mean1, logvar1, mean2, logvar2):
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + th.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )

def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, len(tensor.shape))))
