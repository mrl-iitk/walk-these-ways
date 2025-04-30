# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# # from tensordict import from_modules
# from copy import deepcopy

# class Ensemble(nn.Module):
# 	"""
# 	Vectorized ensemble of modules.
# 	"""

# 	def __init__(self, modules, **kwargs):
# 		super().__init__()
# 		self._n = len(modules)
		
# 		# Store the parameters of all models
# 		self.params = nn.ModuleDict({
# 			str(i): module for i, module in enumerate(modules)
# 		})


# 		# Clone the first module for reference (without parameters)
# 		self.module = deepcopy(modules[0])
# 		for p in self.module.parameters():
# 			p.requires_grad = False  # Ensure cloned module doesn’t accidentally update

# 		self._repr = str(modules[0])
# 		# # combine_state_for_ensemble causes graph breaks
# 		# self.params = from_modules(*modules, as_module=True)
# 		# with self.params[0].data.to("meta").to_module(modules[0]):
# 		# 	self.module = deepcopy(modules[0])
# 		# self._repr = str(modules[0])
# 		# self._n = len(modules)

# 	def __len__(self):
# 		return self._n

# 	def _call(self, params, *args, **kwargs):
# 		with params.to_module(self.module):
# 			return self.module(*args, **kwargs)

# 	def forward(self, *args, **kwargs):
# 		return torch.vmap(self._call, (0, None), randomness="different")(self.params, *args, **kwargs)

# 	def __repr__(self):
# 		return f'Vectorized {len(self)}x ' + self._repr


# class ShiftAug(nn.Module):
# 	"""
# 	Random shift image augmentation.
# 	Adapted from https://github.com/facebookresearch/drqv2
# 	"""
# 	def __init__(self, pad=3):
# 		super().__init__()
# 		self.pad = pad
# 		self.padding = tuple([self.pad] * 4)

# 	def forward(self, x):
# 		x = x.float()
# 		n, _, h, w = x.size()
# 		assert h == w
# 		x = F.pad(x, self.padding, 'replicate')
# 		eps = 1.0 / (h + 2 * self.pad)
# 		arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
# 		arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
# 		base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
# 		base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
# 		shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
# 		shift *= 2.0 / (h + 2 * self.pad)
# 		grid = base_grid + shift
# 		return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)


# class PixelPreprocess(nn.Module):
# 	"""
# 	Normalizes pixel observations to [-0.5, 0.5].
# 	"""

# 	def __init__(self):
# 		super().__init__()

# 	def forward(self, x):
# 		return x.div(255.).sub(0.5)


# class SimNorm(nn.Module):
# 	"""
# 	Simplicial normalization.
# 	Adapted from https://arxiv.org/abs/2204.00616.
# 	"""

# 	def __init__(self, cfg):
# 		super().__init__()
# 		self.dim = cfg.simnorm_dim

# 	def forward(self, x):
# 		shp = x.shape
# 		x = x.view(*shp[:-1], -1, self.dim)
# 		x = F.softmax(x, dim=-1)
# 		return x.view(*shp)

# 	def __repr__(self):
# 		return f"SimNorm(dim={self.dim})"


# class NormedLinear(nn.Linear):
# 	"""
# 	Linear layer with LayerNorm, activation, and optionally dropout.
# 	"""

# 	def __init__(self, *args, dropout=0., act=None, **kwargs):
# 		super().__init__(*args, **kwargs)
# 		self.ln = nn.LayerNorm(self.out_features)
# 		if act is None:
# 			act = nn.Mish(inplace=False)
# 		self.act = act
# 		self.dropout = nn.Dropout(dropout, inplace=False) if dropout else None

# 	def forward(self, x):
# 		x = super().forward(x)
# 		if self.dropout:
# 			x = self.dropout(x)
# 		return self.act(self.ln(x))

# 	def __repr__(self):
# 		repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
# 		return f"NormedLinear(in_features={self.in_features}, "\
# 			f"out_features={self.out_features}, "\
# 			f"bias={self.bias is not None}{repr_dropout}, "\
# 			f"act={self.act.__class__.__name__})"


# def mlp(in_dim, mlp_dims, out_dim, act=None, dropout=0.):
# 	"""
# 	Basic building block of TD-MPC2.
# 	MLP with LayerNorm, Mish activations, and optionally dropout.
# 	"""
# 	if isinstance(mlp_dims, int):
# 		mlp_dims = [mlp_dims]
# 	dims = [in_dim] + mlp_dims + [out_dim]
# 	mlp = nn.ModuleList()
# 	for i in range(len(dims) - 2):
# 		mlp.append(NormedLinear(dims[i], dims[i+1], dropout=dropout*(i==0)))
# 	mlp.append(NormedLinear(dims[-2], dims[-1], act=act) if act else nn.Linear(dims[-2], dims[-1]))
# 	return nn.Sequential(*mlp)


# def conv(in_shape, num_channels, act=None):
# 	"""
# 	Basic convolutional encoder for TD-MPC2 with raw image observations.
# 	4 layers of convolution with ReLU activations, followed by a linear layer.
# 	"""
# 	assert in_shape[-1] == 64 # assumes rgb observations to be 64x64
# 	layers = [
# 		ShiftAug(), PixelPreprocess(),
# 		nn.Conv2d(in_shape[0], num_channels, 7, stride=2), nn.ReLU(inplace=False),
# 		nn.Conv2d(num_channels, num_channels, 5, stride=2), nn.ReLU(inplace=False),
# 		nn.Conv2d(num_channels, num_channels, 3, stride=2), nn.ReLU(inplace=False),
# 		nn.Conv2d(num_channels, num_channels, 3, stride=1), nn.Flatten()]
# 	if act:
# 		layers.append(act)
# 	return nn.Sequential(*layers)


# def enc(cfg, out={}):
# 	"""
# 	Returns a dictionary of encoders for each observation in the dict.
# 	"""
# 	for k in cfg.obs_shape.keys():
# 		if k == 'state':
# 			out[k] = mlp(cfg.obs_shape[k][0] + cfg.task_dim, max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], cfg.latent_dim, act=SimNorm(cfg))
# 		elif k == 'rgb':
# 			out[k] = conv(cfg.obs_shape[k], cfg.num_channels, act=SimNorm(cfg))
# 		else:
# 			raise NotImplementedError(f"Encoder for observation type {k} not implemented.")
# 	return nn.ModuleDict(out)


import torch
import torch.nn as nn
import torch.nn.functional as F
# from tensordict import from_modules
from copy import deepcopy
from functorch import combine_state_for_ensemble
from tensordict.nn import TensorDictParams
from tensordict import TensorDict
from torch.func import functional_call, stack_module_state
from torch import vmap

def custom_new_unsafe(
        cls,
        source = None,
        batch_size = None,
        device = None,
        names = None,
        non_blocking = None,
        lock = False,
        nested = True,
        **kwargs,
    ) -> TensorDict:
	return None
	


def custom_from_module(module, as_module=False):
	destination = {}
	for name, param in module._parameters.items():
				if param is None:
					continue
				destination[name] = param
	for name, buffer in module._buffers.items():
				if buffer is None:
					continue
				destination[name] = buffer

	destination_set = True
	destination = custom_new_unsafe(destination, batch_size=torch.Size(()))
	
	for name, submodule in module._modules.items():
		if submodule is not None:
			subtd = custom_from_module(
				module=submodule,
			)
			if subtd is not None:
				if not destination_set:
					destination = custom_new_unsafe(batch_size=torch.Size(()))
					destination_set = True
				destination._set_str(
					name, subtd, validated=True, inplace=False, non_blocking=False
				)
	if not destination_set:
		return

	if as_module:
		from tensordict.nn.params import TensorDictParams

		return TensorDictParams(destination, no_convert=True)
	return destination
	

def custom_from_modules(*modules, as_module=False):
		param_list = [
			custom_from_module(module) for module in modules # required 
		] 
          
		with torch.no_grad():
			params = torch.stack(param_list)

        # Make sure params are params, buffers are buffers
		def make_param(param, orig_param):

			if isinstance(orig_param, nn.Parameter):
				return nn.Parameter(param.detach(), orig_param.requires_grad)
			return None
		params = params._fast_apply(make_param, param_list[0], propagate_lock=True) # required
	
		if as_module:
			from tensordict.nn import TensorDictParams
			params = TensorDictParams(params, no_convert=True)
		params.lock_()
		return params

class Ensemble(nn.Module):
    """
	Vectorized ensemble of modules.
    """

    def __init__(self, modules):
        super().__init__()
        self.params, self.buffers_ = stack_module_state(modules)
        self._repr = str(modules[0])
        self._n = len(modules)
        self.module = deepcopy(modules[0])

    def __len__(self):
        return self._n

    def _call(self, params, *args, **kwargs):
        empty_buffers = {}
        return functional_call(self.module, (params, empty_buffers), args)
        return self.fmodel(params, empty_buffers, *args, **kwargs)

    def forward(self, *args, **kwargs):
        # print("Input device(s):")
        # for i, arg in enumerate(args):
        #     if isinstance(arg, torch.Tensor):
        #         print(f"arg[{i}]:", arg.device)

        # print("\nParam device(s):")
        # for k, v in self.params.items():
        #     if isinstance(v, torch.Tensor):
        #         print(f"{k}:", v.device)

        self.params = {k: v.to(args[0].device) for k, v in self.params.items()}

        return vmap(self._call, (0, None), randomness="different")(self.params, *args, **kwargs)

        # Pack args and kwargs properly for vmap to handle
        

    def __repr__(self):
        return f'Vectorized {len(self)}x ' + self._repr
		# print("\n \n", params, "\n \n", params_dict)

		
    def __len__(self):
        return self._n

    # def _call(self, params, *args, **kwargs):
    #     with params.to_module(self.module):
    #         return self.module(*args, **kwargs)

    # def forward(self, *args, **kwargs):
    #     return torch.vmap(self._call, (0, None), randomness="different")(self.params, *args, **kwargs)

    def __repr__(self):
        return f'Vectorized {len(self)}x ' + self._repr


class ShiftAug(nn.Module):
	"""
	Random shift image augmentation.
	Adapted from https://github.com/facebookresearch/drqv2
	"""
	def __init__(self, pad=3):
		super().__init__()
		self.pad = pad
		self.padding = tuple([self.pad] * 4)

	def forward(self, x):
		x = x.float()
		n, _, h, w = x.size()
		assert h == w
		x = F.pad(x, self.padding, 'replicate')
		eps = 1.0 / (h + 2 * self.pad)
		arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
		arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
		base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
		base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
		shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
		shift *= 2.0 / (h + 2 * self.pad)
		grid = base_grid + shift
		return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)


class PixelPreprocess(nn.Module):
	"""
	Normalizes pixel observations to [-0.5, 0.5].
	"""

	def __init__(self):
		super().__init__()

	def forward(self, x):
		return x.div(255.).sub(0.5)


class SimNorm(nn.Module):
	"""
	Simplicial normalization.
	Adapted from https://arxiv.org/abs/2204.00616.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.dim = cfg.simnorm_dim

	def forward(self, x):
		shp = x.shape
		x = x.view(*shp[:-1], -1, self.dim)
		x = F.softmax(x, dim=-1)
		return x.view(*shp)

	def __repr__(self):
		return f"SimNorm(dim={self.dim})"


class NormedLinear(nn.Linear):
	"""
	Linear layer with LayerNorm, activation, and optionally dropout.
	"""

	def __init__(self, *args, dropout=0., act=None, **kwargs):
		super().__init__(*args, **kwargs)
		self.ln = nn.LayerNorm(self.out_features)
		if act is None:
			act = nn.Mish(inplace=False)
		self.act = act
		self.dropout = nn.Dropout(dropout, inplace=False) if dropout else None

	def forward(self, x):
		x = super().forward(x)
		if self.dropout:
			x = self.dropout(x)
		return self.act(self.ln(x))

	def __repr__(self):
		repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
		return f"NormedLinear(in_features={self.in_features}, "\
			f"out_features={self.out_features}, "\
			f"bias={self.bias is not None}{repr_dropout}, "\
			f"act={self.act.__class__.__name__})"


def mlp(in_dim, mlp_dims, out_dim, act=None, dropout=0.):
	"""
	Basic building block of TD-MPC2.
	MLP with LayerNorm, Mish activations, and optionally dropout.
	"""
	if isinstance(mlp_dims, int):
		mlp_dims = [mlp_dims]
	dims = [in_dim] + mlp_dims + [out_dim]
	mlp = nn.ModuleList()
	for i in range(len(dims) - 2):
		mlp.append(NormedLinear(dims[i], dims[i+1], dropout=dropout*(i==0)))
	mlp.append(NormedLinear(dims[-2], dims[-1], act=act) if act else nn.Linear(dims[-2], dims[-1]))
	return nn.Sequential(*mlp)


def conv(in_shape, num_channels, act=None):
	"""
	Basic convolutional encoder for TD-MPC2 with raw image observations.
	4 layers of convolution with ReLU activations, followed by a linear layer.
	"""
	assert in_shape[-1] == 64 # assumes rgb observations to be 64x64
	layers = [
		ShiftAug(), PixelPreprocess(),
		nn.Conv2d(in_shape[0], num_channels, 7, stride=2), nn.ReLU(inplace=False),
		nn.Conv2d(num_channels, num_channels, 5, stride=2), nn.ReLU(inplace=False),
		nn.Conv2d(num_channels, num_channels, 3, stride=2), nn.ReLU(inplace=False),
		nn.Conv2d(num_channels, num_channels, 3, stride=1), nn.Flatten()]
	if act:
		layers.append(act)
	return nn.Sequential(*layers)


def enc(cfg, out={}):
	"""
	Returns a dictionary of encoders for each observation in the dict.
	"""
	print(cfg.obs_shape.keys)
	for k in cfg.obs_shape.keys():
		if k == 'state':
			out[k] = mlp(cfg.obs_shape[k][0] + cfg.task_dim, max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], cfg.latent_dim, act=SimNorm(cfg))
		elif k == 'rgb':
			out[k] = conv(cfg.obs_shape[k], cfg.num_channels, act=SimNorm(cfg))
		else:
			raise NotImplementedError(f"Encoder for observation type {k} not implemented.")
	return nn.ModuleDict(out)
