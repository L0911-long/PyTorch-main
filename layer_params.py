import numpy as np
import torch
from lib import utils
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.nn import init


# weightParams
class LayerParams:
    def __init__(self, rnn_network: torch.nn.Module, layer_type: str, prior_log_sigma=0):
        """
        Define attributes, initialize parameters and biases
        """
        self._rnn_network = rnn_network  # RNN network
        self._params_dict = {}  # Dictionary to store parameters
        self._biases_dict = {}  # Dictionary to store biases
        self._type = layer_type  # Type of the layer

        self.prior_log_sigma = prior_log_sigma

    def get_weights(self, shape,
                    init_type='custom_normal'):  # Initialize weights using the Kaiming initialization method by default
        """
        Initialize weight parameters according to the given shape
        """
        print("init+++++++", init_type)
        if shape not in self._params_dict:
            # torch.nn.Parameter(): A Tensor with the ability to be trained
            # This function can be understood as a type conversion function, converting an untrainable Tensor to a trainable parameter
            # and binding this parameter to the module (net.parameter() will have this bound parameter, allowing it to be optimized during training)
            nn_param = torch.nn.Parameter(torch.empty(*shape, device=device))

            # print(nn_param.data.size(),'--------------------------------')
            if init_type == 'xavier':
                torch.nn.init.xavier_normal_(nn_param)  # Initialize nn_param with a normal distribution
            elif init_type == 'kaiming_normal':
                torch.nn.init.kaiming_normal_(nn_param, mode='fan_out')
            elif init_type == 'kaiming_uniform':
                torch.nn.init.kaiming_uniform_(nn_param, mode='fan_in')
            elif init_type == 'custom_normal':
                n = nn_param.data.size(0) * nn_param.data.size(1)
                nn_param.data.normal_(0, math.sqrt(2. / n))
            elif init_type == 'custom_uniform':
                n = nn_param.data.size(0) * nn_param.data.size(1)
                nn_param.data.uniform_(-math.sqrt(6. / n), math.sqrt(6. / n))

            self._params_dict[shape] = nn_param  # Add to parameters dictionary

            # .register_parameter() is similar to torch.nn.Parameter(), but it adds the parameter to the network module
            # The first parameter is the parameter name, and the second parameter is a torch.nn.Parameter object, which is a Tensor matrix
            self._rnn_network.register_parameter('{}_weight_{}'.format(self._type, str(shape)),
                                                 nn_param)
        return self._params_dict[shape]

    def get_biases(self, length, bias_start=0.0):
        """
        Initialize biases according to the given length
        """
        if length not in self._biases_dict:
            biases = torch.nn.Parameter(torch.empty(length, device=device))
            torch.nn.init.constant_(biases, bias_start)  # Fill biases with the value bias_start
            self._biases_dict[length] = biases
            self._rnn_network.register_parameter('{}_biases_{}'.format(self._type, str(length)),
                                                 biases)

        return self._biases_dict[length]


#layer type
# def _concat(x, x_):
#     x_ = x_.unsqueeze(0)
#     return torch.cat([x, x_], dim=0)
#
# def _fc(inputs, state, output_size, bias_start=0.0):
#     batch_size = inputs.shape[0]
#     inputs = torch.reshape(inputs, (batch_size * self._num_nodes, -1))
#     state = torch.reshape(state, (batch_size * self._num_nodes, -1))
#     inputs_and_state = torch.cat([inputs, state], dim=-1)
#     input_size = inputs_and_state.shape[-1]
#
#     weights = self._fc_params.get_weights((input_size, output_size))
#     value = torch.matmul(inputs_and_state, weights)
#     biases = self._fc_params.get_biases(output_size, bias_start)
#     value += biases
#     return value
#
# class FC(nn.Module):
#     def __init__(self, state, output_size, bias_start, num_nodes):
#
#         super().__init__()
#         self.state = state
#         self.output_size = output_size
#         self.bias_start = bias_start
#         self.num_nodes = num_nodes
#
#     def forward(self, inputs):
#
#         batch_size = inputs.shape[0]
#         inputs = torch.reshape(inputs, (batch_size * self.num_nodes, -1))
#         state = torch.reshape(state, (batch_size * self.num_nodes, -1))
#         inputs_and_state = torch.cat([inputs, state], dim=-1)
#         input_size = inputs_and_state.shape[-1]
#
#         return
#
# # dcrnn
# def _dconv(inputs, state, output_size, bias_start=0.0):
#     # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
#     batch_size = inputs.shape[0]
#     inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
#     state = torch.reshape(state, (batch_size, self._num_nodes, -1))
#     inputs_and_state = torch.cat([inputs, state], dim=2)
#     input_size = inputs_and_state.size(2)
#
#     x = inputs_and_state
#     x0 = x.permute(1, 2, 0)  # (num_nodes, total_arg_size, batch_size)
#     x0 = torch.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
#     x = torch.unsqueeze(x0, 0)
#
#     if self._max_diffusion_step == 0:
#         pass
#     else:
#         for support in self._supports:
#             x1 = torch.sparse.mm(support, x0)
#             x = self._concat(x, x1)
#
#             for k in range(2, self._max_diffusion_step + 1):
#                 x2 = 2 * torch.sparse.mm(support, x1) - x0
#                 x = self._concat(x, x2)
#                 x1, x0 = x2, x1
#
#     num_matrices = len(self._supports) * self._max_diffusion_step + 1  # Adds for x itself.
#     x = torch.reshape(x, shape=[num_matrices, self._num_nodes, input_size, batch_size])
#     x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, order)
#     x = torch.reshape(x, shape=[batch_size * self._num_nodes, input_size * num_matrices])
#
#     weights = self._gconv_params.get_weights((input_size * num_matrices, output_size))
#     x = torch.matmul(x, weights)  # (batch_size * self._num_nodes, output_size)
#
#     biases = self._gconv_params.get_biases(output_size, bias_start)
#     x += biases
#     # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
#     return torch.reshape(x, [batch_size, self._num_nodes * output_size])
#
# # gcn
# def _gconv(inputs, state, output_size, bias_start=0.0 ,is_trans=False):
#     # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
#     batch_size = inputs.shape[0]
#     inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
#     state = torch.reshape(state, (batch_size, self._num_nodes, -1))
#     inputs_and_state = torch.cat([inputs, state], dim=2)
#     # print(inputs_and_state.size(),'2222222222222')
#     input_size = inputs_and_state.size(2)
#
#     x = inputs_and_state
#     x0 = x.permute(1, 2, 0)  # (num_nodes, total_arg_size, batch_size)
#     x0 = torch.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
#     x = torch.unsqueeze(x0, 0) # (1, num_nodes, input_size * batch_size)
#
#     for support in self._supports:
#         x1 = torch.sparse.mm(support, x0) # L * X
#         x = self._concat(x, x1)
#
#     num_matrices = len(self._supports) + 1
#     x = torch.reshape(x, shape=[num_matrices, self._num_nodes, input_size, batch_size])
#     x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, order)
#     x = torch.reshape(x, shape=[batch_size * self._num_nodes, input_size * num_matrices])
#
#     weights = self._gconv_params.get_weights((input_size * num_matrices, output_size))
#     # weights = sparse_w.apply(weights, 0.05)
#     weights = nn.Dropout(0.05)(weights)
#     x = torch.matmul(x, weights)  # (batch_size * self._num_nodes, output_size)
#     biases = self._gconv_params.get_biases(output_size, bias_start)
#     x += biases
#     # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
#     if is_trans:
#         return torch.reshape(x, [batch_size, self._num_nodes, output_size])
#
#     return torch.reshape(x, [batch_size, self._num_nodes * output_size])
#
#
#     # gcn
# def _wugconv(inputs, state, output_size, bias_start=0.0 ,is_trans=False):
#     # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
#     batch_size = inputs.shape[0]
#     inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
#     state = torch.reshape(state, (batch_size, self._num_nodes, -1))
#     inputs_and_state = torch.cat([inputs, state], dim=2)
#     # print(inputs_and_state.size(),'2222222222222')
#     input_size = inputs_and_state.size(2)
#
#     x = inputs_and_state
#     x0 = x.permute(1, 2, 0)  # (num_nodes, total_arg_size, batch_size)
#     x0 = torch.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
#     x = torch.unsqueeze(x0, 0)  # (1, num_nodes, input_size * batch_size)
#
#     for support in self._supports:
#         x1 = torch.sparse.mm(support, x0)  # L * X
#         x = self._concat(x, x1)
#
#     num_matrices = len(self._supports) + 1
#     x = torch.reshape(x, shape=[num_matrices, self._num_nodes, input_size, batch_size])
#     x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, order)
#     x = torch.reshape(x, shape=[batch_size * self._num_nodes, input_size * num_matrices])
#
#
#     weight_mu = self._gconv_params.get_bnn_weights((input_size * num_matrices, output_size) ,is_wu=True)
#     weight_log_sigma = self._gconv_params.get_bnn_weights((input_size * num_matrices, output_size) ,is_log_sigma=True)
#     if self.weight_eps is None:
#         weights = weight_mu + torch.exp(weight_log_sigma) * torch.randn_like(weight_log_sigma)
#     else:
#         weights = weight_mu + torch.exp(weight_log_sigma) * self.weight_eps
#     # weights = sparse_w.apply(weights, 0.05)
#     # weights = nn.Dropout(0.05)(weights)
#     x = torch.matmul(x, weights)  # (batch_size * self._num_nodes, output_size)
#
#     biases_mu = self._gconv_params.get_bnn_biases(output_size, is_wu=True)
#     biases_log_sigma = self._gconv_params.get_bnn_biases(output_size, is_log_sigma=True)
#     if self.bias_eps is None:
#         biases = biases_mu + torch.exp(biases_log_sigma) * torch.randn_like(biases_log_sigma)
#     else:
#         biases = biases_mu + torch.exp(biases_log_sigma) * self.bias_eps
#     x += biases
#     # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
#     if is_trans:
#         return torch.reshape(x, [batch_size, self._num_nodes, output_size])
#
#     return torch.reshape(x, [batch_size, self._num_nodes * output_size])
#
# # chebnet
# def _chebconv(inputs, state, output_size, bias_start=0.0):
#     # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
#     batch_size = inputs.shape[0]
#     inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
#     state = torch.reshape(state, (batch_size, self._num_nodes, -1))
#     inputs_and_state = torch.cat([inputs, state], dim=2)
#     input_size = inputs_and_state.size(2)
#
#     x = inputs_and_state
#     x0 = x.permute(1, 2, 0)  # (num_nodes, total_arg_size, batch_size)
#     x0 = torch.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
#     x = torch.unsqueeze(x0, 0) # (1, num_nodes, input_size * batch_size)
#
#     L = self.get_laplacian(torch.tensor(self.adj_mx, device=device)) # [N, N]
#     mul_L = self.cheb_polynomial(L, K=2) # [K, N, N]
#     # print(mul_L.shape, x0.shape) # torch.Size([2, 82, 82]) torch.Size([82, 8320])
#     for _ in range(len(self._supports)):
#         x1 = torch.matmul(mul_L, x0) # (k, num_nodes, input_size * batch_size)
#         x1 = torch.sum(x1, dim=0) # (num_nodes, input_size * batch_size)
#         x = self._concat(x, x1)
#
#     num_matrices = len(self._supports) + 1
#     x = torch.reshape(x, shape=[num_matrices, self._num_nodes, input_size, batch_size])
#     x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, order)
#     x = torch.reshape(x, shape=[batch_size * self._num_nodes, input_size * num_matrices])
#
#     weights = self._gconv_params.get_weights((input_size * num_matrices, output_size))
#     x = torch.matmul(x, weights)  # (batch_size * self._num_nodes, output_size)
#
#     biases = self._gconv_params.get_biases(output_size, bias_start)
#     x += biases
#     # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
#     return torch.reshape(x, [batch_size, self._num_nodes * output_size])

