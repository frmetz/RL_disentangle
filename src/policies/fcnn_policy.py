import torch
import torch.nn as nn
import torch.nn.functional as F


from policies.base_policy import BasePolicy


class FCNNPolicy(nn.Module, BasePolicy):
    """ A policy function parametrized by a fully-connected neural network.
    The model architecture uses fully-connected layers.
    After applying the non-linearity a dropout layer is applied.

    For a network with L layers the architecture will be:
    {affine - leaky-ReLU - [dropout]} x (L - 1) - affine

    The weights of the layers are initialized using Kaiming He uniform distribution.
    """

    def __init__(self, input_size, hidden_sizes, out_size, dropout_rate=0.0):
        """ Initialize a policy model.

        @param input_size (int): Size of the environment state.
        @param hidden_sizes (List[int]): A list of sizes for the hidden layers.
        @param out_size (int): Number of possible actions the agent can choose from.
        @param dropout_rate (float): Dropout probability.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.out_size = out_size
        self.dropout_rate = dropout_rate

        # Store arguments for model initialization.
        # Kwargs dict is used to save and restore the model.
        self.kwargs = dict(input_size=self.input_size,
                           hidden_sizes=self.hidden_sizes,
                           out_size=self.out_size,
                           dropout_rate=self.dropout_rate)

        # Initialize the model architecture.
        self.num_layers = len(hidden_sizes)
        self.hidden_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        fan_in = input_size
        for fan_out in hidden_sizes:
            self.hidden_layers.append(nn.Linear(fan_in, fan_out))
            self.dropout_layers.append(nn.Dropout(dropout_rate))
            fan_in = fan_out
        self.output_layer = nn.Linear(fan_out, out_size)

        # Initialize model parameters.
        for param in self.parameters():
            if len(param.shape) >= 2:
                nn.init.kaiming_uniform_(param)         # weight
            else:
                nn.init.uniform_(param, -0.01, 0.01)    # bias


    def forward(self, x):
        """ Take a mini-batch of environment states and compute scores over
        the possible actions.

        @param x (Tensor): Tensor of shape (b, q), or (b, t, q), giving the
                current state of the environment, where
                b = batch size, t = number of time steps,
                q = size of the quantum system (2 ** num_qubits).
        @return out (Tensor): Tensor of shape (batch_size, num_actions), or
                (batch_size, time_steps, num_actions), giving a score to
                every action from the action space.
        """
        out = x
        for idx in range(self.num_layers):
            out = self.hidden_layers[idx](out)
            out = F.leaky_relu(out)
            out = self.dropout_layers[idx](out)
        out = self.output_layer(out)
        return out

#