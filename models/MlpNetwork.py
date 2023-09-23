import torch
import torch.nn as nn


def pytorch_heaviside(tensor, dtype=torch.float32):
    """
    Function that returns the tensor H given by the element-wise application of the Heaviside function.
    We compute the Heaviside function H(x) = sign(relu(x) + sign(x) + 1).
    In this way, we obtain the Heaviside function: H(x) = 1 for each x>=0, H(x)=0 otherwise.
    In this way, PyTorch can compute the derivative of H using the derivatives of relu and sign functions.

    :param tensor: PyTorch tensor
    :param dtype: output type (default torch.float32)
    :return H: element-wise Heaviside function applied to input tensor
    """
    H = torch.sign(torch.relu(tensor) + torch.sign(tensor) + 1)
    H = H.to(dtype=dtype)

    return H




class DiscontinuityDense(nn.Module):
    def __init__(self, in_features, out_features, activation=None, bias=True, discontinuity_initializer=None):
        super(DiscontinuityDense, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.use_bias = bias

        # Layer's weight matrix and bias vector
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        # Layer's vector of trainable discontinuities
        self.discontinuity = nn.Parameter(torch.Tensor(out_features))

        # Weight and bias initialization
        nn.init.xavier_uniform_(self.weight)
        if bias:
            nn.init.zeros_(self.bias)

        # Discontinuity initialization
        if discontinuity_initializer is None:
            nn.init.xavier_uniform_(self.discontinuity)
        else:
            discontinuity_initializer(self.discontinuity)

    def forward(self, input):
        # W'x + b
        wx = torch.matmul(input, self.weight.t())
        if self.use_bias:
            wx += self.bias

        # H(W'x + b)
        h_wx_b = pytorch_heaviside(wx)

        # eps * H(W'x + b)
        eps_h_wx_b = self.discontinuity * h_wx_b

        # f(W'x + b)
        if self.activation is not None:
            wx = self.activation(wx)

        # L(x) = f(W'x + b) + eps * H(W'x + b)
        out = wx + eps_h_wx_b
        return out


class DiscontinuousNeuralNetwork(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_units,
                 discontinous_units,
                 depth,
                 activation=nn.ReLU,
                 embd_size=None,
                 n_cat=None,
                 dropout = None):
        super(DiscontinuousNeuralNetwork, self).__init__()
        assert (embd_size is None) or (embd_size is not None and isinstance(n_cat, int)), "n_cat must be an integer if embd_size is not None"
        assert depth >= 2, "depth cannot be lower than 2"


        layers = []
        self.embeds = None
        last_output_dim = input_dim

        # Add embedding layer if embd_size and n_cat are provided
        if embd_size is not None and n_cat is not None:
            self.embeds = nn.Embedding(n_cat, embd_size)
            last_output_dim = input_dim - n_cat + embd_size  # Adjust input dimensions

        # Input layer
        if depth >= 5:
            depth = depth - 1
            layers.append(nn.Linear(last_output_dim, hidden_units))

            if dropout is not None and 0 < dropout < 1:
                layers.append(nn.Dropout(dropout))

            layers.append(activation())
            layers.append(nn.Linear(hidden_units, hidden_units))
            last_output_dim = hidden_units

        else:
            layers.append(nn.Linear(last_output_dim, hidden_units))
            last_output_dim = hidden_units

        if dropout is not None and 0 < dropout < 1:
            layers.append(nn.Dropout(dropout))

        layers.append(activation())

        if depth >= 5:
            add_full_connected_at_the_end = True
            depth = depth - 1
        else:
            add_full_connected_at_the_end = False


        for i in range(depth - 1):
            layers.append(DiscontinuityDense(last_layer_dim, discontinous_units, activation= activation()))
            last_layer_dim = discontinous_units


        if add_full_connected_at_the_end:
            layers.append(nn.Linear(last_output_dim, hidden_units))
            last_layer_dim = hidden_units

            if dropout is not None and 0 < dropout < 1:
                layers.append(nn.Dropout(dropout))

            layers.append(activation())

        # Output layer

        layers.append(nn.Linear(last_output_dim,output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x_cont, x_cat=None):
        if self.embeds is not None and x_cat is not None:
            if x_cat.shape != (0,):
                x_cat = self.embeds(x_cat)
            x = torch.cat([x_cont, x_cat], dim=1)
        elif x_cat is not None or x_cont.shape == (0,):
            x = torch.cat([x_cont, x_cat], dim=1)
        else:
            x = x_cont

        return self.model(x)




class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_units, depth, activation=nn.ReLU,
                 regularize=None, embd_size=None, n_cat=None):
        super(MLP, self).__init__()

        layers = []
        self.embeds = None

        # Add embedding layer if embd_size and n_cat are provided
        if embd_size is not None and n_cat is not None:
            self.embeds = nn.Embedding(n_cat, embd_size)
            input_dim = input_dim - n_cat + embd_size  # Adjust input dimensions

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_units))

        if regularize == 'bn':
            layers.append(nn.BatchNorm1d(hidden_units))
        elif regularize is not None and 0 < regularize < 1:
            layers.append(nn.Dropout(regularize))

        layers.append(activation())

        # Hidden layers
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))

            if regularize == 'bn':
                layers.append(nn.BatchNorm1d(hidden_units))
            elif regularize is not None and 0 < regularize < 1:
                layers.append(nn.Dropout(regularize))

            layers.append(activation())

        # Output layer
        layers.append(nn.Linear(hidden_units, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x_cont, x_cat=None):
        if self.embeds is not None and x_cat is not None:
            if x_cat.shape != (0,):
                x_cat = self.embeds(x_cat)
            x = torch.cat([x_cont, x_cat], dim=1)
        elif x_cat is not None or x_cont.shape == (0,):
            x = torch.cat([x_cont, x_cat], dim=1)
        else:
            x = x_cont

        return self.model(x)
