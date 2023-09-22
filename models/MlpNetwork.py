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
                 activation,
                 depth:int,
                 input_dim:int,
                 output_dim:int,
                 n_units:int,
                 d_units:int,
                 embd_size = None,
                 n_cat = None,
                 dropout = None):
        super(DiscontinuousNeuralNetwork, self).__init__()
        assert (embd_size is None) or (embd_size is not None and isinstance(n_cat, int)), "n_cat must be an integer if embd_size is not None"
        assert depth >= 3, "depth cannot be lower than 2"

        self.activation = activation  # PyTorch activation function like nn.ReLU()

        if embd_size:
            self.embeds = nn.Embedding(n_cat, embd_size)
        else:
            self.embeds = None

        if self.embeds is not None:
            self.layers = nn.ModuleList([nn.Linear(input_dim - n_cat + embd_size, n_units)])
            last_layer_dim = n_units

            if dropout is not None:
                self.layers.append(nn.Dropout(dropout))

        else:
            self.layers = nn.ModuleList([nn.Linear(input_dim, n_units)])
            last_layer_dim = n_units

            if dropout is not None:
                self.layers.append(nn.Dropout(dropout))

        if depth >= 5:
            self.layers.append(nn.Linear(last_layer_dim, n_units))
            last_layer_dim = n_units
            depth = depth - 1

            if dropout is not None:
                self.layers.append(nn.Dropout(dropout))


        if depth >= 5:
            add_full_connected_at_the_end = True
            depth = depth - 1
        else:
            add_full_connected_at_the_end = False


        for i in range(depth - 2):
            self.layers.append(DiscontinuityDense(last_layer_dim, d_units, activation= self.activation))
            last_layer_dim = d_units

        if add_full_connected_at_the_end:
            self.layers.append(nn.Linear(last_layer_dim, n_units))
            last_layer_dim = n_units

            if dropout is not None:
                self.layers.append(nn.Dropout(dropout))


        self.output_layer = nn.Linear(last_layer_dim, output_dim)




    def embed_forward(self, x_cont, x_cat):
        assert self.embeds is not None, "for non_embedded output, use forward()"
        x_cat_embed = self.embeds(x_cat)
        x = torch.cat([x_cont, x_cat_embed], dim=1)

        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                x = self.activation(layer(x))
            elif isinstance(layer, DiscontinuityDense):
                x = layer(x)
            else:
                x = layer(x)  # Handles Dropout

        return self.output_layer(x)


    def forward(self, x):
        assert self.embeds is None, "for training with embedding use .embed_forward"

        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                x = self.activation(layer(x))
            elif isinstance(layer, DiscontinuityDense):
                x = layer(x)
            else:
                x = layer(x)  # Handles Dropout

        return self.output_layer(x)




class MLP(nn.Module):
    def __init__(self,
                 activation,
                 depth:int,
                 input_dim:int,
                 output_dim:int,
                 n_units:int,
                 embd_size = None,
                 n_cat = None,
                 dropout = None):
        super(MLP, self).__init__()
        assert (embd_size is None) or (embd_size is not None and isinstance(n_cat, int)), "n_cat must be an integer if embd_size is not None"
        assert depth >= 2, "depth cannot be lower than 2"


        self.activation = activation  # PyTorch activation function like nn.ReLU()

        if embd_size:
            self.embeds = nn.Embedding(n_cat, embd_size)
        else:
            self.embeds = None


        if self.embeds is not None:
            self.layers = nn.ModuleList([nn.Linear(input_dim - n_cat + embd_size, n_units)])
            last_layer_dim = n_units

            if dropout is not None:
                self.layers.append(nn.Dropout(dropout))

        else:
            self.layers = nn.ModuleList([nn.Linear(input_dim, n_units)])
            last_layer_dim = n_units

            if dropout is not None:
                self.layers.append(nn.Dropout(dropout))


        for i in range(depth - 2):
            self.layers.append(nn.Linear(n_units, n_units))

            if dropout is not None:
                self.layers.append(nn.Dropout(dropout))


        self.output_layer = nn.Linear(n_units, output_dim)


    def embed_forward(self, x_cont, x_cat):
        assert self.embeds is not None, "for non_embedded output, use forward()"

        x_cat_embed = self.embeds(x_cat)
        x = torch.cat([x_cont, x_cat_embed], dim=1)

        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                x = self.activation(layer(x))
            else:
                x = layer(x)  # Handles Dropout

        return self.output_layer(x)

    def forward(self, x):
        assert self.embeds is None, "for training with embedding use .embed_forward"

        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                x = self.activation(layer(x))
            else:
                x = layer(x)  # Handles Dropout

        return self.output_layer(x)