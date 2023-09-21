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
                 n_cat = None):
        super(DiscontinuousNeuralNetwork, self).__init__()
        assert (embd_size is None) or (embd_size is not None and isinstance(n_cat, int)), "n_cat must be an integer if embd_size is not None"
        assert depth >= 3, "depth cannot be lower than 1"

        self.activation = activation  # PyTorch activation function like nn.ReLU()



        if embd_size:
            self.embeds = nn.Embedding(n_cat, embd_size)
        else:
            self.embeds = None

        if self.embeds is not None:
            self.layers = nn.ModuleList([nn.Linear(input_dim - n_cat, n_units)])
        else:
            self.layers = nn.ModuleList([nn.Linear(input_dim, n_units)])

        for i in range(depth - 2):
            if i == 0:


        self.output_layer = nn.Linear(n_units, output_dim)



        # Define the layers
        self.dense1 = nn.Linear(input_dim, n_units)
        self.dense2 = nn.Linear(n_units, n_units)
        self.d_dense3 = DiscontinuityDense(n_units, d_units, activation = self.activation)
        self.d_dense4 = DiscontinuityDense(d_units, d_units, activation = self.activation)
        self.dense5 = nn.Linear(d_units, n_units)


        self.out = nn.Linear(n_units, output_dim)  # Linear activation for the output layer


    def embed_forward(self, x_cont, x_cat):
        assert self.embeds is not None, ""

    def forward(self, x):
        assert self.embeds is None, "for training with embedding use .embed_forward"

        return x


