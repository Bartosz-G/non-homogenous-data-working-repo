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
            nn.init.zeros_(self.discontinuity)
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
