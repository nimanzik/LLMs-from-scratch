import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import torch
    import torch.nn as nn

    return nn, torch


@app.cell
def _(torch):
    inputs = torch.tensor(
      [[0.43, 0.15, 0.89], # Your     (x^1)
       [0.55, 0.87, 0.66], # journey  (x^2)
       [0.57, 0.85, 0.64], # starts   (x^3)
       [0.22, 0.58, 0.33], # with     (x^4)
       [0.77, 0.25, 0.10], # one      (x^5)
       [0.05, 0.80, 0.55]] # step     (x^6)
    )
    return (inputs,)


@app.cell
def _(nn, torch):
    class SelfAttentionV1(nn.Module):
        def __init__(self, d_in: int, d_out: int) -> None:
            super().__init__()
            self.W_q = nn.Parameter(torch.rand(d_in, d_out))
            self.W_k = nn.Parameter(torch.rand(d_in, d_out))
            self.W_v = nn.Parameter(torch.rand(d_in, d_out))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            Q = x @ self.W_q
            K = x @ self.W_k
            V = x @ self.W_v
        
            d_k = K.shape[1]
            # Attention weights
            A = torch.softmax((Q @ K.T) / d_k**0.5, dim=-1)

            # context vectors
            return A @ V


    class SelfAttentionV2(nn.Module):
        def __init__(self, d_in: int, d_out: int, qkv_bias: bool = False) -> None:
            super().__init__()
            self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            Q = self.W_q(x)
            K = self.W_k(x)
            V = self.W_v(x)
        
            d_k = K.shape[1]
            # Attention weights
            A = torch.softmax((Q @ K.T) / d_k**0.5, dim=-1)

            # context vectors
            return A @ V

    return SelfAttentionV1, SelfAttentionV2


@app.cell
def _(SelfAttentionV1, inputs, torch):
    d_in = 3
    d_out = 2

    torch.manual_seed(123)
    sa_v1 = SelfAttentionV1(d_in, d_out)
    print(sa_v1(inputs))
    return d_in, d_out, sa_v1


@app.cell
def _(SelfAttentionV2, d_in, d_out, inputs, torch):
    torch.manual_seed(789)
    sa_v2 = SelfAttentionV2(d_in, d_out)
    print(sa_v2(inputs))
    return (sa_v2,)


@app.cell
def _(inputs, nn, sa_v1, sa_v2):
    sa_v1.W_q = nn.Parameter(sa_v2.W_q.weight.T)
    sa_v1.W_k = nn.Parameter(sa_v2.W_k.weight.T)
    sa_v1.W_v = nn.Parameter(sa_v2.W_v.weight.T)
    print(sa_v1(inputs))
    return


if __name__ == "__main__":
    app.run()
