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
        def __init__(self, in_dim: int, out_dim: int) -> None:
            super().__init__()
            self.W_q = nn.Parameter(torch.rand(in_dim, out_dim))
            self.W_k = nn.Parameter(torch.rand(in_dim, out_dim))
            self.W_v = nn.Parameter(torch.rand(in_dim, out_dim))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            Q = x @ self.W_q
            K = x @ self.W_k
            V = x @ self.W_v
        
            d_k = K.shape[1]
            # Attention weights
            A = torch.softmax((Q @ K.T) / d_k**0.5, dim=-1)

            # context vectors
            return A @ V

    return (SelfAttentionV1,)


@app.cell
def _(SelfAttentionV1, inputs, torch):
    torch.manual_seed(123)
    in_dim = 3
    out_dim = 2

    sa_v1 = SelfAttentionV1(in_dim, out_dim)
    print(sa_v1(inputs))
    return


if __name__ == "__main__":
    app.run()
