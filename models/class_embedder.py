import torch
import torch.nn as nn


class ClassEmbedder(nn.Module):
    """Embed class labels for classifier-free guidance.

    Reserves index ``n_classes`` as the unconditional / null token.
    During training, labels are randomly swapped to the null token
    with probability ``cond_drop_rate`` so the same UNet learns both
    ``p(x | y)`` and ``p(x)``. At inference, ``pipelines/ddpm.py``
    builds a separate null-token batch and interpolates the two
    noise predictions via the guidance scale ``w``.
    """

    def __init__(
        self,
        embed_dim: int,
        n_classes: int = 1000,
        cond_drop_rate: float = 0.1,
    ) -> None:
        super().__init__()
        # +1 slot for the unconditional / null token at index n_classes
        self.embedding = nn.Embedding(n_classes + 1, embed_dim)
        self.cond_drop_rate = cond_drop_rate
        self.num_classes = n_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]

        if self.cond_drop_rate > 0 and self.training:
            drop_mask = torch.rand(b, device=x.device) < self.cond_drop_rate
            x = torch.where(
                drop_mask,
                torch.full_like(x, self.num_classes),
                x,
            )

        return self.embedding(x)
