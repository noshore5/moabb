"""Residual wrappers shared by primitive builders."""

from __future__ import annotations

import torch
import torch.nn as nn

from .configs import ResidualConfig
from .regularization import DropPath


class ResidualWrapper(nn.Module):
    """Wrap a branch with shortcut, optional normalization, and branch scaling."""

    def __init__(
        self,
        branch: nn.Module,
        cfg: ResidualConfig,
        *,
        shortcut: nn.Module | None = None,
        norm_in: nn.Module | None = None,
        norm_out: nn.Module | None = None,
        scale_shape: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        if cfg.rezero and cfg.layer_scale is not None:
            raise ValueError("rezero and layer_scale cannot both be enabled.")
        if cfg.scale < 0.0:
            raise ValueError("scale must be non-negative.")

        self.branch = branch
        self.cfg = cfg
        self.shortcut = shortcut if shortcut is not None else nn.Identity()
        self.norm_in = norm_in
        self.norm_out = norm_out
        self.drop_path = DropPath(cfg.drop_path)
        self.fixed_scale = float(cfg.scale)

        self.rezero = (
            nn.Parameter(torch.zeros(())) if cfg.rezero else None
        )
        self.layer_scale = None
        if cfg.layer_scale is not None:
            if scale_shape is None:
                raise ValueError("layer_scale requires scale_shape.")
            self.layer_scale = nn.Parameter(
                torch.full(scale_shape, float(cfg.layer_scale))
            )

        if cfg.norm_position in {"pre", "sandwich"} and self.norm_in is None:
            raise ValueError(f"norm_position={cfg.norm_position!r} requires norm_in.")
        if cfg.norm_position in {"post", "sandwich"} and self.norm_out is None:
            raise ValueError(f"norm_position={cfg.norm_position!r} requires norm_out.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.norm_position == "none":
            branch_out = self.branch(x)
            out = self._add_residual(self.shortcut(x), branch_out)
        elif self.cfg.norm_position == "pre":
            if self.norm_in is None:
                raise RuntimeError("norm_in is not initialized.")
            branch_out = self.branch(self.norm_in(x))
            out = self._add_residual(self.shortcut(x), branch_out)
        elif self.cfg.norm_position == "post":
            if self.norm_out is None:
                raise RuntimeError("norm_out is not initialized.")
            branch_out = self.branch(x)
            out = self._add_residual(self.shortcut(x), branch_out)
            out = self.norm_out(out)
        elif self.cfg.norm_position == "sandwich":
            if self.norm_in is None or self.norm_out is None:
                raise RuntimeError("sandwich residual norms are not initialized.")
            branch_out = self.norm_out(self.branch(self.norm_in(x)))
            out = self._add_residual(self.shortcut(x), branch_out)
        else:
            raise ValueError(f"Unsupported norm_position: {self.cfg.norm_position!r}")

        return out

    def _add_residual(
        self,
        shortcut_out: torch.Tensor,
        branch_out: torch.Tensor,
    ) -> torch.Tensor:
        scaled_branch = self._scale_branch(branch_out)
        if shortcut_out.shape != scaled_branch.shape:
            raise ValueError(
                "Residual branch and shortcut shapes must match exactly, got "
                f"{tuple(scaled_branch.shape)} and {tuple(shortcut_out.shape)}."
            )
        return shortcut_out + scaled_branch

    def _scale_branch(self, branch_out: torch.Tensor) -> torch.Tensor:
        out = branch_out
        if self.rezero is not None:
            out = out * self.rezero
        if self.layer_scale is not None:
            out = out * self.layer_scale
        out = out * self.fixed_scale
        out = self.drop_path(out)
        return out
