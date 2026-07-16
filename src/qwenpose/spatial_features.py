from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SpatialFeatureBatch:
    """A padded BCHW tensor plus each sample's valid native spatial shape."""

    tensor: torch.Tensor
    spatial_shapes: torch.Tensor

    def __post_init__(self) -> None:
        if self.tensor.ndim != 4:
            raise ValueError(f"Spatial features must be BCHW, got {tuple(self.tensor.shape)}.")
        if self.spatial_shapes.shape != (int(self.tensor.shape[0]), 2):
            raise ValueError(
                "spatial_shapes must be [batch, 2] in (height, width) order, "
                f"got {tuple(self.spatial_shapes.shape)}."
            )
        shapes = self.spatial_shapes.to(device=self.tensor.device, dtype=torch.long)
        if bool((shapes <= 0).any().item()):
            raise ValueError("Every native spatial height and width must be positive.")
        if bool((shapes[:, 0] > self.tensor.shape[-2]).any().item()) or bool(
            (shapes[:, 1] > self.tensor.shape[-1]).any().item()
        ):
            raise ValueError("Native spatial shapes cannot exceed the padded tensor shape.")
        object.__setattr__(self, "spatial_shapes", shapes)

    @classmethod
    def from_maps(cls, maps: list[torch.Tensor]) -> "SpatialFeatureBatch":
        if not maps:
            raise ValueError("Cannot build an empty spatial feature batch.")
        normalized: list[torch.Tensor] = []
        channels = None
        for feature in maps:
            if feature.ndim == 4 and int(feature.shape[0]) == 1:
                feature = feature.squeeze(0)
            if feature.ndim != 3:
                raise ValueError(f"Each spatial feature must be CHW, got {tuple(feature.shape)}.")
            if channels is None:
                channels = int(feature.shape[0])
            elif int(feature.shape[0]) != channels:
                raise ValueError("All spatial feature maps must have the same channel count.")
            normalized.append(feature)
        max_h = max(int(feature.shape[-2]) for feature in normalized)
        max_w = max(int(feature.shape[-1]) for feature in normalized)
        padded = [
            F.pad(feature, (0, max_w - int(feature.shape[-1]), 0, max_h - int(feature.shape[-2])))
            for feature in normalized
        ]
        tensor = torch.stack(padded, dim=0)
        shapes = torch.tensor(
            [[int(feature.shape[-2]), int(feature.shape[-1])] for feature in normalized],
            device=tensor.device,
            dtype=torch.long,
        )
        return cls(tensor=tensor, spatial_shapes=shapes)

    @classmethod
    def concatenate(cls, batches: list["SpatialFeatureBatch"]) -> "SpatialFeatureBatch":
        maps: list[torch.Tensor] = []
        for batch in batches:
            for row, (height, width) in enumerate(batch.spatial_shapes.detach().cpu().tolist()):
                maps.append(batch.tensor[row, :, :height, :width])
        return cls.from_maps(maps)

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    @property
    def dtype(self) -> torch.dtype:
        return self.tensor.dtype

    @property
    def shape(self) -> torch.Size:
        return self.tensor.shape

    def new_zeros(self, *size: int) -> torch.Tensor:
        return self.tensor.new_zeros(*size)

    def detach(self) -> "SpatialFeatureBatch":
        return SpatialFeatureBatch(self.tensor.detach(), self.spatial_shapes.detach())

    def to(self, *args, **kwargs) -> "SpatialFeatureBatch":
        tensor = self.tensor.to(*args, **kwargs)
        return SpatialFeatureBatch(
            tensor=tensor,
            spatial_shapes=self.spatial_shapes.to(device=tensor.device, dtype=torch.long),
        )

    def index_select(self, dim: int, index: torch.Tensor) -> "SpatialFeatureBatch":
        if int(dim) != 0:
            raise ValueError("SpatialFeatureBatch only supports selection along the batch dimension.")
        index = index.to(device=self.tensor.device, dtype=torch.long)
        selected_tensor = self.tensor.index_select(0, index)
        selected_shapes = self.spatial_shapes.index_select(0, index)
        maps = [
            selected_tensor[row, :, : int(shape[0]), : int(shape[1])]
            for row, shape in enumerate(selected_shapes.detach().cpu().tolist())
        ]
        return SpatialFeatureBatch.from_maps(maps)

    def map_samples(self, function: Callable[[torch.Tensor], torch.Tensor]) -> "SpatialFeatureBatch":
        maps = []
        for row, (height, width) in enumerate(self.spatial_shapes.detach().cpu().tolist()):
            maps.append(function(self.tensor[row : row + 1, :, :height, :width]).squeeze(0))
        return SpatialFeatureBatch.from_maps(maps)

    def valid_mask(self) -> torch.Tensor:
        height, width = self.tensor.shape[-2:]
        rows = torch.arange(height, device=self.tensor.device)[None, :, None]
        cols = torch.arange(width, device=self.tensor.device)[None, None, :]
        return (rows < self.spatial_shapes[:, 0, None, None]) & (
            cols < self.spatial_shapes[:, 1, None, None]
        )


@dataclass(frozen=True)
class MultiScaleSpatialFeatureBatch:
    """Ordered native-grid feature levels for one image batch.

    LocatePose uses the convention ``levels=(P2, P3)`` when true MoonViT
    pre-merger features are available.  Each level keeps its own native shape
    and padding mask, while all levels share the same batch order.
    """

    levels: tuple[SpatialFeatureBatch, ...]

    def __post_init__(self) -> None:
        levels = tuple(self.levels)
        if not levels:
            raise ValueError("A multi-scale feature batch must contain at least one level.")
        batch_size = int(levels[0].shape[0])
        device = levels[0].device
        for level in levels:
            if int(level.shape[0]) != batch_size:
                raise ValueError("All multi-scale feature levels must share the same batch size.")
            if level.device != device:
                raise ValueError("All multi-scale feature levels must be on the same device.")
        object.__setattr__(self, "levels", levels)

    @classmethod
    def from_levels(
        cls,
        levels: Iterable[SpatialFeatureBatch],
    ) -> "MultiScaleSpatialFeatureBatch":
        return cls(tuple(levels))

    @property
    def device(self) -> torch.device:
        return self.levels[0].device

    @property
    def dtype(self) -> torch.dtype:
        return self.levels[0].dtype

    @property
    def shape(self) -> torch.Size:
        return self.levels[0].shape

    def detach(self) -> "MultiScaleSpatialFeatureBatch":
        return MultiScaleSpatialFeatureBatch(tuple(level.detach() for level in self.levels))

    def to(self, *args, **kwargs) -> "MultiScaleSpatialFeatureBatch":
        return MultiScaleSpatialFeatureBatch(tuple(level.to(*args, **kwargs) for level in self.levels))

    def index_select(self, dim: int, index: torch.Tensor) -> "MultiScaleSpatialFeatureBatch":
        return MultiScaleSpatialFeatureBatch(
            tuple(level.index_select(dim, index) for level in self.levels)
        )
