from __future__ import annotations

from types import SimpleNamespace

import torch

from qwenpose.eagle_lora import EagleFeatureExtractor
from qwenpose.train_pose import iter_named_floating_tensors, synchronized_finite_check


class _DummyEagle(torch.nn.Module):
    def __init__(self, hidden_size: int = 4) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(hidden_size=hidden_size)
        )


def test_multimodal_fusion_starts_from_stage1_visual_map() -> None:
    extractor = EagleFeatureExtractor(
        _DummyEagle(),
        output_size=2,
        refiner_layers=0,
    )
    raw_maps = torch.randn(2, 4, 2, 2)
    lm_maps = torch.randn_like(raw_maps)

    fused = extractor.fuse_feature_maps(raw_maps, lm_maps)
    stage1_map = extractor.normalize_raw_feature_maps(raw_maps)

    torch.testing.assert_close(fused, stage1_map, rtol=0.0, atol=0.0)


def test_synchronized_finite_check_reports_nested_nonfinite_tensor() -> None:
    outputs = {
        "keypoints": torch.ones(2, 3),
        "refine": [torch.zeros(1), torch.tensor([float("nan")])],
    }

    finite, bad_names = synchronized_finite_check(
        iter_named_floating_tensors(outputs),
        torch.device("cpu"),
    )

    assert not finite
    assert bad_names == ["refine[1]"]


def test_synchronized_finite_check_accepts_finite_tensors() -> None:
    finite, bad_names = synchronized_finite_check(
        iter_named_floating_tensors({"x": torch.ones(2)}),
        torch.device("cpu"),
    )

    assert finite
    assert bad_names == []
