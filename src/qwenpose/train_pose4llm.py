"""LocatePose training entry that keeps LocateAnything's autoregressive box head.

The implementation intentionally reuses the current pose architecture, data
pipeline, high-resolution pyramid, BoxDN and keypoint-DN from ``train_pose``.
The companion ``scripts/locatepose4llm.sh`` selects the restored LLM-box paths
without changing the defaults used by ``scripts/locatepose.sh``.
"""

from qwenpose.train_pose import main


if __name__ == "__main__":
    main()
