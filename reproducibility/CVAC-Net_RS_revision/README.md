# CVAC-Net revision: partial reproducibility release

This package contains the exact UGVT train/test sequence membership and the code and metadata for the derived Griffin 2D single-object tracking protocol. It is a **partial** release. It does not include third-party images, original annotations, pretrained weights, trained CVAC-Net weights, or the complete training/inference implementation.

| Directory | Released material | Purpose |
|---|---|---|
| `ugvt_split/` | 165 training and 45 test pair names, split-reconstruction check, protocol notes | Reproduce the UGVT sequence membership without relying on filesystem enumeration order |
| `griffin_2d_sot_protocol/` | conversion and validation scripts, frozen protocol, split lists, source mapping, audit hashes | Recreate and check the Griffin-derived paired 2D tracking sequences from the upstream dataset |

The official UGVT and Griffin datasets must be obtained from their respective providers under the applicable terms. The protocol names the physical ground and aerial views explicitly.

This package is published in the CVAC-Net repository as part of the revision materials. The remaining implementation and weight files need further cleanup and provenance checks before a wider release.

The included scripts are distributed under the GPL-3.0 license in `LICENSE`. Third-party datasets and weights are not redistributed.
