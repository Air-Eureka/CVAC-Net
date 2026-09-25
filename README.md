# CVAC-Net

This repository provides a partial **data and evaluation protocol release** for the CVAC-Net Remote Sensing revision. The CVAC-Net model implementation and trained weights are not included at this stage.

- [UGVT: exact 165 training and 45 test paired sequences, with a split-reconstruction check](reproducibility/CVAC-Net_RS_revision/ugvt_split/)
- [Griffin-derived 2D tracking: 303 training, 67 validation, and Difficult26 paired sequences, with conversion and validation scripts](reproducibility/CVAC-Net_RS_revision/griffin_2d_sot_protocol/)

The [revision package](reproducibility/CVAC-Net_RS_revision/) contains the source-to-sequence mapping and protocol metadata needed to inspect and recreate the Griffin-derived 2D sequences. The underlying UGVT and Griffin datasets must be obtained from their original providers under the applicable terms. No dataset images, annotations, pretrained weights, trained CVAC-Net weights, or CVAC-Net model source are redistributed here.
