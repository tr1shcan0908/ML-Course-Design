# FR-UNet attribution

Source: https://github.com/lseventeen/FR-UNet/tree/1856bc206055fdfccde184d0ebf9ac8c3207387d

Paper: Full-Resolution Network and Dual-Threshold Iteration for Retinal Vessel and Coronary Angiograph Segmentation, IEEE JBHI 2022.

The model file is copied unchanged from models/fr_unet.py under the attached MIT license. The sole adaptation in utils.py replaces timm.models.layers.trunc_normal_ with torch.nn.init.trunc_normal_, avoiding an otherwise unnecessary dependency. No published retinal segmentation checkpoint is used.

Checkpoint/configuration provenance records the selected input preprocessing, feature scale and training budget; those differ from the original paper when specified.
