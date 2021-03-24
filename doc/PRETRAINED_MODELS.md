# SR4RS Pre-models

This section in currently in contruction.

## Notes

- The checkpoints can be used to start the training in `train.py` (use the `--load_ckpt` parameter). You can retrieve the model parameters values from the checkpoint file name.
- The SavedModels can be used directly on remote sensing images from `sr.py` to generate high-resolution images.
- Before applying a model, check that your input image has the same spectral content/bands + in the same order as indicated in the table below. 

## Models

| Model name | Nb. patches used | Area | Source Sensor | Target Sensor | L1 loss weight | L2 loss weight | VGG loss weight (feats) | Loss type | Depth | Nb. res. blocks | Comment | Links |
| ---------- | ---------------- | -----| -------------- | ------------ | -------------- | -------------- | ----------------------- | --------- | ----- | --------------- | ------- | ----- |
| mini-mtp-2.5 | 3984 | Montpellier area, Fr. | Sentinel-2 (B4328, TOC reflectance) from THEIA Land data center | Spot-7 (B1234, corrected) | 0.0 | 1000.0 | 0.00001 ("1234") | WGAN-GP | 64 | 16 | Quickly trained model, not very nice. For testing purposes. You can apply it on Sentinel-2 images from ESA hub, though it was trained on a TOC reflectance product. | [checkpoint](https://nextcloud.inrae.fr/s/MWaqnKCsRmkQmtm) / [SavedModel](https://nextcloud.inrae.fr/s/JLsak68H2KYzPyG) |

