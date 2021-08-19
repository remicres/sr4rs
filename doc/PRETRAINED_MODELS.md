# SR4RS Pre-trained models

## Notes

- The checkpoints can be used to start the training in `train.py` (use the `--load_ckpt` parameter). You can retrieve the training parameters values from the checkpoint file name.
- The SavedModels can be used directly on remote sensing images from `sr.py` to generate high-resolution images.
- Before applying a model, check that your input image has the same spectral content/bands + in the same order as indicated in the table below. 

## Pre-trained model for Sentinel-2

The model aims to upscale Sentinel-2 images from 10 meters to 2.5 meters, with the **four spectral bands ordered in the following**:
| order | band | 
| ----- | ---- |
| 1     | red (B4) |
| 2     | green (B3) |
| 3     | blue (B2) |
| 4     | near infrared (B8) |

The model was trained from 250 different Spot-6 and Spot-7 scenes covering the entire France Mainland, acquired during the year 2020, from march to october, and the Sentinel-2 images acquired close to the same day. We used TOC Sentinel-2 products from the THEIA Land data center.
Spot-6 and Spot-7 images were interpolated at 2.5 meters and radiometrically calibrated to match the Sentinel-2 radiometry.
Around 150k patches were used to train the model.

You can **download** the pre-trained model here:
- [SavedModel](https://nextcloud.inrae.fr/s/boabW9yCjdpLPGX)
- [Checkpoint](https://nextcloud.inrae.fr/s/LG5e5t6jdLHzAe4)
