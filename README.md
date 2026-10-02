# FCBNet: A Parameter-efficient Convolutional Approach for Camouflaged Weed Detection in Multispectral Aerial Imagery

<div align="center">

[Leo Thomas Ramos](https://www.linkedin.com/in/leo-thomas-ramos/), [Angel D. Sappa](https://es.linkedin.com/in/angel-sappa-61532b17)<br/>
Correspondence: ltramos@cvc.uab.cat

Computer Vision Center (CVC), Universitat Autònoma de Barcelona (UAB) 

</div>

<div align="center">
<img src="./assets/fcbnet_teaser.png" width="100%"/>
</div>

## Announcements

- FCBNet has been accepted at CVPR Workshops 2026
- The paper introduces an efficient frozen-backbone strategy for camouflaged weed segmentation
- Official implementation and model code are available in this repository

## About the project

FCBNet is a parameter-efficient semantic segmentation model for camouflaged weed detection in aerial imagery. It combines a fully frozen ConvNeXt encoder with lightweight **Feature Correction Blocks (FCB)** and an FPN-style decoder. The main design goal is to keep strong segmentation performance while drastically reducing trainable parameters and training cost.

## Installation

```bash
pip install -r requirements.txt
```

or install directly from source:

```bash
pip install .
```

## Quick usage

FCBNet can be instantiated directly from a single file:

```python
from fcbnet import FCBNet

model = FCBNet(
    in_channels=3,
    num_classes=4,
)
```

The default arguments are already configured with the best-performing setup from our experiments.  
You can still tune parameters such as `variant`, `pretrained`, `freeze_backbone`, `fpn_dim`, and FCB-related settings if needed.

## Input and output

- **Input:** tensor of shape `(B, C, H, W)`
- **Output:** logits of shape `(B, num_classes, H, W)`

Example forward pass:

```python
import torch
from fcbnet import FCBNet

model = FCBNet(in_channels=3, num_classes=4)
x = torch.randn(2, 3, 256, 256)
y = model(x)
print(y.shape)  # (2, 4, 256, 256)
```
## Datasets

To evaluate FCBNet, we use two camouflaged weed detection/segmentation datasets based on aerial imagery:

- [**WeedBananaCOD**](https://cod-espol.github.io/COD-Weeds/)
- [**WeedMap**](https://github.com/viariasv/weedMap)

## Main results

### WeedBananaCOD

| Model | Modality | mIoU | Notes |
|------|------|------|------|
| U-Net | RGB / MS | Baseline | Reference model |
| DeepLabV3+ | RGB / MS | Strong baseline | Competitive segmentation |
| SegFormer | RGB / MS | Transformer baseline | Higher complexity |
| WeedSense / SK-U-Net | RGB / MS | Recent methods | Task-specific baselines |
| **FCBNet (ours)** | RGB / MS | **> 85%** | Best balance of performance and efficiency |

### WeedMap

| Model | Modality | mIoU | Notes |
|------|------|------|------|
| U-Net | RGB / MS | Baseline | Reference model |
| DeepLabV3+ | RGB / MS | Strong baseline | Competitive segmentation |
| SegFormer | RGB / MS | Transformer baseline | Higher complexity |
| WeedSense / SK-U-Net | RGB / MS | Recent methods | Task-specific baselines |
| **FCBNet (ours)** | RGB / MS | **> 85%** | Superior overall trade-off |

## Why FCBNet is efficient

- Fully frozen ConvNeXt backbone
- Lightweight Feature Correction Blocks (pointwise + depthwise convolutions)
- Compact FPN decoder and segmentation head
- Lower memory footprint and faster optimization

## Citation

If you find this work useful, please star the repository and cite:

```bibtex
@InProceedings{Ramos_2026_CVPR,
    author    = {Ramos, Leo Thomas and Sappa, Angel D.},
    title     = {A Parameter-efficient Convolutional Approach for Camouflaged Weed Detection in Multispectral Aerial Imagery},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2026},
    pages     = {1349-1358}
}
```

## License

Distributed under the GNU General Public License v3.0. See `LICENSE` for more information.
