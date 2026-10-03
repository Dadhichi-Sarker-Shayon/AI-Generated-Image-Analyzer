# License of the model weights

The **source code** of this repository is released under the [MIT License](LICENSE).

The **trained model weights** (`src/ai_image_analyzer/models/weights/*.onnx`, the matching `*.json` files, and the PyTorch
checkpoints published on Hugging Face) are released under the
**[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International license (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)**.

## Why the weights are not MIT

The models were trained on data whose licenses restrict commercial use or are not stated:

| Dataset | Used for | License |
|---|---|---|
| [Tiny-GenImage](https://huggingface.co/datasets/TheKernel01/Tiny-GenImage) (derived from GenImage) | training + validation | CC BY-NC-SA 4.0 (non-commercial, share-alike) |
| [Defactify Image Dataset](https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset) (real images from MS COCO) | training, validation, held-out test | no license declared on the dataset card; COCO images carry Flickr licenses |
| [Synthbuster-plus](https://huggingface.co/datasets/marco-willi/synthbuster-plus) (real images from RAISE) | held-out test only (not trained on) | no license declared on the dataset card |

Whether a trained model is a derivative of its training data is legally unsettled. We chose the most conservative reading:
the weights inherit the non-commercial / share-alike terms of the most restrictive dataset. If you need the weights for a
commercial purpose, train your own with the scripts in this repository on data you are licensed to use
(see [docs/REPORT.md](docs/REPORT.md#reproduction)).

This is a statement of the authors' intent, not legal advice.

## What this means for you

- You may use, share and adapt the weights for **non-commercial** purposes, with attribution, under the same license.
- The MIT-licensed code can be used freely, including commercially, but the bundled weights stay CC BY-NC-SA 4.0.
  Using the library with `use_trained=False` (heuristics only) involves no CC-licensed material.
