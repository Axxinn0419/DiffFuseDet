# DiffFuseDet
DiffFuseDet_code
# DiffFuseDet

Official implementation of the paper:

> **DiffFuseDet: Gradient-Frequency Enhanced Difference-Guided Fusion for Real-Time Multimodal Aerial Object Detection**
> Zixin Li, Ye Li
> School of Optical-Electrical and Computer Engineering, University of Shanghai for Science and Technology



DiffFuseDet is a visible--infrared (RGB--IR) object detector for aerial imagery. On top of a YOLOv11
dual-stream backbone it adds two modules:

| Module | Role |
| --- | --- |
| **GFEM** (Gradient--Frequency Enhancement Module) | Applied at the shallow feature-extraction stage. Combines Scharr spatial-gradient enhancement with FFT-based frequency-domain enhancement to strengthen edges, contours and structural detail under weak-texture and low-contrast conditions. |
| **DGA-Fusion** (Difference-Guided Adaptive Fusion) | Applied at each fusion scale. Gates the difference between the detail-rich early-fusion feature and the semantic-rich detection-head feature, so that complementary information is injected adaptively while redundant background responses are suppressed. |

Reported performance of the full model: **84.4 % mAP50** on DroneVehicle (43.7 FPS, 7.18 M parameters,
11.6 GFLOPs), **73.7 % mAP50** on VEDAI, and **88.3 % mAP50** on M3FD, with real-time inference on an
NVIDIA Jetson Orin NX 16 GB (35.6 FPS, TensorRT FP16).

## Repository layout

```
.
├── train.py                 # single-modality (IR) training baseline        <!-- TODO: 确认用途/数据集 -->
├── trainMM.py               # multimodal training, M3FD street scenes
├── trainMM_kfold.py         # multimodal training with 10-fold cross-validation (VEDAI)
├── valMM.py                 # validation for a trained multimodal model
├── valCOCO.py               # COCO-style evaluation on a dataset yaml
├── predictMM.py             # inference / prediction with a trained multimodal model
├── GFEM.py                  # GFEM module definition
├── visGFEM.py               # feature-response visualization for the GFEM branches (Fig. 6)
├── visMM.py                 # multimodal feature visualization
├── vis_gca.py               # Grad-CAM visualization of different module configurations (Fig. 5)
├── checkdga.py              # utility for inspecting the DGA-Fusion module
├── requirements.txt         # Python environment
├── wiki/
│   └── vis_usage.md         # documentation of the visualization arguments
└── ultralytics/             # vendored, modified Ultralytics package (adds the multimodal model
                             # class YOLOMM and the DiffFuseDet modules)
    └── cfg/models/          # model configuration files used in the paper
        ├── yolomm.yaml                          # dual-stream baseline (no GFEM, no DGA-Fusion)
        ├── yolomm_GFEM.yaml                     # baseline + GFEM
        ├── yolomm_DGAFUSION.yaml                     # baseline + DGA-Fusion
        ├── DiffFuseDet.yaml         # full DiffFuseDet
        ├── GFEM_FRE.yaml                        # GFEM internal-ablation variants
        ├── yolo11.yaml, yolo12.yaml, yolov10n.yaml, yolo26n.yaml, rtdetr-l.yaml   # comparison backbones
```



## Environment

* Ubuntu 20.04, Python 3.x, CUDA 12.1
* PyTorch 2.3.0 / torchvision 0.18.0 (CPU users: install the matching torch build first)
* Other packages: `pip install -r requirements.txt`

```bash
pip install -r requirements.txt
```

The custom model class and modules are part of the vendored `ultralytics` package in this repository;
no separate installation step is required.

<!-- TODO: 确认 requirements.txt 精简后的版本能装上（当前这份是全环境 freeze，含 conda 内部包） -->

## Datasets

Experiments use three public RGB--IR benchmarks. Download them from their official sources and
arrange each one as shown below; RGB and IR images must be paired by identical file name (stem).

| Dataset | Notes | Source |
| --- | --- | --- |
| DroneVehicle | aerial vehicle detection, images resized to 640×640 | <!-- TODO: 官方链接 --> |
| VEDAI | aerial vehicle detection, 512×512 / 1024×1024, 10-fold cross-validation | <!-- TODO: 官方链接 --> |
| M3FD | multimodal street scenes, 1024×768; no official split, we use a random 80 %/20 % train/test split | <!-- TODO: 官方链接 --> |

Example `data.yaml` (edit `path`, `train`, `val` to match your local layout):

```yaml
path: /path/to/DroneVehicle
train: images/train
val: images/val
names:
  0: car
  1: truck
  2: freight_car
  3: bus
  4: van
```

```
/path/to/DroneVehicle
├── images/{train,val}/        # RGB and IR frames, paired by file stem
└── labels/{train,val}/        # YOLO-format annotations
```

For the 10-fold VEDAI protocol, `trainMM_kfold.py` builds the per-fold yaml files automatically from
the dataset root; the reported numbers in the paper are averaged over the 10 folds.

<!-- TODO: 说明 DroneVehicle 用的训练脚本/配置（当前仓库里没有明显的 DroneVehicle 训练入口） -->

## Training

Hyperparameters used in the paper: AdamW, initial learning rate 0.01, momentum 0.937, weight decay
0.0005, 400 epochs, batch size 32 (see the paper, Sect. 4.2).

<!-- TODO: 各数据集实际用的 epochs / batch / imgsz 请与代码和论文核对后填入下表（M3FD 当前脚本是 300 epochs / batch 8 / 1024） -->

| Dataset | Script | imgsz | epochs | batch |
| --- | --- | --- | --- | --- |
| DroneVehicle | TODO | 640 | 400 | 32 |
| VEDAI (10-fold) | `trainMM_kfold.py` | 1024 | TODO | TODO |
| M3FD | `trainMM.py` | 1024 | 300 | 8 |

Each script currently reads its dataset path and model configuration from variables defined near the
top (or from the `data=` argument inside the training call). Edit those paths first, then run:

```bash
python trainMM.py          # M3FD
python trainMM_kfold.py    # VEDAI, 10-fold cross-validation
```

If you place this repository somewhere other than the original working directory, update the model
configuration path:

```python
model = YOLOMM('ultralytics/cfg/models/yolo11n_DGCAFusion_GFEM.yaml')
```

## Evaluation

```bash
python valMM.py            # mAP50 / mAP50:95 for a trained multimodal model
python valCOCO.py          # COCO-style evaluation against a dataset yaml
python valsingle.py        # single-modality baselines
python predictMM.py        # qualitative predictions
```

`Params` and `GFLOPs` reported in the paper are obtained with `thop` (included in the environment).

## Reproducing the ablation studies

| Configuration | Model config |
| --- | --- |
| Baseline (no GFEM, no DGA-Fusion) | `ultralytics/cfg/models/yolomm.yaml` |
| + GFEM only | `ultralytics/cfg/models/yolomm_GFEM.yaml` |
| + DGA-Fusion only | `ultralytics/cfg/models/yolomm_CAFM.yaml` |
| Full model | `ultralytics/cfg/models/yolo11n_DGCAFusion_GFEM.yaml` |

<!-- TODO: 核对这张对应表，并补上 GFEM 内部消融（仅空间分支 / 仅频率分支 / 两者）所用的配置文件名 -->

## Visualization

Feature-response maps and Grad-CAM results (Figs. 5–7 of the paper) are produced by:

```bash
python visGFEM.py     # spatial branch / frequency branch / fused GFEM responses
python vis_gca.py     # Grad-CAM under different module configurations
python visMM.py       # multimodal feature visualization
```

The full argument reference (`rgb_source`, `x_source`, `method`, `layers`, `modality`, `overlay`,
`top_k`, …) is documented in [`wiki/vis_usage.md`](wiki/vis_usage.md).

## Embedded deployment

The real-time results on the NVIDIA Jetson Orin NX 16 GB (JetPack 6.2, TensorRT 10.3, FP16, MAXN power
mode) are obtained by exporting the trained model to TensorRT and benchmarking it at 1024×1024 with
batch size 1.

<!-- TODO: 补上导出与测速的具体命令/脚本（工程里的 bench_realtime.py 未包含在本仓库中） -->

## License

The vendored `ultralytics/` directory is distributed under the **AGPL-3.0** license (see
`ultralytics/LICENSE`). Because this repository redistributes a modified copy of that package, the
repository as a whole must be released under AGPL-3.0-compatible terms. Please review the Ultralytics
license before any commercial use.

<!-- TODO: 若你的学校/单位有指定开源许可，在这里补充说明；不要用与 AGPL 冲突的许可 -->

## Citation

```bibtex
@article{li2026difffusedet,
  title   = {DiffFuseDet: Gradient-Frequency Enhanced Difference-Guided Fusion for Real-Time
             Multimodal Aerial Object Detection},
  author  = {Li, Zixin and Li, Ye},
  journal = {<!-- TODO: 期刊名 -->},
  year    = {<!-- TODO: 年份 -->}
}
```

## Contact

Questions about the code: <!-- TODO: 邮箱 --> (Zixin Li).
