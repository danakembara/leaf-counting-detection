# Leaf Counting and Detection

Deep learning for plant phenotyping: counting and detecting leaves in top-down images of *Arabidopsis* plants using CNNs in PyTorch.

> Group project for Deep Learning, Wageningen University & Research.

## Results

| Task | Best model | Metric | Score |
|---|---|---|---|
| Leaf counting (regression) | AlexNet + data augmentation | Pearson r | 0.76 |
| Leaf detection | ResNet34 / VGG16 (frozen weights) + data augmentation | Average Precision | 0.38 |

**Key findings**
- Data augmentation improved performance on a small dataset (~150 training images).
- A simpler model (AlexNet) outperformed fine-tuned ResNets for leaf counting.
- Tuning the non-maximum suppression threshold mattered for overlapping leaves.

## Quick Start

```bash
git clone https://github.com/danakembara/leaf-counting-detection.git
cd leaf-counting-detection
pip install -r requirements.txt

python task_1_leaf_counting.py    # leaf counting
python task_2_leaf_detection.py   # leaf detection
```

**Data:** Download the *Ara2012* and *Ara2013-Canon* sets from the CVPPP Leaf Counting / Leaf Segmentation Challenge and place them in `data/`.

## Repository Structure

```
├── task_1_leaf_counting.py    # CNN regression for leaf counting
├── task_2_leaf_detection.py   # CNN object detection for leaves
└── README.md
```

---

# Full Report

## 1. Background

Plant phenotypes are the observable traits that result from the interaction between genes and environment. Monitoring them accurately supports better cultivation decisions and higher yields. Manual measurement is slow, labour-intensive and prone to observer bias, so image-based phenotyping with deep learning is an attractive alternative. CNNs have been applied successfully to both leaf counting [1] and leaf detection [2][3].

This project tackles two tasks on the same dataset:

1. **Leaf counting:** predict the number of leaves per plant (regression).
2. **Leaf detection:** locate each leaf with a bounding box (object detection).

For each task, several CNN architectures were compared, testing the effect of data augmentation, hyperparameter tuning and freezing pre-trained weights.

## 2. Data

Images come from the *Ara2012* and *Ara2013-Canon* sets of the CVPPP Leaf Segmentation Challenge (LSC) and Leaf Counting Challenge (LCC): top-down images of *Arabidopsis* plants with ground-truth annotations.

The main challenge is dataset size: about **150 training** and **50 test** images. To reduce overfitting, the training set was expanded with augmentation:

| Augmentation | Setting |
|---|---|
| Random crop | 70–90% of the original area, aspect ratio between 1 and 2 |
| Colour jitter | Brightness and saturation scaled to 80–90%, contrast up to 150% |
| Horizontal flip | 50% probability |

![Figure 1 – Original vs augmented images](FIGURE_1_LINK)

## 3. Methods

### Task 1 – Leaf counting

A CNN regression head predicts the leaf count.

| Model | Initialisation | Batch size | Image size | Learning rate | Epochs |
|---|---|---|---|---|---|
| AlexNet | Trained from scratch | 256 | 128 | 1e-4 | 35 |
| ResNet18 | ImageNet, fine-tuned | 256 | 128 | 5e-5 | 50 |
| ResNet50 | ImageNet, fine-tuned | 256 | 128 | 5e-5 | 50 |

- **Loss:** Mean Squared Error (MSE)
- **Optimiser:** Adam
- **Metric:** Pearson correlation coefficient (r) between predicted and true counts on the test set. Values closer to 1 indicate better agreement.

### Task 2 – Leaf detection

| Model | Batch size | Image size | Learning rate | Epochs | Weight decay |
|---|---|---|---|---|---|
| ObjectDetectorMultiScale (baseline) | 32 | 256 | 1e-4 | 35 | 1e-4 |
| ResNet18 / ResNet34 / VGG16 backbones | 32 | 128 | 1e-4 | 35 | 1e-4 |

Each backbone was trained **with and without frozen pre-trained weights (FW)**.

Hyperparameter sweeps on the baseline (one parameter varied at a time):

| Parameter | Values tested |
|---|---|
| Image size | 128, 256, 384 |
| Learning rate | 1e-2, 1e-3, 1e-4 |
| Epochs | 20, 35, 50 |
| Weight decay | 1e-2, 1e-3, 1e-4 |

- **Loss:** Cross-entropy
- **Optimiser:** Adam
- **Metric:** Average Precision (AP) on the test set
- **Non-maximum suppression (NMS):** Because many leaves overlap, the NMS threshold was set to 0.5 for AP evaluation and 0.3 for inference.

### Environment

Python 3.6 and PyTorch on Google Colab (GPU). Training took 0.5–1 hour per model for counting and 1–1.5 hours for detection.

## 4. Results

### Task 1 – Leaf counting

| Model | Original data (r) | Augmented data (r) |
|---|---|---|
| AlexNet | 0.74 | **0.76** |
| ResNet18 (fine-tuned) | [r] | [r] |
| ResNet50 (fine-tuned) | [r] | [r] |

![Figure 2 – Training curves](FIGURE_2_LINK)

- **Augmentation helped AlexNet,** raising r from 0.74 to 0.76. This is consistent with the ~2% gain reported by [1] on the same dataset.
- **Augmentation is model-dependent.** ResNet50 scored higher on the original data, although it converged faster with augmentation [1].
- **AlexNet beat both fine-tuned ResNets.** Replacing the final layer of an ImageNet model for a regression task can reduce performance [4], which may explain why the simpler network trained from scratch did better.

### Task 2 – Leaf detection

![Figure 3 – Detection examples on original and augmented data](FIGURE_3_LINK)

**Augmentation:** The baseline detector improved from AP 0.33 (original) to **AP 0.37** (augmented), a gain of 4 percentage points.

**Best value per hyperparameter** (baseline, original data):

| Parameter | Best value | AP |
|---|---|---|
| Image size | 128 | 0.32 |
| Learning rate | 1e-4 | 0.33 |
| Epochs | 35 | 0.34 |
| Weight decay | 1e-4 | 0.30 |

These settings were then used for the backbone experiments.

![Figure 4 – Hyperparameter results](FIGURE_4_LINK)

**Backbone comparison** (augmented data):

| Backbone | Fine-tuned (AP) | Frozen weights (AP) |
|---|---|---|
| ResNet18 | 0.31 | 0.26 |
| ResNet34 | **0.38** | 0.23 |
| VGG16 | 0.30 | **0.38** |

- ResNet34 (fine-tuned) and VGG16 (frozen weights) tied for the best AP of 0.38.
- Freezing weights hurt both ResNets but helped VGG16, suggesting VGG16's pre-trained features transfer well when data is scarce.

## 5. Limitations and Next Steps

- **Small dataset:** About 150 training images limits how much complex structure the models can learn.
- **Compute:** Limited GPU memory prevented testing larger backbones such as ResNet50 for detection.
- **Metrics:** Counting was evaluated only with Pearson r. Adding MAE or difference-in-count (the standard LCC metrics) would give a more direct measure of counting accuracy.
- **Next steps:** Try dedicated one-stage detectors such as YOLO, more photorealistic augmentation [2], and additional training images.

## 6. Conclusions

- **Leaf counting:** AlexNet with data augmentation performed best (r = 0.76). Augmentation helps with limited data, but its effect depends on the model.
- **Leaf detection:** ResNet34 and VGG16 with frozen weights both reached AP 0.38. Tuning the NMS threshold was important for separating overlapping leaves.

## References

[1] Gomes, D. P. S., & Zheng, L. (2020). Recent data augmentation strategies for deep learning in plant phenotyping and their significance. engrxiv.org/t3q5p

[2] Buzzy, M., Thesma, V., Davoodi, M., & Mohammadpour Velni, J. (2020). Real-time plant leaf counting using deep object detection networks. *Sensors*, 20, 6896. https://doi.org/10.3390/s20236896

[3] Kuznichov, D., Zvirin, A., Honen, Y., & Kimmel, R. (2019). Data augmentation for leaf segmentation and counting tasks in rosette plants. *CVPR Workshops*. https://doi.org/10.48550/arXiv.1903.08583

[4] Poojary, R., Raina, R., & Mondal, A. K. (2021). Effect of data-augmentation on fine-tuned CNN model performance. *International Journal of Artificial Intelligence*, 10(1), 84–92. https://doi.org/10.11591/ijai.v10.i1.pp84-92
