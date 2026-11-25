# Flower Classification with a Shallow Linear Neural Network

An educational image-classification project built with TensorFlow and Keras. It classifies flower images into five categories using a **shallow linear neural network**.

This repository is the simple/linear baseline in a pair of flower-classification projects. Its companion project uses a deeper neural-network architecture, so the repository names distinguish the approaches.

## Flower categories
- Daisy
- Dandelion
- Roses
- Sunflowers
- Tulips

## 1. Project objective
Understand the basic image-classification workflow:
1. Load images and folder-based labels.
2. Resize images to a consistent size.
3. Normalize pixel values to `[0, 1]`.
4. Flatten each image into a one-dimensional feature vector.
5. Pass the vector through a fully connected Dense layer.
6. Convert output scores into class probabilities with Softmax.
7. Train and evaluate using training and validation data.

## 2. Dataset
This project uses the [TensorFlow Flowers dataset](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz), containing approximately 3,670 JPEG images across the five flower categories.

The notebook downloads and extracts the dataset in the Colab runtime. The dataset is not stored in this repository.

## 3. Image preparation
Images are loaded using `tf.keras.utils.image_dataset_from_directory`.
- Image size: `224 × 224`
- Colour channels: RGB (3 channels)
- Training split: 80%
- Validation split: 20%
- Batch size: 32
- Pixel normalization: divide values by 255, scaling `[0, 255]` to `[0, 1]`

Each image contains `224 × 224 × 3 = 150,528` pixel values. The `Flatten` layer reshapes these into a one-dimensional feature vector.

## 4. Model architecture
```text
RGB image (224 × 224 × 3)
          |
        Flatten
          |
  Dense layer (5 units)
          |
        Softmax
          |
 Probabilities for 5 classes
```

The Dense layer calculates a linear transformation:

`logits = W · X + b`

- `X`: flattened image input
- `W`: trainable weight matrix
- `b`: trainable bias vector
- `logits`: five output scores, one per flower category

Softmax converts the five scores into probabilities that sum to 1. There is no hidden nonlinear activation between layers; Softmax is applied at the final output.

### Model configuration
| Setting | Value |
|---|---|
| Input shape | `224 × 224 × 3` |
| Flattened input size | 150,528 |
| Dense output units | 5 |
| Output activation | Softmax |
| Optimizer | Adam |
| Loss | Sparse categorical cross-entropy |
| Metric | Accuracy |
| Batch size | 32 |
| Epochs | 15 |

The Dense layer contains `(150,528 × 5) + 5 = 752,645` trainable parameters.

## 5. Technologies
- Python
- TensorFlow
- Keras (`tf.keras`)
- NumPy
- Matplotlib

Install dependencies with:
```bash
pip install -r requirements.txt
```

## 6. Run the notebook
The notebook is designed for Google Colab and was run with a T4 GPU runtime.
1. Open `notebooks/flower_classification_shallow_linear_nn.ipynb` in Google Colab.
2. Select a GPU runtime if available (Runtime → Change runtime type → T4 GPU).
3. Run notebook cells in order.
4. The notebook downloads the dataset automatically.

A GPU can accelerate computation, but this model can also run on a CPU.

## 7. Training and evaluation
The model is trained with `model.fit()` and evaluated using training and validation accuracy and loss. The notebook plots both metrics across epochs and displays example validation predictions.

Fill in the actual values from the final notebook run:
| Metric | Result |
|---|---:|
| Training accuracy | TODO |
| Validation accuracy | TODO |
| Training loss | TODO |
| Validation loss | TODO |

Compare training and validation results. A large performance gap can indicate overfitting; low performance on both can indicate underfitting.

## 8. Disadvantages and limitations of a shallow linear neural network
This architecture is useful for learning fundamentals, but it has important limitations for image classification:

1. **Limited accuracy on complex image tasks:** A single linear transformation may not capture the complex visual patterns needed to distinguish flowers reliably. Accuracy can therefore be lower than a suitably designed deep model, although actual results depend on the dataset and training setup.
2. **No hierarchy of visual features:** It does not build layers of increasingly complex features such as edges, textures, petal shapes, and object parts.
3. **Spatial structure is not explicitly modelled:** Flattening turns the image into a long vector. The Dense layer does not directly model local relationships between neighbouring pixels as convolutional layers do.
4. **Sensitivity to position and image variation:** Similar visual features in different locations correspond to different input positions. The model has no built-in translation equivariance or pooling mechanism.
5. **Limited representational capacity:** Without hidden nonlinear layers, its decision boundaries are linear in the flattened input space. Final Softmax converts scores into probabilities but does not make the feature transformation nonlinear.
6. **Many parameters for raw pixels:** The Dense layer has 752,645 trainable parameters, yet the model still lacks strong image-specific feature extraction.
7. **May underfit:** The model may be too simple to learn distinctions in a visually varied dataset, even after training.

These limitations make this project best treated as a **fundamental baseline**. CNNs and deeper architectures can learn richer representations and are generally more suitable for image classification. The companion deep-neural-network project can be compared with this baseline to study the effects of architecture and depth.

## 9. Saving the trained model
Save the model from the notebook:
```python
model.save("flower_classification_shallow_linear_nn.keras")
```
The model file is not required in the repository because it can be regenerated by running the notebook.

## 10. Repository structure
```text
flower-classification-shallow-linear-nn/
├── notebooks/
│   └── flower_classification_shallow_linear_nn.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## 11. Future work
- Compare this shallow linear baseline with the companion deep-neural-network project.
- Build a CNN and compare validation metrics.
- Experiment with batch size, optimizer settings, and training duration.
- Evaluate predictions on additional flower images.
