# Emotion Classification with BERT

This project uses a pre-trained BERT model for emotion classification in textual data. The primary objective is to classify emotions into six categories: **sadness**, **anger**, **love**, **surprise**, **fear**, and **joy**.

## Project Structure

The project contains two Python files:

1. `main.py`: This is the main script responsible for orchestrating training and testing.
2. `function.py`: This contains all the auxiliary functions and classes required for data processing, model definition, training, testing, and evaluation.

## Requirements

The project requires Python 3.7+ and the following Python packages:
- `torch`
- `transformers`
- `pandas`
- `sklearn`
- `matplotlib`

To install these, run:

```bash
pip install torch transformers pandas scikit-learn matplotlib
```

## Dataset

The dataset used is in the form of text files (`train.txt`, `valid.txt`, `test.txt`), where each line is in the format `"text;label"`. The dataset paths are specified in the `CFG` class within `function.py`.

**Label Mapping**:
- sadness: 0
- anger: 1
- love: 2
- surprise: 3
- fear: 4
- joy: 5

## Configuration

Configuration settings for the entire model training and inference pipeline are in the `CFG` class. It includes the following parameters:
- **Model name**: The base model used (e.g., `bert-base-cased`).
- **Batch Size**: Batch size for training and validation.
- **Epochs**: Number of epochs.
- **Learning Rate**: Optimizer learning rate.
- **Device**: Device for training and inference.

## How to Run

### Training

To train the model, set the `is_train` variable to `True` in `main.py` and then run the script:

```bash
python main.py
```
This will train the model on the training data and evaluate on the validation set, saving the best model to the path specified in `CFG.model_save_path`.

### Testing

For testing the model using a pre-trained checkpoint:
1. Set `is_train` to `False` in `main.py`.
2. Ensure the path to the model is correctly set in `CFG.model_save_path`.
3. Run the script again:

```bash
python main.py
```

### Results

The training process generates a plot of losses and F1 scores over the epochs. The final results will also be printed in the console.

## Functions Overview

- **`create_data_loader`**: Creates and returns a data loader object.
- **`CustomModel`**: Defines the custom BERT model with classification layers.
- **`training`**: Handles model training and validation.
- **`testing`**: Conducts testing using a pre-trained model.
- **`predict_and_evaluate`**: Predicts and evaluates performance with F1 score and loss.
- **`load_model`**: Loads a pre-trained model.
- **`get_f1`**: Computes the macro F1 score.

## Contact

For any questions or suggestions, please feel free to open an issue or reach out directly via GitHub.
