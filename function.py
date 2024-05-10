import os
import torch
from transformers import AutoTokenizer
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
# from sklearn.metrics import f1_score, precision_score, recall_score


class CFG:
    """
    Configuration settings for the entire model training and inference pipeline.
    """

    # Basic model configuration
    model_name = 'bert-base-cased'  # Name of the model to use
    tokenizer = AutoTokenizer.from_pretrained(model_name)  # Initialize tokenizer
    max_len = 94  # Maximum length of tokens
    num_classes = 6  # Number of target classes
    batch_size = 32  # Batch size for training and validation
    epochs = 20  # Number of epochs to train
    learning_rate = 2e-5  # Learning rate for the optimizer
    eps = 1e-8  # Epsilon for the optimizer
    seed = 42  # Seed for reproducibility
    patience = 5  # Patience for training
    
    # Paths for data and model saving
    train_data_path = 'Datasets/emotions_dataset_for_nlp/train.txt'  # Path to training data
    valid_data_path = 'Datasets/emotions_dataset_for_nlp/valid.txt'  # Path to validation data
    test_data_path = 'Datasets/emotions_dataset_for_nlp/test.txt'  # Path to test data
    model_save_path = './model.pth'  # Directory to save trained models
    results_save_path = './results/'  # Directory to save output results

    # Device configuration
    cuda = 0
    device = f'cuda:{cuda}' if torch.cuda.is_available() else 'cpu'  # Automatically use GPU if available
    # device = "mps" if torch.backends.mps.is_available() else "cpu"


class TextDataset(Dataset):
    """
    General Dataset class for handling text data for training, validation, and testing.
    Each line in the file is expected to be in the format "text;label".
    """
    def __init__(self, file_path, max_len, tokenizer):
        """
        Initializes the Dataset object.
        """
        self.data = pd.read_csv(file_path, delimiter=';', header=None, names=['text', 'label'])
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.texts = self.data['text']

        # Mapping labels from string to integer
        self.label_mapping = {'sadness': 0, 'anger': 1, 'love': 2, 'surprise': 3, 'fear': 4, 'joy': 5}
        self.labels = self.data['label'].map(self.label_mapping)

    def __len__(self):
        """
        Returns the number of items in the dataset.
        """
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Retrieves a single item from the dataset.
        """
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()

        # Convert label to tensor, ensuring it is an integer
        label = torch.tensor(label, dtype=torch.long)  # This should not raise an error now

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': label}

def create_data_loader(data_path, tokenizer, batch_size, is_train=True):
    """
    Creates DataLoader for training or testing.
    :param data_path: str, path to data file.
    :param tokenizer: Tokenizer object.
    :param batch_size: int, size of the batch.
    :param is_train: bool, flag indicating if the loader is for training or testing.
    :return: DataLoader object.
    """
    dataset = TextDataset(data_path, CFG.max_len, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=is_train, pin_memory=True)


class CustomModel(nn.Module):
    """
    Defines a custom neural network model that integrates a pre-trained BERT model with a classification head.
    """
    def __init__(self, pretrained_model_name=CFG.model_name, num_classes=CFG.num_classes):
        """
        Initializes the model by loading a pre-trained BERT and adding a linear classification layer.
        :param pretrained_model_name: str, name of the pre-trained model to load.
        :param num_classes: int, number of classes for the classification task.
        """
        super(CustomModel, self).__init__()
        # Load pre-trained model configuration and model from HuggingFace's transformers
        self.config = AutoConfig.from_pretrained(pretrained_model_name)
        self.bert = AutoModel.from_pretrained(pretrained_model_name, config=self.config)
        # Add a dropout layer for regularization
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        # Add a linear layer for classification
        # self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size//2),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(self.config.hidden_size//2, num_classes),
        )
        

    def forward(self, input_ids, attention_mask):
        """
        Defines the forward pass of the model.
        :param input_ids: tensor, encoded input ids from the tokenizer.
        :param attention_mask: tensor, attention mask from the tokenizer.
        :return: tensor, logits from the classifier.
        """
        # Pass inputs through BERT model
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Extract the last hidden state of the token `[CLS]` for classification
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        # Pass the output through the classifier to get logits
        logits = self.classifier(cls_output)
        return logits


def get_f1(y_trues, y_preds):
    y_trues, y_preds = y_trues.cpu(), y_preds.cpu()
    y_predicted = y_preds.argmax(axis=1)  # Convert probabilities to class predictions
    macro_f1 = f1_score(y_trues, y_predicted, average='macro')
    return macro_f1


def predict_and_evaluate(model, data_loader, device, criterion):
    model.eval()
    total_outputs = []
    total_labels = []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data['input_ids'].to(device), data['labels'].to(device)
            attention_mask = data['attention_mask'].to(device)
            outputs = model(inputs, attention_mask)

            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)  # Accumulate scaled loss
            total_samples += labels.size(0)
            
            total_outputs.append(outputs)
            total_labels.append(labels)

    total_outputs = torch.cat(total_outputs, dim=0)  # Concatenate all output tensors
    total_labels = torch.cat(total_labels, dim=0)  # Concatenate all label tensors
    
    # Convert outputs to predicted labels
    avg_loss = total_loss / total_samples
    avg_f1 = get_f1(total_labels, total_outputs)  # Calculate F1 on the entire set

    return avg_loss, avg_f1


def training(model, train_loader, valid_loader, test_loader=None, device=CFG.device, patience=CFG.patience, plot_save_path="training_results.png"):
    """
    Function to handle the training and validation loop with dual axis plot.
    """
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=CFG.learning_rate, eps=CFG.eps)

    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=CFG.epochs, eta_min=CFG.learning_rate * 1e-2)

    # Loss Criterion
    criterion = nn.CrossEntropyLoss(reduction="mean")

    best_f1 = 0
    no_improve = 0  # Counter for early stopping

    # Initialize lists to hold training and validation metrics for plotting
    train_losses = []
    train_f1s = []
    valid_losses = []
    valid_f1s = []

    print("Training Starts!")
    
    # Training Loop
    for epoch in range(CFG.epochs):
        total_loss = 0
        total_f1 = 0
        for data in train_loader:
            model.train()
            inputs, labels = data['input_ids'].to(device), data['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs, data['attention_mask'].to(device))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_f1 += get_f1(labels, outputs)

        scheduler.step()  # Adjust the learning rate using the scheduler

        avg_train_loss = total_loss / len(train_loader)
        avg_train_f1 = total_f1 / len(train_loader)

        avg_val_loss, avg_val_f1 = predict_and_evaluate(model, valid_loader, device, criterion)

        # Save the metrics to the lists
        train_losses.append(avg_train_loss)
        train_f1s.append(avg_train_f1)
        valid_losses.append(avg_val_loss)
        valid_f1s.append(avg_val_f1)

        print(f"Epoch {epoch+1}/{CFG.epochs}, Train Loss: {avg_train_loss:.4e}, Train F1: {avg_train_f1:.3f}, Valid Loss: {avg_val_loss:.4e}, Valid F1: {avg_val_f1:.3f}")

        # Save the model
        if best_f1 < avg_val_f1:  # Saving the best model w.r.t the score 
            best_f1 = avg_val_f1
            best_model = model
            torch.save(best_model.state_dict(), CFG.model_save_path)
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:  # Check if no improvement in last 'n' epochs
            print(f"Early stopping as there's no improvement in Valid F1 = {best_f1:.3f}.")
            break
    print("Training Ends!")

    if test_loader != None:
        avg_test_loss, avg_test_f1 = predict_and_evaluate(best_model, test_loader, device, criterion)
        print(f"Test Loss: {avg_test_loss:.4e}, Test F1: {avg_test_f1:.3f}")

    # Plotting the results
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', color='tab:red', linestyle='-')
    ax1.plot(range(1, len(valid_losses) + 1), valid_losses, label='Valid Loss', color='tab:red', linestyle='--')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()  # instantiate a second y-axis
    ax2.set_ylabel('F1 Score', color='tab:blue')
    ax2.plot(range(1, len(train_f1s) + 1), train_f1s, label='Train F1', color='tab:blue', linestyle='-')
    ax2.plot(range(1, len(valid_f1s) + 1), valid_f1s, label='Valid F1', color='tab:blue', linestyle='--')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.legend(loc='upper right')

    fig.tight_layout()  # adjusts subplot parameters to give the plot more space
    plt.savefig(plot_save_path)

    return best_model, best_f1, avg_test_f1


def load_model(model_path, device):
    """
    Loads a saved model from the specified path.
    :param model_path: str, path to the saved model file.
    :param device: str, device to load the model on ('cuda' or 'cpu').
    :return: Loaded model.
    """
    model = CustomModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model


def testing(model, data_path, tokenizer, batch_size, device):
    """
    Main function to perform inference.
    :param model_path: str, path to the saved model.
    :param data_path: str, path to the test data file.
    :param tokenizer: Pre-trained tokenizer to process the data.
    :param batch_size: int, size of the batches for processing.
    :param device: str, device to perform inference on.
    :return: List of predictions.
    """
    data_loader = create_data_loader(data_path, tokenizer, batch_size, is_train=False)
    loss, f1 = predict_and_evaluate(model, data_loader, device, criterion=nn.CrossEntropyLoss(reduction="mean"))

    print(f"Loss: {loss:.4f}, F1: {f1:.4f}")

    return loss, f1


# # explainability.py
# import torch
# from transformers import BertTokenizer, BertModel
# from captum.attr import LayerIntegratedGradients, VisualizationDataRecord
# from captum.attr import visualization as viz

# class ModelExplainability:
#     """
#     Handles the explainability of a model by visualizing attention and using Layer Integrated Gradients.
#     """
#     def __init__(self, model, tokenizer, device):
#         """
#         Initializes the ModelExplainability class with a model, tokenizer, and device.
#         :param model: the model to explain.
#         :param tokenizer: tokenizer used for model input preparation.
#         :param device: the device on which computations will be performed.
#         """
#         self.model = model
#         self.tokenizer = tokenizer
#         self.device = device
#         self.model.eval()  # Ensure the model is in evaluation mode
#         self.model.to(device)

#     def interpret_sentence(self, sentence, label):
#         """
#         Interprets a sentence using Integrated Gradients and visualizes the attributions.
#         :param sentence: the input sentence to interpret.
#         :param label: the true label of the sentence for reference.
#         """
#         # Preprocess the sentence
#         input_ids = torch.tensor([self.tokenizer.encode(sentence, add_special_tokens=True)], device=self.device)
#         input_ids.requires_grad = True

#         # Forward pass
#         preds = self.model(input_ids)[0]
#         pred_prob, pred_label = torch.max(preds.softmax(dim=1), dim=1)

#         # Compute attributions using Layer Integrated Gradients
#         lig = LayerIntegratedGradients(self.model, self.model.bert.embeddings)
#         attributions = lig.attribute(input_ids, target=pred_label.item(), baselines=input_ids * 0)

#         # Visualize the results
#         tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
#         viz.visualize_text([VisualizationDataRecord(
#             attributions[0].cpu().detach().numpy(),
#             pred_prob.item(),
#             pred_label.item(),
#             label,
#             sentence,
#             attributions.sum().item(),
#             tokens,
#             delta=0.1,
#         )])

# def setup_explainability(model_path, device):
#     """
#     Sets up model and tokenizer for explainability.
#     :param model_path: path to the model file.
#     :param device: computation device.
#     :return: ModelExplainability instance.
#     """
#     model = BertModel.from_pretrained(model_path)
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     return ModelExplainability(model, tokenizer, device)


def process(is_train=True):
    # is_train = True  # Whether train or not

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Set the tokenizer not to use multithreading

    if is_train == True:
        model = CustomModel().to(CFG.device)
        train_loader = create_data_loader(CFG.train_data_path, CFG.tokenizer, CFG.batch_size, is_train=True)
        valid_loader = create_data_loader(CFG.valid_data_path, CFG.tokenizer, CFG.batch_size, is_train=False)
        test_loader = create_data_loader(CFG.test_data_path, CFG.tokenizer, CFG.batch_size, is_train=False)
        model, _, _ = training(model, train_loader=train_loader, valid_loader=valid_loader, test_loader=test_loader)
    else:
        model = load_model(CFG.model_save_path, CFG.device)
        testing(model, CFG.test_data_path, CFG.tokenizer, CFG.batch_size, CFG.device)

if __name__ == '__main__':
    is_train = True  # Whether train or not
    process(is_train)
