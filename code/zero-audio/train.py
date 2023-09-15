from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification


# Custom dataset class to load and preprocess the text data
class TextDataset(Dataset):
    def __init__(self, text_file1, text_file2, max_vocab_size):
        self.texts = []
        self.labels = []
        
        with open(text_file1, 'r', encoding='utf-8') as file:
            for line in file:
                self.texts.append('Audio of a ' + line.strip())
                self.labels.append(0)  # Label 0 for category 1
        
        with open(text_file2, 'r', encoding='utf-8') as file:
            for line in file:
                self.texts.append('Audio of a ' + line.strip())
                self.labels.append(1)  # Label 1 for category 2

        # Vectorize the text using CountVectorizer
        # self.vectorizer = CountVectorizer(max_features=max_vocab_size)
        # self.X = self.vectorizer.fit_transform(self.texts)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], torch.tensor(self.labels[idx], dtype=torch.long)


# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


def eval_ds(ds, model, tokenizer, epoch):
    model.eval()
    accuracy = []
    pbar = tqdm(ds)
    for text, labels in pbar:
        encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        output = model(encoded_input['input_ids'].cuda(), encoded_input['attention_mask'].cuda())
        predictions = output.logits
        y_pred = F.softmax(predictions, dim=1)
        auc = roc_auc_score(labels.cpu().numpy(), y_pred[:, 1].cpu().numpy())
        # predicted_labels = torch.argmax(predictions.cpu(), dim=1)
        # accuracy = accuracy_score(labels.numpy(), predicted_labels.numpy())
        accuracy.append(auc)
        pbar.set_description('(Inference | {task}) Epoch {epoch} :: acc {acc:.4f}'.format(task='baseline',
                                                                                          epoch=epoch,
                                                                                          acc=np.mean(accuracy)))
    average_accuracy = np.mean(accuracy)
    return average_accuracy


def train(ds, model, tokenizer, optimizer, epoch):
    model.train()
    loss_list = []
    pbar = tqdm(ds)
    for ix, (text, labels) in enumerate(pbar):
        optimizer.zero_grad()
        encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        output = model(encoded_input['input_ids'].cuda(), encoded_input['attention_mask'].cuda())
        predictions = output.logits
        loss = nn.functional.cross_entropy(predictions, labels.cuda())
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        pbar.set_description(
            '(train | {}) epoch {epoch} ::'
            ' loss {loss:.4f}'.format(
                'baseline',
                epoch=epoch,
                loss=np.mean(loss_list)
            ))
    return np.mean(loss_list)


def main(max_vocab_size = 10000, train_size = 0.7, val_size = 0.15, test_size = 0.15, epochs = 100000):
    dataset = TextDataset('./audibility-dataset/audible.txt', './audibility-dataset/notAudible.txt', max_vocab_size)

    total_samples = len(dataset)
    train_samples = int(train_size * total_samples)
    val_samples = int(val_size * total_samples)
    test_samples = total_samples - train_samples - val_samples

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_samples, val_samples, test_samples])

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased").cuda()

    optimizer = optim.AdamW(model.parameters(),
                            lr=0.0003)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10)
    best = 0
    path_res = 'results.csv'
    f_res = open(path_res, 'w')
    f_res.write('epoch,val,test\n')
    f_res.flush()
    for epoch in range(epochs):
        train(train_loader, model, tokenizer, optimizer, epoch)
        with torch.no_grad():
            val_auc = eval_ds(val_loader, model, tokenizer, epoch)
            scheduler.step(val_auc)
            if val_auc > best:
                best_test = eval_ds(test_loader, model, tokenizer, epoch)
                best = val_auc
                torch.save(model.state_dict(), "best_model.pth")
                f_res.write(str(epoch) + ',' +
                            str(val_auc) + ',' +
                            str(best_test) + '\n')
                f_res.flush()


if __name__ == "__main__":
    main(max_vocab_size = 10000,
    train_size = 0.7,
    val_size = 0.15,
    test_size = 0.15,
    epochs = 100000)
    
    

        
