import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np

# Sample Dataset
data = [
    ("I loved this movie, it was fantastic!", 1),
    ("What a terrible film, I hated it.", 0),
    ("It was an amazing experience, really enjoyed it.", 1),
    ("Worst movie ever, do not waste your time.", 0),
    ("Great acting and wonderful plot!", 1),
    ("The film was dull and boring.", 0),
    ("Absolutely brilliant, would watch again.", 1),
    ("Not good, the story was weak.", 0)
]

texts, labels = zip(*data)
labels = np.array(labels)

# Split data into train/test
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.25, random_state=42
)

# Tokenizer and vocab
def tokenize(text):
    return text.lower().split()

all_tokens = [token for text in train_texts for token in tokenize(text)]
vocab = {word: i+1 for i, (word, _) in enumerate(Counter(all_tokens).most_common())}  # 0 is padding

def encode(text):
    return [vocab.get(token, 0) for token in tokenize(text)]

train_encoded = [torch.tensor(encode(text)) for text in train_texts]
test_encoded = [torch.tensor(encode(text)) for text in test_texts]

class SentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

def collate_fn(batch):
    texts, labels = zip(*batch)
    texts_padded = pad_sequence(texts, batch_first=True) # type: ignore
    labels = torch.tensor(labels).long()  # <-- fix here
    return texts_padded, labels

train_dataset = SentimentDataset(train_encoded, train_labels)
test_dataset = SentimentDataset(test_encoded, test_labels)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

class LSTMSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim=50, hidden_dim=64, output_dim=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return out

vocab_size = len(vocab)
model = LSTMSentiment(vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.01)

# Training Loop
epochs = 20
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for texts_batch, labels_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(texts_batch)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(train_loader):.4f}")

# Evaluation
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for texts_batch, labels_batch in test_loader:
        outputs = model(texts_batch)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels_batch.tolist())

print("LSTM Accuracy:", accuracy_score(all_labels, all_preds))
print("LSTM Classification Report:")
print(classification_report(all_labels, all_preds, zero_division=0))
