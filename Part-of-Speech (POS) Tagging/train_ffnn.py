# train_ffnn.py

import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from sklearn.metrics import classification_report

class FFNNTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=10, hidden_dim=32):
        super(FFNNTagger, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, tagset_size)

    def forward(self, x):
        embeds = self.embedding(x)
        out = self.fc1(embeds)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Training Data
tagged_data = [
    [('THE', 'DET'), ('cat', 'NOUN'), ('sleeps', 'VERB')],
    [('A', 'DET'), ('dog', 'NOUN'), ('barks', 'VERB')],
    [('THE', 'DET'), ('dog', 'NOUN'), ('sleeps', 'VERB')],
    [('MY', 'DET'), ('dog', 'NOUN'), ('runs', 'VERB'), ('fast', 'ADV')],
    [('A', 'DET'), ('cat', 'NOUN'), ('meows', 'VERB'), ('loudly', 'ADV')],
    [('YOUR', 'DET'), ('cat', 'NOUN'), ('runs', 'VERB')],
    [('THE', 'DET'), ('bird', 'NOUN'), ('sings', 'VERB'), ('sweetly', 'ADV')],
    [('A', 'DET'), ('bird', 'NOUN'), ('chirps', 'VERB')]
]

# Build vocab
word2idx = defaultdict(lambda: len(word2idx))
tag2idx = defaultdict(lambda: len(tag2idx))

word2idx['<UNK>'] = 0
tag2idx['<UNK>'] = 0

X = []
y = []

for sentence in tagged_data:
    for word, tag in sentence:
        X.append(word2idx[word.upper()])
        y.append(tag2idx[tag])

X = torch.tensor(X)
y = torch.tensor(y)

vocab_size = len(word2idx)
tag_size = len(tag2idx)

model = FFNNTagger(vocab_size, tag_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(50):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

# Inference helper
idx2tag = {v: k for k, v in tag2idx.items()}

def predict(sentence):
    indices = torch.tensor([word2idx.get(word.upper(), 0) for word in sentence])
    outputs = model(indices)
    pred_indices = torch.argmax(outputs, dim=1)
    return [idx2tag[idx.item()] for idx in pred_indices] # type: ignore

# Test sentences
test_sentences = [
    ['The', 'cat', 'meows'],
    ['My', 'dog', 'barks', 'loudly']
]

for sentence in test_sentences:
    print("Sentence:", sentence)
    print("Predicted Tags:", predict(sentence))

# Evaluate on test data
test_data = [
    [('A', 'DET'), ('bird', 'NOUN'), ('sings', 'VERB')],
    [('MY', 'DET'), ('cat', 'NOUN'), ('runs', 'VERB')],
    [('YOUR', 'DET'), ('dog', 'NOUN'), ('barks', 'VERB'), ('loudly', 'ADV')]
]

def evaluate_ffnn(model, test_data, word2idx, tag2idx):
    model.eval()
    all_true = []
    all_pred = []
    idx2tag = {v: k for k, v in tag2idx.items()}

    with torch.no_grad():
        for sentence in test_data:
            words = [w for w, t in sentence]
            true_tags = [t for w, t in sentence]
            inputs = torch.tensor([word2idx.get(w.upper(), 0) for w in words])
            outputs = model(inputs)
            pred_indices = torch.argmax(outputs, dim=1).tolist()
            pred_tags = [idx2tag[idx] for idx in pred_indices]

            all_true.extend(true_tags)
            all_pred.extend(pred_tags)

    accuracy = sum(t == p for t, p in zip(all_true, all_pred)) / len(all_true)
    print(f"FFNN Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(all_true, all_pred))

evaluate_ffnn(model, test_data, word2idx, tag2idx)
