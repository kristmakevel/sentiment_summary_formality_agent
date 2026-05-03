import torch
import numpy as np
import pickle
import string
import os
from nltk.tokenize import word_tokenize
from backend.models.sentiment_model import SentimentModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
artifacts_dir = os.path.join(base_dir, "artifacts")

#загружаем артефакты из нашего обучения
word2idx = pickle.load(open(os.path.join(artifacts_dir, "word2idx.pkl"), "rb"))
embedding_matrix = np.load(os.path.join(artifacts_dir, "embedding_matrix.npy"))
model_path = os.path.join(artifacts_dir, "sentiment_model.pth")

max_len = 150
embed_dim = 200
hidden_dim = 256
num_classes = 3

model = SentimentModel(
    embedding_matrix=embedding_matrix,
    embed_dim=embed_dim,
    hidden_dim=hidden_dim,
    num_classes=num_classes
)

model.load_state_dict(
    torch.load(model_path, map_location=device)
)

model.to(device)
model.eval()

punctuation = set(string.punctuation)

def preprocess(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in punctuation])
    return word_tokenize(text)

def vectorize(tokens):
    indices = [word2idx.get(w, word2idx["<UNK>"]) for w in tokens]

    if len(indices) < max_len:
        indices += [word2idx["<PAD>"]] * (max_len - len(indices))
    else:
        indices = indices[:max_len]

    return indices

def predict_sentiment(text):
    tokens = preprocess(text)
    indices = vectorize(tokens)

    x = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
    lengths = torch.tensor([len(tokens)]).to(device)

    with torch.no_grad():
        outputs = model(x, lengths)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, pred = torch.max(probabilities, dim=1)

        pred_idx = pred.item()
        confidence_score = confidence.item()

    label_map = {
        0: "negative",
        1: "neutral",
        2: "positive"
    }

    #уверенность возвращаем как доп данные
    return {
        "label": label_map[pred_idx],
        "confidence": round(confidence_score, 2)
    }