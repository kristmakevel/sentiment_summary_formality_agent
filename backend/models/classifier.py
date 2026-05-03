import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence
import pandas as pd
from tqdm.auto import tqdm
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import nltk
from collections import Counter
import string
import os
import random
from backend.models.sentiment_model import SentimentModel

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

#задаем директории к тому, что нам потом понадобится
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "../data/sentiment_dataset.csv")
artifacts_dir = os.path.join(base_dir, "../artifacts")
glove_path = os.path.join(base_dir, "glove.6B.200d.txt")

os.makedirs(artifacts_dir, exist_ok=True)

#загрузка необходимых ресурсов nltk для токенизации (разделения на слова)
nltk.download('punkt')
nltk.download('punkt_tab')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#датасет. столбцы: id,text,label,sentiment
dataset = pd.read_csv(data_path)

#разделяем на train и остальное
train_data, temp_data = train_test_split(
    dataset,
    test_size=0.2,
    random_state=seed,
    stratify=dataset['label']
)

#разделяем остаток на val и test
val_data, test_data = train_test_split(
    temp_data,
    test_size=0.5,
    random_state=seed,
    stratify=temp_data['label']
)

words = Counter()
punctuation = set(string.punctuation) #все знаки пунктуации

#удаляем из текста пунктуацию и токенизируем все
for text in train_data['text']:
    #пустые строчки пропускаем
    if not isinstance(text, str):
        continue
    text = text.lower()
    text = ''.join([ch for ch in text if ch not in punctuation])
    tokens = word_tokenize(text)
    words.update(tokens)

#немного сокращаем словарь
max_words = 15000
top_words = [word for word, _ in words.most_common(max_words)]

#словарь слово - индекс
word2idx = {'<PAD>': 0, '<UNK>': 1}
for idx, word in enumerate(top_words, start=2):
    word2idx[word] = idx

vocab_size = len(word2idx)
embed_dim = 200 #ориентируемся на glove который мы берем

#задаем матрицу эмбеддингов с рандомными значениями
embedding_matrix = np.random.normal(scale=0.6, size=(vocab_size, embed_dim))

found = 0

#для слов, которые есть в glove, записываем значения эмбеддингов оттуда
with open(glove_path, encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype="float32")

        if word in word2idx:
            embedding_matrix[word2idx[word]] = vector
            found += 1

#проверяем, нормально ли нам вообще использовать glove или все плохо
print(f"GloVe coverage: {found}/{vocab_size} ({found/vocab_size:.2%})")

max_len = 150

#переводим токенизированный текст в индексы
#если длиннее нужного, обрезаем кусок
#если короче, дополняем заглушками
def vectorize_and_pad(text, word_dict, max_len, punctuation):
    if not isinstance(text, str):
        return [word_dict['<PAD>']] * max_len

    text = text.lower()
    text = ''.join([char for char in text if char not in punctuation])
    tokens = word_tokenize(text)

    indices = [word_dict.get(word, word_dict['<UNK>']) for word in tokens]

    if len(indices) >= max_len:
        indices = indices[:max_len]
    else:
        indices += [word_dict['<PAD>']] * (max_len - len(indices))

    return indices

#векторизируем все наши данные
train_data['vectorized'] = train_data['text'].apply(
    lambda x: vectorize_and_pad(x, word2idx, max_len, punctuation)
)
val_data['vectorized'] = val_data['text'].apply(
    lambda x: vectorize_and_pad(x, word2idx, max_len, punctuation)
)
test_data['vectorized'] = test_data['text'].apply(
    lambda x: vectorize_and_pad(x, word2idx, max_len, punctuation)
)

#класс датасета
class SentimentDataset(Dataset):
    def __init__(self, dataframe):
        self.texts = dataframe['vectorized'].tolist()
        self.labels = dataframe['label'].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = self.texts[idx]
        length = (np.array(x) != 0).sum() #реальная длина текста
        length = max(length, 1)

        #возвращает векторизированный текст, метку класса и реальную длину
        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
            torch.tensor(length, dtype=torch.long)
        )

#даталоадеры для всех выборок
#перемешиваем данные только для train
train_loader = DataLoader(SentimentDataset(train_data), batch_size=64, shuffle=True)
val_loader = DataLoader(SentimentDataset(val_data), batch_size=64, shuffle=False)
test_loader = DataLoader(SentimentDataset(test_data), batch_size=64, shuffle=False)

#используем модельку из другого файла
model = SentimentModel(embedding_matrix, embed_dim, 256, 3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#само обучение модели
#длину передаем, чтобы модель не смотрела на нули в конце
for epoch in range(5):
    model.train()

    total_loss, correct, total = 0, 0, 0

    for x, y, lengths in tqdm(train_loader):
        x, y, lengths = x.to(device), y.to(device), lengths.to(device)

        optimizer.zero_grad()
        outputs = model(x, lengths)
        loss = criterion(outputs, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (outputs.argmax(1) == y).sum().item()
        total += y.size(0)

    print(f"Epoch {epoch+1}: loss={total_loss/total:.4f}, acc={correct/total:.4f}")

model.eval()
correct, total = 0, 0

#проверяем на тесте
with torch.no_grad():
    for x, y, lengths in test_loader:
        x, y, lengths = x.to(device), y.to(device), lengths.to(device)

        preds = model(x, lengths).argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)

print("TEST ACC:", correct / total)

#сохраняем все наши данные, чтобы не обучать модель заново каждый раз
pickle.dump(word2idx, open(os.path.join(artifacts_dir, "word2idx.pkl"), "wb"))
np.save(os.path.join(artifacts_dir, "embedding_matrix.npy"), embedding_matrix)
torch.save(model.state_dict(), os.path.join(artifacts_dir, "sentiment_model.pth"))

print("ALL ARTIFACTS SAVED SUCCESSFULLY")