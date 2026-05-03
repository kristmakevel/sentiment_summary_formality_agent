import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.models.inference import predict_sentiment

#берем из написанной нами модельки функцию
def classify(text):
    return predict_sentiment(text)