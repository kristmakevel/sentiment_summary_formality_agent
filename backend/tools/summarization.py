from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_name = "facebook/bart-large-cnn"
device = "cuda" if torch.cuda.is_available() else "cpu"

_tokenizer = None
_model = None


def get_models():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    return _tokenizer, _model


def summarize(text):
    tokenizer, model = get_models()

    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True).to(device)

    input_length = inputs["input_ids"].shape[1]
    dynamic_max = max(15, int(input_length * 0.6))
    dynamic_min = min(10, int(input_length * 0.2))

    with torch.inference_mode():
        summary_ids = model.generate(
            inputs["input_ids"],
            max_new_tokens=dynamic_max,
            min_length=dynamic_min,
            do_sample=False,
            num_beams=2,
            length_penalty=2.0,
            early_stopping=True
        )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)