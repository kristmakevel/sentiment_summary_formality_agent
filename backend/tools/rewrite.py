from transformers import pipeline

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

rewriter = pipeline(
    "text-generation",
    model=model_name,
)

#пишем промт для уже обученной модели
def rewrite(text, style="formal"):
    messages = [
        {"role": "system", "content": "You are a professional text editor. Your task is to rewrite the text in the specified style while preserving the original meaning."},
        {"role": "user", "content": f"Rewrite the following text. Style: {style}. Make it sound professional and natural.\n\nText:\n{text}"}
    ]

    output = rewriter(
        messages,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7
    )

    return output[0]["generated_text"][-1]["content"]