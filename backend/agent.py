from backend.tools.sentiment import classify
from backend.tools.summarization import summarize
from backend.tools.rewrite import rewrite
from backend.utils.logger import log

def run_agent(text, instruction):
    instruction = instruction.lower().strip()

    log("input", text)

    original_text = text
    working_text = text

    tasks = []

    if instruction == "summary":
        tasks.append("summary")
    elif instruction == "formal":
        tasks.append("rewrite_formal")
    elif instruction == "informal":
        tasks.append("rewrite_informal")
    elif instruction == "tone":
        tasks.append("sentiment")

    log("tasks", str(tasks))

    result = {"input": original_text}

    for task in tasks:

        if task == "summary":
            log("step", "summarization started")
            working_text = summarize(working_text)
            result["summary"] = working_text
            log("step", "summarization finished")

        elif task == "rewrite_formal":
            log("step", "formal rewrite started")
            working_text = rewrite(working_text, "formal")
            result["text"] = working_text
            log("step", "formal rewrite finished")

        elif task == "rewrite_informal":
            log("step", "informal rewrite started")
            working_text = rewrite(working_text, "informal")
            result["text"] = working_text
            log("step", "informal rewrite finished")

        elif task == "sentiment":
            log("step", "sentiment classification started")
            result["sentiment"] = classify(original_text)
            log("step", f"sentiment done: {result['sentiment']}")

    log("output", str(result))

    return result