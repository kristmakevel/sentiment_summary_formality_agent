from backend.tools.sentiment import classify
from backend.tools.summarization import summarize
from backend.tools.rewrite import rewrite
from backend.utils.logger import log
from transformers import pipeline
import json
import re

_planner = None

TOOLS_DESCRIPTION = """
You have access to the following tools:
- sentiment(text) — classifies the tone of the text: positive, negative, or neutral
- summarize(text) — creates a short summary of the text
- rewrite_formal(text) — rewrites the text in a formal, professional style
- rewrite_informal(text) — rewrites the text in a casual, informal style
"""

SYSTEM_PROMPT = f"""You are an intelligent text processing agent.
{TOOLS_DESCRIPTION}

Your job is to analyze the user's instruction and decide which tools to call and in what order.

Rules:
- You can call multiple tools if needed (e.g. summarize first, then analyze sentiment of the summary)
- Tools execute sequentially; each tool receives the output of the previous one as input, EXCEPT sentiment — it always runs on the ORIGINAL text
- Return ONLY a JSON array of tool names, nothing else

Available tool names: ["sentiment", "summarize", "rewrite_formal", "rewrite_informal"]

Examples:
Instruction: "summarize this and tell me the tone" → ["summarize", "sentiment"]
Instruction: "make it formal" → ["rewrite_formal"]
Instruction: "what is the sentiment?" → ["sentiment"]
Instruction: "give me a short formal version" → ["summarize", "rewrite_formal"]
Instruction: "analyze tone and rewrite casually" → ["sentiment", "rewrite_informal"]
"""


def get_planner():
    global _planner
    if _planner is None: #если модель еще не загружена
        _planner = pipeline(
            "text-generation",
            model="Qwen/Qwen2.5-1.5B-Instruct",
        )
    return _planner


def plan_tasks(instruction: str) -> list[str]:
    planner = get_planner()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Instruction: {instruction}"}
    ]

    output = planner(
        messages,
        max_new_tokens=64,
        do_sample=False, #без случайности
    )

    raw = output[0]["generated_text"][-1]["content"].strip() #ответ модели
    log("agent_plan_raw", raw)

    cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()

    try:
        tasks = json.loads(cleaned)
        if not isinstance(tasks, list):
            raise ValueError("Expected a list")
    except Exception as e:
        log("agent_plan_parse_error", str(e))
        found = re.findall(r'"(sentiment|summarize|rewrite_formal|rewrite_informal)"', cleaned)
        tasks = found if found else ["sentiment"] #в случае ошибки ищем вручную все слова

    valid = {"sentiment", "summarize", "rewrite_formal", "rewrite_informal"}
    tasks = [t for t in tasks if t in valid]

    if not tasks:
        tasks = ["sentiment"]

    log("agent_plan_final", str(tasks))
    return tasks

def execute_tasks(original_text: str, tasks: list[str]) -> dict:
    result = {"input": original_text}
    working_text = original_text

    for task in tasks:
        log("step_start", task)

        if task == "sentiment":
            sentiment = classify(original_text)
            result["sentiment"] = sentiment
            log("step_done", f"sentiment={sentiment}")

        elif task == "summarize":
            working_text = summarize(working_text)
            result["summary"] = working_text
            log("step_done", f"summary len={len(working_text)}")

        elif task == "rewrite_formal":
            working_text = rewrite(working_text, "formal")
            result["rewritten"] = working_text
            log("step_done", "rewrite_formal")

        elif task == "rewrite_informal":
            working_text = rewrite(working_text, "informal")
            result["rewritten"] = working_text
            log("step_done", "rewrite_informal")

    result["tasks_executed"] = tasks
    return result


def run_agent(text: str, instruction: str) -> dict:
    log("input_text", text[:200])
    log("input_instruction", instruction)

    tasks = plan_tasks(instruction)

    result = execute_tasks(text, tasks)

    log("output", str(result))
    return result