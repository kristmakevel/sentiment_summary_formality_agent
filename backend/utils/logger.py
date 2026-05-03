import time

#понимать, что сейчас происходит
def log(step, text):
    print(f"[{time.time()}] {step}: {text[:80]}")