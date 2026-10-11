"""Generate Bonsai completions for MTP self-distillation and write them as chat-rendered raw text.

Usage: python -m tools.mtp_distill.gen_data <base_url> <out.jsonl> [--prompts P.jsonl] [--threads 3]
Each output line: {"id", "category", "thinking", "text"} where text is the Qwen chat rendering
of the user prompt plus the model's reasoning and answer, ready for ninfer-score-export.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import threading
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor

LANGS = ["Python", "TypeScript", "C++", "Rust", "Go", "Java", "C#", "SQL", "Bash", "JavaScript"]
CODE_TASKS = [
    "a function that parses an ISO 8601 date string and returns the weekday",
    "an LRU cache class with get and put in O(1)",
    "a function that merges overlapping intervals",
    "a thread-safe counter with increment and snapshot",
    "a function that validates an email address without regex libraries",
    "a binary search that returns the first index of a target",
    "a small CLI that counts word frequencies in a file",
    "a function that flattens a nested list or array",
    "a rate limiter using the token bucket algorithm",
    "a function that converts a Roman numeral to an integer",
    "an HTTP retry wrapper with exponential backoff",
    "a function that computes the Levenshtein distance",
    "a trie with insert, search and prefix search",
    "a function that topologically sorts a dependency graph and reports cycles",
    "a debounce utility",
    "a function that reads a CSV file and sums one column, skipping bad rows",
]
EDIT_SNIPPETS = [
    ("Python", "def total(items):\n    t = 0\n    for i in items:\n        t = t + i['price'] * i['qty']\n    return t\n",
     "add type hints, skip items with a missing price, and add a docstring"),
    ("TypeScript", "function getUser(id) {\n  return fetch('/api/users/' + id).then(r => r.json());\n}\n",
     "convert it to async/await, add types for the user, and handle HTTP errors"),
    ("C++", "int sum(std::vector<int> v) {\n  int s = 0;\n  for (int i = 0; i < v.size(); i++) s += v[i];\n  return s;\n}\n",
     "avoid the copy, use a range-based loop or std::accumulate, and return long long"),
    ("Python", "import requests\n\ndef get(url):\n    r = requests.get(url)\n    return r.json()\n",
     "add a timeout, raise for bad status, and retry twice on connection errors"),
    ("Go", "func Max(a []int) int {\n\tm := a[0]\n\tfor _, x := range a {\n\t\tif x > m {\n\t\t\tm = x\n\t\t}\n\t}\n\treturn m\n}\n",
     "handle the empty slice by returning an error, and add a test"),
    ("SQL", "SELECT name, SUM(amount) FROM orders JOIN users ON users.id = orders.user_id GROUP BY name;\n",
     "only include orders from 2025, show users with no orders as 0, and sort by total descending"),
]
ES_TOPICS = [
    "como funciona la memoria cache de una CPU", "que es una API REST y como se diseña una buena",
    "las causas de la independencia de Chile", "como preparar una entrevista de trabajo tecnica",
    "la diferencia entre procesos e hilos", "como funciona el protocolo HTTPS",
    "consejos para aprender un idioma nuevo", "que es la inflacion y como se mide",
    "como organizar un proyecto con Git y ramas", "el ciclo del agua explicado para niños",
    "ventajas y desventajas de trabajar remoto", "como funciona una red neuronal",
    "como escribir un buen correo de reclamo", "que es Docker y para que sirve",
    "historia breve de la computacion", "como hacer un presupuesto mensual personal",
]
EN_TOPICS = [
    "how garbage collection works in modern runtimes", "the history of the printing press",
    "how a database index speeds up queries", "why the sky is blue",
    "how public key cryptography works", "the main ideas of stoic philosophy",
    "how a compiler turns source code into machine code", "the causes of the 2008 financial crisis",
    "how vaccines train the immune system", "what makes a good code review",
    "how GPS determines your position", "the tradeoffs between SQL and NoSQL databases",
    "how speculative decoding speeds up language models", "the life cycle of a star",
]
STYLES_EN = ["Explain {t} in a few paragraphs.", "Write a short essay about {t}.",
             "Give me a step-by-step explanation of {t}.", "Summarize {t} as a bulleted list with short explanations."]
STYLES_ES = ["Explica {t} en pocos parrafos.", "Escribe una explicacion clara sobre {t}.",
             "Dame una lista con los puntos clave sobre {t}, cada uno con una frase."]
MATH = [
    "A shop sells pens at 3 for $2 and notebooks at $4.50 each. Ana buys 12 pens and 5 notebooks and pays with $50. How much change does she get?",
    "How many positive integers less than 1000 are divisible by 3 or 7 but not by 21?",
    "A train travels 300 km at 80 km/h and returns at 120 km/h. What is its average speed for the round trip?",
    "Find all real x such that x^2 - 5x + 6 = 0 and explain the method.",
    "The sum of three consecutive odd numbers is 147. What are they?",
    "In how many ways can 5 people sit around a round table if two of them refuse to sit together?",
    "A rectangle has a perimeter of 46 cm and an area of 120 cm^2. What are its sides?",
    "What is the probability of getting at least one six in four rolls of a fair die?",
    "Un tanque se llena en 6 horas con una llave y en 4 horas con otra. Cuanto tarda con ambas?",
    "Si un producto sube 20% y luego baja 20%, cual es el cambio total en porcentaje?",
    "Prove that the sum of the first n odd numbers is n^2.",
    "A bag has 4 red and 6 blue balls. Two are drawn without replacement. What is the probability both are the same color?",
    "Solve the system: 2x + 3y = 13, 5x - y = 7.",
    "How many trailing zeros does 125! have?",
    "A clock shows 4:20. What is the angle between the hands?",
    "What is the smallest positive integer that leaves remainder 1 when divided by 2, 3, 4, 5 and 6, and is divisible by 7?",
]
JSON_TASKS = [
    "Extract the people mentioned as JSON (name, role, company): 'Maria Lopez, CTO at Nubia, met Tom Reed, a sales lead from Orbit, and Ken Sato, Orbit's CEO.'",
    "Return a JSON array of 5 fictional products with fields id, name, price, tags.",
    "Convert this to JSON with keys date, amount, currency, merchant: 'Paid 45.90 EUR at Cafe Luna on 3 March 2026'.",
    "Classify these tickets as bug, feature or question and return JSON: 1) app crashes on login 2) add dark mode 3) how do I export data?",
    "Write a JSON Schema for a user object with name, email, age (optional) and roles (array of strings).",
    "Give a JSON config for a web server with host, port, tls settings and a list of routes.",
]


def prompts(seed: int) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for lang, task in itertools.product(LANGS[:8], CODE_TASKS[:8]):
        out.append({"category": "code", "prompt": f"Write {task} in {lang}. Include a short explanation."})
    for lang, task in zip(itertools.cycle(LANGS), CODE_TASKS[8:] * 2):
        out.append({"category": "code", "prompt": f"Implement {task} in {lang}, with tests."})
    for lang, code, ask in EDIT_SNIPPETS * 3:
        out.append({"category": "edit",
                    "prompt": f"Here is some {lang} code:\n```\n{code}```\nPlease {ask}. Return the full updated code."})
    for t in ES_TOPICS:
        for style in rng.sample(STYLES_ES, 2):
            out.append({"category": "es", "prompt": style.format(t=t)})
    for t in EN_TOPICS:
        for style in rng.sample(STYLES_EN, 2):
            out.append({"category": "en", "prompt": style.format(t=t)})
    for q in MATH:
        out.append({"category": "math", "prompt": q})
    for q in JSON_TASKS * 2:
        out.append({"category": "json", "prompt": q})
    rng.shuffle(out)
    for i, item in enumerate(out):
        # Thinking on for math and for a share of the rest, as a coding agent would see it.
        item["thinking"] = item["category"] == "math" or rng.random() < 0.3
        item["id"] = f"{item['category']}-{i:04d}"
    return out


def render(prompt: str, reasoning: str, content: str, thinking: bool) -> str:
    think = f"<think>\n{reasoning.strip()}\n</think>\n\n" if thinking else "<think>\n\n</think>\n\n"
    return (f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            f"{think}{content.strip()}<|im_end|>\n")


def complete(base_url: str, item: dict, max_tokens: int) -> dict:
    body = {
        "model": "bonsai",
        "messages": [{"role": "user", "content": item["prompt"]}],
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": item["thinking"]},
        "seed": zlib.crc32(item["id"].encode()) & 0x7FFFFFFF,
    }
    request = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
                                     data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=1800) as response:
        reply = json.load(response)
    message = reply["choices"][0]["message"]
    return {
        "id": item["id"], "category": item["category"], "thinking": item["thinking"],
        "finish": reply["choices"][0].get("finish_reason"),
        "completion_tokens": reply.get("usage", {}).get("completion_tokens"),
        "text": render(item["prompt"], message.get("reasoning_content") or "", message.get("content") or "",
                       item["thinking"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_url")
    parser.add_argument("out")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=1536)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--prompts", help="JSONL of {id, category, prompt, thinking}; default: the pilot set")
    parser.add_argument("--skip", type=int, default=0, help="resume: skip the first N prompts")
    args = parser.parse_args()
    if args.prompts:
        with open(args.prompts, encoding="utf-8") as f:
            items = [json.loads(line) for line in f if line.strip()]
    else:
        items = prompts(args.seed)
    items = items[args.skip :]
    if args.limit:
        items = items[: args.limit]
    lock = threading.Lock()
    done = 0
    total = 0
    with open(args.out, "a" if args.skip else "w", encoding="utf-8") as out, ThreadPoolExecutor(args.threads) as pool:
        for result in pool.map(lambda it: complete(args.base_url, it, args.max_tokens), items):
            with lock:
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                out.flush()
                done += 1
                total += result["completion_tokens"] or 0
                if done % 10 == 0:
                    print(f"{done}/{len(items)} done, {total} completion tokens", flush=True)
    print(f"finished: {done} completions, {total} completion tokens", flush=True)


if __name__ == "__main__":
    main()
