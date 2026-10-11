"""Build the full MTP training prompt set: Magicoder code problems, oasst2 English and Spanish
user prompts, and the pilot's edit/JSON/math templates. Excludes every pilot prompt text.

  python -m tools.mtp_distill.build_prompts OUT.jsonl --oasst OASST_PROMPTS.jsonl.gz \
      --magicoder MAGICODER.jsonl [--code 1500 --en 1000 --es 800 --seed 7]

Sources: OpenAssistant/oasst2 `2023-11-05_oasst2_prompts.messages.jsonl.gz` (Apache-2.0) and
ise-uiuc/Magicoder-OSS-Instruct-75K `data-oss_instruct-decontaminated.jsonl` (MIT).
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import random

from tools.mtp_distill import gen_data



def oasst(path: str, lang: str, count: int, rng: random.Random) -> list[dict]:
    rows, seen = [], set()
    for line in gzip.open(path, "rt", encoding="utf-8"):
        r = json.loads(line)
        text = (r.get("text") or "").strip()
        if (r.get("lang") != lang or r.get("role") != "prompter" or r.get("deleted")
                or r.get("review_result") is False or not 20 <= len(text) <= 2000):
            continue
        key = text.lower()[:200]
        if key in seen:
            continue
        seen.add(key)
        rows.append(text)
    rng.shuffle(rows)
    return [{"category": lang, "prompt": t} for t in rows[:count]]


def magicoder(path: str, count: int, rng: random.Random) -> list[dict]:
    by_lang = collections.defaultdict(list)
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        problem = (r.get("problem") or "").strip()
        if 50 <= len(problem) <= 4000:
            by_lang[r["lang"]].append(problem)
    langs = sorted(by_lang)
    # Python keeps 40 %; the other languages share the rest evenly.
    quota = {lang: int(count * 0.4) if lang == "python" else int(count * 0.6 / (len(langs) - 1)) for lang in langs}
    out = []
    for lang in langs:
        rng.shuffle(by_lang[lang])
        out += [{"category": "code", "prompt": p} for p in by_lang[lang][: quota[lang]]]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--oasst", required=True)
    parser.add_argument("--magicoder", required=True)
    parser.add_argument("--code", type=int, default=1500)
    parser.add_argument("--en", type=int, default=1000)
    parser.add_argument("--es", type=int, default=800)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    pilot = {p["prompt"] for p in gen_data.prompts(1)}
    items = (magicoder(args.magicoder, args.code, rng) + oasst(args.oasst, "en", args.en, rng)
             + oasst(args.oasst, "es", args.es, rng))
    # Templates the datasets lack: code edits, JSON and math, with new seeds.
    items += [p for p in gen_data.prompts(args.seed) if p["category"] in ("edit", "json", "math")]
    items = [p for p in items if p["prompt"] not in pilot or p["category"] in ("edit", "json", "math")]
    rng.shuffle(items)
    with open(args.out, "w", encoding="utf-8") as out:
        for i, item in enumerate(items):
            item["thinking"] = item["category"] == "math" or rng.random() < 0.3
            item["id"] = f"{item['category']}-{i:05d}"
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
    counts = collections.Counter(p["category"] for p in items)
    print(f"{len(items)} prompts: {dict(counts)}, thinking {sum(p['thinking'] for p in items)}")


if __name__ == "__main__":
    main()
