"""Exercise the live local service; saves actual results to logs/verification.json."""
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parent
config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
BASE = f'http://127.0.0.1:{config["port"]}'
results = []

def check(path, payload=None, status=200, predicate=lambda body: True):
    start = time.perf_counter()
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
    request = Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=120) as response:
            actual = response.status
            body = json.load(response)
    except HTTPError as error:
        actual = error.code
        body = json.load(error)
    passed = actual == status and predicate(body)
    result = dict(path=path, expected_status=status, actual_status=actual,
                  passed=passed, seconds=round(time.perf_counter()-start, 3), response=body)
    results.append(result)
    print(json.dumps(result, ensure_ascii=False), flush=True)

check("/", predicate=lambda b: b.get("status") == "Running")
check("/generate", {"prompt": "请用中文一句话介绍你能做什么。", "max_tokens": 512}, predicate=lambda b: bool(b.get("response")))
check("/generate", {"prompt": "小明有3个苹果，又买了5个，送出2个，还剩几个？只回答数字。", "max_tokens": 512}, predicate=lambda b: "6" in b.get("response", ""))
check("/rdf", {"prompt": "Use prefix ex: <https://example.com/>. Alice knows Bob. Return one Turtle triple.", "max_tokens": 512}, predicate=lambda b: "ex:" in b.get("response", ""))
check("/sparql", {"prompt": "Return a SPARQL query selecting 10 triples from the graph using variables s, p, o.", "max_tokens": 512}, predicate=lambda b: "SELECT" in b.get("response", "").upper())
check("/similarity", {"text1": "The wall is concrete.", "text2": "The wall is concrete."}, predicate=lambda b: b.get("score", 0) > 0.99)
check("/dictionary_similarity", {"sentence": "The wall is made of concrete.", "dictionary": {"Concrete": "A wall made of concrete.", "Fruit": "A sweet apple grows on a tree."}}, predicate=lambda b: b["scores"]["Concrete"] > b["scores"]["Fruit"])
check("/generate", {"prompt": "test", "model": 999}, status=400)
check("/rdf", {"prompt": "test", "model": 999}, status=400)
check("/generate", {"prompt": "test", "prompt_file": "../../config.json"}, status=400)
check("/generate", {"prompt": ""}, status=422)
check("/generate", {"prompt": " hello" * 5000}, status=400)
(ROOT / "logs" / "verification.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
raise SystemExit(0 if all(r["passed"] for r in results) else 1)

