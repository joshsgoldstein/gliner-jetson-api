#!/usr/bin/env python3
"""
evalcli - a mini eval CLI for the GLiNER2 extraction API.

Companion to the LLM evalcli in harness-testing, same philosophy: cases live in
evals/cases/*.jsonl as data and grow over time - append lines (or use `add`), no
code changes. Every `run` appends one JSON line to evals/results.jsonl so
correctness and latency are tracked across models over time.

Because MODEL_ID is fixed at container start, comparing checkpoints means
running the same suite against several base URLs (one container per model) and
labelling each run.

Commands:
    python3 evalcli.py run [--base-url URL] [--suite GLOB] [--tags a,b]
                           [--id CASE] [--label "..."] [--repeat N] [-v]
    python3 evalcli.py list [--suite GLOB] [--tags a,b]
    python3 evalcli.py summary [--full]
    python3 evalcli.py add --id X --endpoint /extract_entities
                           --payload '{...}' --check '{...}' [--tags a,b]

Case shape (one JSON object per line):
    {"id":"clinical-basic","category":"entities","tags":["clinical","en"],
     "endpoint":"/extract_entities",
     "payload":{"text":"...","labels":["medication"]},
     "check":{"type":"entities_exact","value":{"medication":["ibuprofen"]}}}

Check types:
    {"type":"entities_exact","value":{"label":["span",...]}}   exact set per label
    {"type":"entities_subset","value":{...}}                   expected must be present
    {"type":"entities_f1","value":{...},"min":0.85}            micro-F1 over all labels
    {"type":"key_equals","key":"sentiment","value":"negative"}
    {"type":"record_count","key":"product","count":1}          catches duplicate records
    {"type":"field_equals","path":"product.0.price","value":"$399"}
    {"type":"relations_contain","value":{"works_for":[["a","b"]]}}
    {"type":"http_status","value":413}
    {"type":"all_of","value":[<check>,...]}  /  {"type":"any_of","value":[<check>,...]}

`run` exits non-zero if any case fails, so it can gate a rebuild or run from cron.
"""
import argparse
import glob
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SUITE = os.path.join(HERE, "cases", "*.jsonl")
RESULTS = os.path.join(HERE, "results.jsonl")
DEFAULT_BASE_URL = os.environ.get("GLINER_BASE_URL", "http://192.168.1.177:8013")
# A crashed worker is back in ~20s (model reload); wait past that before retrying.
TRANSPORT_RETRY_WAIT = float(os.environ.get("GLINER_TRANSPORT_RETRY_WAIT", "30"))


# --------------------------------------------------------------------------- io

def load_cases(suite_glob, tags=None, only_id=None):
    cases = []
    for path in sorted(glob.glob(suite_glob)):
        with open(path, "r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    case = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"{path}:{lineno}: invalid JSON: {exc}")
                case.setdefault("suite", os.path.basename(path))
                cases.append(case)

    if only_id:
        cases = [c for c in cases if c.get("id") == only_id]
    if tags:
        want = set(tags)
        cases = [c for c in cases if want & set(c.get("tags", []))]
    return cases


def post(base_url, endpoint, payload, timeout=120):
    """Return (status_code, parsed_body_or_none, elapsed_ms)."""
    req = urllib.request.Request(
        base_url.rstrip("/") + endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            elapsed = (time.monotonic() - start) * 1000
            return resp.status, _maybe_json(body), elapsed
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        elapsed = (time.monotonic() - start) * 1000
        return exc.code, _maybe_json(body), elapsed
    except Exception as exc:
        # Deliberately broad. The service can die mid-run -- the process crashes
        # and Docker restarts it ~20s later -- which surfaces as
        # RemoteDisconnected, ConnectionResetError, ConnectionRefusedError or a
        # bare URLError depending on exactly when the socket dropped. Catching
        # only URLError/TimeoutError meant a crash killed the CLI with a
        # traceback instead of being recorded, which is precisely the event
        # worth recording. Status 0 marks a transport failure.
        elapsed = (time.monotonic() - start) * 1000
        return 0, {"_transport_error": f"{type(exc).__name__}: {exc}"}, elapsed


def _maybe_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"_raw": text}


# ----------------------------------------------------------------------- checks

def _entities(resp):
    """Entity payloads come back under 'entities'; be tolerant of a bare dict."""
    if isinstance(resp, dict):
        if isinstance(resp.get("entities"), dict):
            return resp["entities"]
        return resp
    return {}


def _norm(values):
    return {str(v).strip() for v in (values or [])}


def _dig(obj, path):
    """Resolve a dotted path; integer segments index into lists."""
    cur = obj
    for seg in path.split("."):
        if isinstance(cur, list):
            try:
                cur = cur[int(seg)]
            except (ValueError, IndexError):
                return None
        elif isinstance(cur, dict):
            if seg not in cur:
                return None
            cur = cur[seg]
        else:
            return None
    return cur


def run_check(check, status, resp):
    """Return (passed: bool, detail: str)."""
    kind = check.get("type")

    if kind == "http_status":
        want = check["value"]
        return status == want, f"status {status} (want {want})"

    if kind in ("all_of", "any_of"):
        results = [run_check(sub, status, resp) for sub in check["value"]]
        passed = all(r[0] for r in results) if kind == "all_of" else any(r[0] for r in results)
        detail = "; ".join(f"{'ok' if r[0] else 'FAIL'}:{r[1]}" for r in results)
        return passed, detail

    # Everything below expects a 200 with a JSON body.
    if status != 200:
        return False, f"expected 200, got {status}: {json.dumps(resp)[:160]}"

    if kind == "entities_exact":
        actual, expected = _entities(resp), check["value"]
        problems = []
        for label, want in expected.items():
            got = _norm(actual.get(label))
            if got != _norm(want):
                problems.append(f"{label}: got {sorted(got)} want {sorted(_norm(want))}")
        extra = set(actual) - set(expected)
        if extra:
            problems.append(f"unexpected labels {sorted(extra)}")
        return not problems, "; ".join(problems) or "exact match"

    if kind == "entities_subset":
        actual, expected = _entities(resp), check["value"]
        missing = []
        for label, want in expected.items():
            got = _norm(actual.get(label))
            absent = _norm(want) - got
            if absent:
                missing.append(f"{label} missing {sorted(absent)}")
        return not missing, "; ".join(missing) or "all expected spans present"

    if kind == "entities_f1":
        actual, expected = _entities(resp), check["value"]
        tp = fp = fn = 0
        for label in set(expected) | set(actual):
            want, got = _norm(expected.get(label)), _norm(actual.get(label))
            tp += len(want & got)
            fp += len(got - want)
            fn += len(want - got)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        floor = check.get("min", 0.9)
        return f1 >= floor, f"f1={f1:.3f} p={prec:.3f} r={rec:.3f} (min {floor})"

    if kind == "key_equals":
        got = resp.get(check["key"]) if isinstance(resp, dict) else None
        return got == check["value"], f"{check['key']}={got!r} (want {check['value']!r})"

    if kind == "record_count":
        records = resp.get(check["key"]) if isinstance(resp, dict) else None
        n = len(records) if isinstance(records, list) else -1
        return n == check["count"], f"{check['key']} has {n} record(s) (want {check['count']})"

    if kind == "field_equals":
        got = _dig(resp, check["path"])
        return got == check["value"], f"{check['path']}={got!r} (want {check['value']!r})"

    if kind == "relations_contain":
        rels = resp.get("relation_extraction", resp) if isinstance(resp, dict) else {}
        missing = []
        for name, pairs in check["value"].items():
            got = {tuple(str(x) for x in p) for p in (rels.get(name) or [])}
            for pair in pairs:
                if tuple(str(x) for x in pair) not in got:
                    missing.append(f"{name} missing {pair}")
        return not missing, "; ".join(missing) or "all expected relations present"

    return False, f"unknown check type {kind!r}"


# ---------------------------------------------------------------------- commands

def cmd_run(args):
    cases = load_cases(args.suite, args.tags, args.id)
    if not cases:
        raise SystemExit("no cases matched")

    health_status, health, _ = _get(args.base_url + "/health")
    model_id = health.get("model_id") if isinstance(health, dict) else None
    arch = health.get("architecture") if isinstance(health, dict) else None
    if health_status != 200:
        raise SystemExit(f"{args.base_url}/health returned {health_status}; is the service up?")

    label = args.label or model_id or args.base_url
    print(f"suite   : {args.suite}")
    print(f"base_url: {args.base_url}")
    print(f"model   : {model_id} (architecture={arch})")
    print(f"cases   : {len(cases)}  repeat={args.repeat}\n")

    rows, failures = [], 0
    transport_failures = []
    for case in cases:
        latencies, passed, detail = [], False, ""
        for _ in range(args.repeat):
            status, resp, ms = post(args.base_url, case["endpoint"], case["payload"])
            if status == 0:
                # Transport failure: the service may be restarting. Record it,
                # wait out a model reload, and give the case one more chance.
                transport_failures.append({"id": case["id"], "error": resp.get("_transport_error")})
                log_line = resp.get("_transport_error")
                print(f"  [WARN] {case['id']:<26} transport failure: {log_line}; retrying in {TRANSPORT_RETRY_WAIT}s")
                time.sleep(TRANSPORT_RETRY_WAIT)
                status, resp, ms = post(args.base_url, case["endpoint"], case["payload"])
            latencies.append(ms)
        passed, detail = run_check(case["check"], status, resp)
        if not passed:
            failures += 1
        med = statistics.median(latencies)
        rows.append({"id": case["id"], "category": case.get("category", ""),
                     "passed": passed, "detail": detail, "median_ms": round(med, 1)})
        mark = "PASS" if passed else "FAIL"
        print(f"  [{mark}] {case['id']:<26} {med:6.1f}ms  {detail}")
        if args.verbose and not passed:
            print(f"         response: {json.dumps(resp, ensure_ascii=False)[:400]}")

    total = len(rows)
    passed_n = total - failures
    med_all = statistics.median([r["median_ms"] for r in rows])
    print(f"\n{passed_n}/{total} passed  |  median latency {med_all:.1f}ms")
    if transport_failures:
        print(f"\n  !! {len(transport_failures)} transport failure(s) -- the service dropped "
              f"connections mid-run. This is the signature of the process crashing and "
              f"being restarted, not a slow request:")
        for tf in transport_failures:
            print(f"     {tf['id']}: {tf['error']}")

    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label": label, "base_url": args.base_url, "model_id": model_id,
        "architecture": arch, "suite": args.suite,
        "passed": passed_n, "total": total, "median_ms": round(med_all, 1),
        "failures": [r["id"] for r in rows if not r["passed"]],
        "transport_failures": transport_failures,
        "cases": rows,
    }
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    print(f"appended to {RESULTS}")
    return 1 if failures else 0


def _get(url, timeout=15):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, _maybe_json(resp.read().decode("utf-8")), None
    except urllib.error.HTTPError as exc:
        return exc.code, _maybe_json(exc.read().decode("utf-8", errors="replace")), None
    except (urllib.error.URLError, TimeoutError) as exc:
        return 0, {"_transport_error": str(exc)}, None


def cmd_list(args):
    cases = load_cases(args.suite, args.tags, None)
    for case in cases:
        tags = ",".join(case.get("tags", []))
        print(f"{case['id']:<28} {case['endpoint']:<26} {case['check']['type']:<18} [{tags}]")
    print(f"\n{len(cases)} case(s)")
    return 0


def cmd_summary(args):
    if not os.path.exists(RESULTS):
        print("no results yet")
        return 0
    with open(RESULTS, "r", encoding="utf-8") as fh:
        runs = [json.loads(l) for l in fh if l.strip()]
    print(f"{'when':<20} {'label':<28} {'score':>8} {'median':>9}   failures")
    for run in runs:
        score = f"{run['passed']}/{run['total']}"
        fails = ",".join(run.get("failures", [])) or "-"
        tf = run.get("transport_failures") or []
        if tf:
            fails = f"{fails}  [+{len(tf)} transport]"
        print(f"{run['ts'][:19]:<20} {str(run.get('label'))[:27]:<28} {score:>8} "
              f"{run['median_ms']:>8.1f}ms   {fails}")
    if args.full:
        print()
        for run in runs[-1:]:
            for case in run["cases"]:
                print(f"  {'PASS' if case['passed'] else 'FAIL'} {case['id']:<26} {case['detail']}")
    return 0


def cmd_add(args):
    case = {
        "id": args.id,
        "category": args.category,
        "tags": args.tags or [],
        "endpoint": args.endpoint,
        "payload": json.loads(args.payload),
        "check": json.loads(args.check),
    }
    target = args.out or os.path.join(HERE, "cases", "core.jsonl")
    with open(target, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"appended {args.id} to {target}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Eval CLI for the GLiNER2 extraction API")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--suite", default=DEFAULT_SUITE, help="glob of case files")
        p.add_argument("--tags", type=lambda s: s.split(","), help="comma-separated tag filter")

    r = sub.add_parser("run", help="run the suite against a live service")
    common(r)
    r.add_argument("--base-url", default=DEFAULT_BASE_URL)
    r.add_argument("--id", help="run a single case by id")
    r.add_argument("--label", help="label this run in results.jsonl")
    r.add_argument("--repeat", type=int, default=3, help="calls per case; median is recorded")
    r.add_argument("-v", "--verbose", action="store_true")
    r.set_defaults(func=cmd_run)

    l = sub.add_parser("list", help="list loaded cases")
    common(l)
    l.set_defaults(func=cmd_list)

    s = sub.add_parser("summary", help="history of past runs")
    s.add_argument("--full", action="store_true")
    s.set_defaults(func=cmd_summary)

    a = sub.add_parser("add", help="append a new case")
    a.add_argument("--id", required=True)
    a.add_argument("--endpoint", required=True)
    a.add_argument("--payload", required=True)
    a.add_argument("--check", required=True)
    a.add_argument("--category", default="")
    a.add_argument("--tags", type=lambda s: s.split(","))
    a.add_argument("--out")
    a.set_defaults(func=cmd_add)

    args = ap.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
