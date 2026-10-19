"""Regression tests for the caption pipeline: format unification, --captions-file
injection, and prompt wiring.

Runs fully offline -- no API key, no GPU, no real images, no network. Captioner
traffic is intercepted at `captioners._post`, so the tests assert on the actual
request payload rather than on internal state.

    python3 tests/test_caption_pipeline.py

`test_capture_patch.py` covers the CAPTURE monkeypatch separately, because it needs
stub torch/capture_metric modules to be registered before the patch is imported.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
CODE = HERE.parent / "code"
sys.path.insert(0, str(CODE))

import captioners  # noqa: E402
from caption_io import load_captions, resolve_prompt  # noqa: E402

N = 4
_pass = _fail = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _pass, _fail
    if cond:
        print(f"  PASS  {name}")
        _pass += 1
    else:
        print(f"  FAIL  {name} {detail}")
        _fail += 1


def _img(path: Path, seed: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)).save(path)
    return str(path)


# ---------------------------------------------------------------------------
# Fixtures: one tiny dataset per benchmark, in that benchmark's native schema
# ---------------------------------------------------------------------------

def build_fixtures(root: Path) -> dict:
    ids = {}

    cap = root / "DetailCaps-4870"
    ids["capture"] = [f"detailcaps_{i:05d}" for i in range(N)]
    json.dump([{"image_id": i, "image": _img(cap / "images" / f"{i}.png", n),
                "references": [f"ref a {n}", f"ref b {n}"]}
               for n, i in enumerate(ids["capture"])],
              open(cap / "detailcaps_4870.json", "w"), indent=2)

    cc = root / "CompreCap_dataset"
    ids["comprecap"] = [f"000000{i:06d}.jpg" for i in range(N)]
    for n, i in enumerate(ids["comprecap"]):
        _img(cc / "images" / i, 100 + n)
    json.dump([{"image": i, "captions": [f"cc ref {n}"]} for n, i in enumerate(ids["comprecap"])],
              open(cc / "anno.json", "w"), indent=2)

    cb = root / "CapsBench"
    ids["capsbench"] = [f"capsbench_{i:04d}" for i in range(N)]
    json.dump([{"image_id": i, "image": _img(cb / "images" / f"{i}.png", 200 + n),
                "questions": [{"question": f"q{n}?", "answer": "yes", "category": "object"}]}
               for n, i in enumerate(ids["capsbench"])],
              open(cb / "capsbench.json", "w"), indent=2)

    pr = root / "PerceptionRubrics"
    ids["perceptionrubrics"] = [f"pr_{i:06d}" for i in range(N)]
    json.dump([{"id": i, "image": _img(pr / "images" / f"{i}.png", 300 + n)}
               for n, i in enumerate(ids["perceptionrubrics"])],
              open(pr / "perceptionrubrics.json", "w"), indent=2)

    return ids


def build_caption_files(root: Path, ids: dict) -> dict:
    caps = root / "caps"
    caps.mkdir(parents=True, exist_ok=True)
    paths = {}

    for bench, keys in ids.items():
        p = caps / f"{bench}.canonical.jsonl"
        with open(p, "w") as f:
            for k in keys:
                f.write(json.dumps({"image_id": k, "caption": f"canonical {k}"}) + "\n")
        paths[(bench, "canonical")] = p

    for bench in ("capture", "comprecap"):
        p = caps / f"{bench}.legacy.json"
        json.dump({k: f"legacy {k}" for k in ids[bench]}, open(p, "w"), indent=2)
        paths[(bench, "legacy")] = p

    p = caps / "perceptionrubrics.predictions.jsonl"
    with open(p, "w") as f:
        for k in ids["perceptionrubrics"]:
            f.write(json.dumps({"id": k, "prediction": f"pr {k}"}) + "\n")
    paths[("perceptionrubrics", "predictions")] = p

    p = caps / "capsbench.incomplete.jsonl"
    with open(p, "w") as f:
        for k in ids["capsbench"][:-1]:
            f.write(json.dumps({"image_id": k, "caption": "x"}) + "\n")
    paths[("capsbench", "incomplete")] = p

    return paths


# ---------------------------------------------------------------------------
# 1. Format unification
# ---------------------------------------------------------------------------

def test_dialects(caps: dict) -> None:
    print("=== format: all three dialects load through one reader ===")
    for (bench, dialect), path in sorted(caps.items()):
        want = N - 1 if dialect == "incomplete" else N
        try:
            d = load_captions(path)
            check(f"{bench}/{dialect} loads", len(d) == want,
                  f"n={len(d)} expected {want}")
        except Exception as e:  # noqa: BLE001
            check(f"{bench}/{dialect} loads", False, f"{type(e).__name__}: {e}")

    # malformed rows must be rejected, not silently dropped
    bad = caps[("capsbench", "canonical")].parent / "bad.jsonl"
    bad.write_text('{"image_id": "x", "wrong_key": "y"}\n')
    try:
        load_captions(bad)
        check("malformed row rejected", False, "no exception raised")
    except ValueError as e:
        check("malformed row rejected", "'image_id'+'caption'" in str(e), str(e)[:120])


# ---------------------------------------------------------------------------
# 2. --captions-file injection (offline: no key present)
# ---------------------------------------------------------------------------

def test_injection(root: Path, out: Path, caps: dict) -> None:
    import run_capture, run_comprecap, run_capsbench, run_perceptionrubrics

    print("=== injection: no API key, captions from file ===")
    cases = [
        ("capture/canonical", run_capture,
         ["--data", str(root / "DetailCaps-4870"), "--out", str(out / "c1/score.json")],
         caps[("capture", "canonical")], 0),
        ("capture/legacy-dict", run_capture,
         ["--data", str(root / "DetailCaps-4870"), "--out", str(out / "c2/score.json")],
         caps[("capture", "legacy")], 0),
        ("comprecap/canonical", run_comprecap,
         ["--data", str(root / "CompreCap_dataset"), "--out", str(out / "c3")],
         caps[("comprecap", "canonical")], 0),
        ("comprecap/legacy-dict", run_comprecap,
         ["--data", str(root / "CompreCap_dataset"), "--out", str(out / "c4")],
         caps[("comprecap", "legacy")], 0),
        ("capsbench/canonical", run_capsbench,
         ["--data", str(root / "CapsBench"), "--out", str(out / "c5/judged.jsonl")],
         caps[("capsbench", "canonical")], 0),
        ("perceptionrubrics/predictions", run_perceptionrubrics,
         ["--data", str(root / "PerceptionRubrics"), "--out", str(out / "c6")],
         caps[("perceptionrubrics", "predictions")], 0),
        ("perceptionrubrics/canonical", run_perceptionrubrics,
         ["--data", str(root / "PerceptionRubrics"), "--out", str(out / "c7")],
         caps[("perceptionrubrics", "canonical")], 0),
        ("capsbench/incomplete -> error", run_capsbench,
         ["--data", str(root / "CapsBench"), "--out", str(out / "c8/judged.jsonl")],
         caps[("capsbench", "incomplete")], "SystemExit"),
    ]

    for name, mod, extra, captions_file, want in cases:
        argv = ["--captioner", "test-model", "--caption-only",
                "--captions-file", str(captions_file)] + extra
        try:
            rc = mod.main(argv)
        except SystemExit as e:
            rc = "SystemExit" if str(e).startswith(("[CAPTURE]", "[CompreCap]", "[CapsBench]", "[PR]")) else e
        check(name, rc == want, f"got {rc!r}, want {want!r}")

    # artifacts land where the contract says
    cap_jsonl = out / "c1" / "score.captions.jsonl"
    check("capture wrote <out>.captions.jsonl", cap_jsonl.exists(), str(cap_jsonl))
    check("comprecap wrote canonical jsonl", (out / "c3" / "captions.jsonl").exists())
    check("comprecap wrote legacy dict for evaluator", (out / "c3" / "captions.json").exists())
    if cap_jsonl.exists():
        first = json.loads(cap_jsonl.read_text().splitlines()[0])
        check("canonical keys are image_id+caption",
              set(first) == {"image_id", "caption"}, sorted(first))
    check("pr wrote predictions.jsonl", (out / "c6" / "predictions.jsonl").exists())
    if (out / "c6" / "predictions.jsonl").exists():
        first = json.loads((out / "c6" / "predictions.jsonl").read_text().splitlines()[0])
        check("pr predictions use official id+prediction",
              set(first) == {"id", "prediction"}, sorted(first))


# ---------------------------------------------------------------------------
# 3. Prompt wiring -- the experiment's independent variable
# ---------------------------------------------------------------------------

def test_prompt_wiring(root: Path, out: Path) -> None:
    import run_comprecap, run_capsbench, run_capture, run_perceptionrubrics

    sent = []

    def fake_post(url, headers, payload, retries=8):
        sent.append(payload)
        return {"choices": [{"message": {"content": "FAKE"}}]}

    real_post = captioners._post
    captioners._post = fake_post
    try:
        print("=== prompt: reaches the system role of the real payload ===")
        c = captioners.build_captioner("test-model", prompt_override="MARK", max_tokens=4096)
        check("prompt_override -> .system", c.system == "MARK", repr(c.system))
        check("max_tokens honoured", c.max_tokens == 4096, c.max_tokens)

        sent.clear()
        c.caption(np.zeros((16, 16, 3), dtype=np.uint8))
        p = sent[-1]
        check("system role carries the prompt", p["messages"][0]["content"] == "MARK")
        check("max_tokens on the wire", p["max_tokens"] == 4096)
        texts = [b["text"] for b in p["messages"][1]["content"] if b.get("type") == "text"]
        check("user turn is the hardcoded default",
              texts == ["Describe this image in detail."], texts)

        print("=== prompt: per-runner default and override ===")
        arm = root / "arm.txt"
        arm.write_text("ARM_MARKER")
        runners = [
            ("capture", run_capture, ["--data", str(root / "DetailCaps-4870"),
                                      "--out", str(out / "w1.json")],
             captioners.CAPTIONER_PROMPT),
            ("capsbench", run_capsbench, ["--data", str(root / "CapsBench"),
                                          "--out", str(out / "w2.jsonl")],
             captioners.CAPTIONER_PROMPT),
            ("comprecap", run_comprecap, ["--data", str(root / "CompreCap_dataset"),
                                          "--out", str(out / "w3")],
             captioners.CAPTIONER_PROMPT),
            ("perceptionrubrics", run_perceptionrubrics,
             ["--data", str(root / "PerceptionRubrics"), "--out", str(out / "w4")],
             run_perceptionrubrics.PR_PROMPT),
        ]
        for name, mod, extra, default in runners:
            sent.clear()
            mod.main(["--captioner", "test-model", "--caption-only"] + extra)
            check(f"{name}: default prompt on the wire",
                  {q["messages"][0]["content"] for q in sent} == {default},
                  f"got {[q['messages'][0]['content'][:40] for q in sent]}")

            sent.clear()
            mod.main(["--captioner", "test-model", "--caption-only",
                      "--prompt-file", str(arm)] + extra)
            check(f"{name}: --prompt-file overrides",
                  {q["messages"][0]["content"] for q in sent} == {"ARM_MARKER"},
                  f"got {[q['messages'][0]['content'][:40] for q in sent]}")

        print("=== prompt: resolve_prompt precedence ===")
        ns = lambda **kw: type("NS", (), kw)()  # noqa: E731
        check("--prompt used", resolve_prompt(ns(prompt="P", prompt_file="")) == "P")
        check("empty -> default", resolve_prompt(ns(prompt="", prompt_file=""), default="D") == "D")
        check("--prompt-file beats --prompt",
              resolve_prompt(ns(prompt="P", prompt_file=str(arm))) == "ARM_MARKER")
    finally:
        captioners._post = real_post


# ---------------------------------------------------------------------------
# 3a. Image paths -- the shipped JSONs carry another machine's absolute paths
# ---------------------------------------------------------------------------

def test_image_path_fallback() -> None:
    """DetailCaps/CapsBench record absolute paths from the build machine
    (`/mnt/shared_resources/...`), which exist nowhere else. The loader must fall
    back to the copy shipped alongside the JSON -- and only when it is really there.
    """
    import datasets

    print("=== image: stale absolute path falls back to <data>/images ===")
    tmp = Path(tempfile.mkdtemp(prefix="capeval_imgpath_"))
    try:
        (tmp / "images").mkdir(parents=True)
        real = tmp / "images" / "sample_0000.jpg"
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(real)

        stale = {"image": "/mnt/shared_resources/datasets/baselines/X/images/sample_0000.jpg"}
        check("stale path resolves to the local copy",
              datasets.resolve_image(tmp, stale) == str(real),
              datasets.resolve_image(tmp, stale))

        check("a valid path is left alone",
              datasets.resolve_image(tmp, {"image": str(real)}) == str(real))

        gone = {"image": "/nope/nothing_here.jpg"}
        check("a genuinely missing image is not papered over",
              datasets.resolve_image(tmp, gone) == gone["image"],
              datasets.resolve_image(tmp, gone))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# 3b. Image payload -- the declared media type must match the encoded bytes
# ---------------------------------------------------------------------------

def test_image_payload() -> None:
    """Anthropic rejects a data URI whose declared type disagrees with the bytes.

    OpenAI-compatible routes ignore the label, so a mismatch is silent on GPT and
    fatal on Claude -- and it presents as a flaky endpoint (the retry loop burns
    ~2 minutes, then fails) rather than as a bad request. Keep the two in step.
    """
    import base64

    print("=== image: data URI label matches the encoded bytes ===")
    sent = []

    def fake_post(url, headers, payload, retries=8):
        sent.append(payload)
        return {"choices": [{"message": {"content": "FAKE"}}]}

    real_post = captioners._post
    captioners._post = fake_post
    try:
        img = np.zeros((24, 24, 3), dtype=np.uint8)

        captioners.build_captioner("test-model").caption(img)
        url = sent[-1]["messages"][1]["content"][0]["image_url"]["url"]
        mime, sep, b64 = url.partition(";base64,")
        check("image sent as a base64 data URI", bool(sep) and mime.startswith("data:image/"),
              mime)
        _check_bytes_match(mime[len("data:"):], base64.b64decode(b64), "openai route")

        print("=== image: Anthropic media_type matches the bytes too ===")
        sent.clear()
        captioners.AnthropicCaptioner(api_key="x").caption(img)
        src = sent[-1]["messages"][0]["content"][0]["source"]
        _check_bytes_match(src["media_type"], base64.b64decode(src["data"]), "anthropic route")
    finally:
        captioners._post = real_post


def _check_bytes_match(declared: str, raw: bytes, label: str) -> None:
    """The declared MIME must match the magic bytes actually sent."""
    magic = {"image/jpeg": b"\xff\xd8\xff",
             "image/png": b"\x89PNG\r\n\x1a\n"}.get(declared)
    if magic is None:
        check(f"{label}: known media type", False, declared)
        return
    check(f"{label}: declared {declared} matches the bytes", raw.startswith(magic),
          f"starts with {raw[:8]!r}")


# ---------------------------------------------------------------------------
# 4. Generator -> runner handoff (the seamless pairing)
# ---------------------------------------------------------------------------

def test_handoff(root: Path, out: Path) -> None:
    import generate_captions
    import run_comprecap

    print("=== handoff: generate_captions output feeds a runner ===")
    sent = []

    def fake_post(url, headers, payload, retries=8):
        sent.append(payload)
        return {"choices": [{"message": {"content": "GEN"}}]}

    real_post = captioners._post
    captioners._post = fake_post
    try:
        gen = out / "gen" / "captions.jsonl"
        rc = generate_captions.main([
            "--benchmark", "comprecap", "--data", str(root / "CompreCap_dataset"),
            "--captioner", "test-model", "--out", str(gen), "--workers", "2"])
        check("generator exits 0", rc == 0, rc)
        check("generator wrote canonical jsonl", gen.exists())
        check("meta sidecar written", (gen.parent / (gen.name + ".meta.json")).exists())
        if (gen.parent / (gen.name + ".meta.json")).exists():
            meta = json.loads((gen.parent / (gen.name + ".meta.json")).read_text())
            check("meta records the prompt hash", len(meta.get("system_prompt_sha256", "")) == 64)

        rc2 = run_comprecap.main([
            "--data", str(root / "CompreCap_dataset"), "--captioner", "test-model",
            "--captions-file", str(gen), "--out", str(out / "handoff"), "--caption-only"])
        check("runner consumes generator output", rc2 == 0, rc2)
    finally:
        captioners._post = real_post


# ---------------------------------------------------------------------------
# 5. Incremental checkpointing -- a long generation must not lose its progress
# ---------------------------------------------------------------------------

def test_incremental_flush(root: Path, out: Path) -> None:
    """Generation runs for hours at full scale, so it can outlive a batch job's
    wall-clock limit. It must checkpoint as it goes, and each checkpoint must be
    complete: a truncated file would break the `--resume` that reads it back.
    """
    import generate_captions

    print("=== checkpoint: progress lands on disk before the run ends ===")

    calls = []

    def fake_post(url, headers, payload, retries=8):
        calls.append(payload)
        return {"choices": [{"message": {"content": "C"}}]}

    # Every checkpoint is a temp-file write followed by os.replace. Spying on the
    # rename reports each checkpoint's row count exactly, with no scheduling race.
    real_replace = generate_captions.os.replace
    snapshots = []

    def _rows(path):
        """Count caption rows without going through load_captions, which dispatches
        on the file extension and so rejects the ".tmp" checkpoint name."""
        return sum(1 for ln in Path(path).read_text(encoding="utf-8").splitlines()
                   if ln.strip())

    def spy_replace(src, dst):
        snapshots.append((Path(dst).name, _rows(src)))
        return real_replace(src, dst)

    real_post = captioners._post
    captioners._post = fake_post
    generate_captions.os.replace = spy_replace
    try:
        gen = out / "flush" / "captions.jsonl"
        rc = generate_captions.main([
            "--benchmark", "comprecap", "--data", str(root / "CompreCap_dataset"),
            "--captioner", "test-model", "--out", str(gen),
            "--workers", "1", "--flush-every", "2"])
        check("flush run exits 0", rc == 0, rc)
        check("checkpointed every 2 images",
              [n for _, n in snapshots] == [2, 4], snapshots)
        check("checkpoints replace the real output",
              all(name == gen.name for name, _ in snapshots), snapshots)
        check("no .tmp left behind", not gen.with_name(gen.name + ".tmp").exists())
        check("final file is complete", len(load_captions(gen)) == N,
              len(load_captions(gen)))

        # a checkpointed file must be re-readable by --resume, not just by load
        print("=== checkpoint: a partial file is a valid --resume input ===")
        part = out / "flush2" / "captions.jsonl"
        part.parent.mkdir(parents=True, exist_ok=True)
        keep = []
        real_replace2 = real_replace  # the true os.replace, not the first spy

        def spy2(src, dst):
            # Freeze the run at its first checkpoint, as a mid-run kill would.
            if not keep:
                shutil.copyfile(src, part)
                keep.append(("frozen", _rows(src)))
            return real_replace2(src, dst)

        generate_captions.os.replace = spy2
        generate_captions.main([
            "--benchmark", "comprecap", "--data", str(root / "CompreCap_dataset"),
            "--captioner", "test-model", "--out", str(part.parent / "live.jsonl"),
            "--workers", "1", "--flush-every", "2"])
        check("froze a partial file mid-run", len(keep) == 1 and keep[0][1] == 2, keep)
        if keep:
            load_captions(part)  # raises if a truncated checkpoint was written
            check("partial file loads cleanly", len(load_captions(part)) == 2)

            print("=== checkpoint: --resume tops up the frozen file ===")
            calls.clear()
            before = len(calls)
            rc = generate_captions.main([
                "--benchmark", "comprecap", "--data", str(root / "CompreCap_dataset"),
                "--captioner", "test-model", "--out", str(part),
                "--workers", "1", "--flush-every", "2", "--resume"])
            check("resume exits 0", rc == 0, rc)
            check("resume only redid the missing images",
                  len(calls) - before == N - 2, f"{len(calls) - before} calls, want {N - 2}")
            check("resume produced the full set", len(load_captions(part)) == N,
                  len(load_captions(part)))
    finally:
        captioners._post = real_post
        generate_captions.os.replace = real_replace


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="capeval_tests_"))
    try:
        root = tmp / "data"
        out = tmp / "out"
        root.mkdir(parents=True)
        out.mkdir(parents=True)

        ids = build_fixtures(root)
        caps = build_caption_files(root, ids)

        test_dialects(caps)
        test_injection(root, out, caps)
        test_prompt_wiring(root, out)
        test_image_path_fallback()
        test_image_payload()
        test_handoff(root, out)
        test_incremental_flush(root, out)

        print(f"\nRESULT: {_pass} passed, {_fail} failed")
        return 1 if _fail else 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
