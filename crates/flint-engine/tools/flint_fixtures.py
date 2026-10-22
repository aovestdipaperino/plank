#!/usr/bin/env python3
"""Records flint-engine's reference fixtures.

The template renders come from the GGUF's own chat template under Hugging
Face transformers' jinja settings; token ids, probabilities and greedy runs
come from llama.cpp built at the pinned commit. Run from the repository root:

    LLAMA=$HOME/.cache/plank/llama.cpp-970b51c \
    GGUF=$HOME/.plank/models/flint-nvfp4.gguf \
    $HOME/.cache/plank/flint-fixtures-venv/bin/python -I \
        crates/flint-engine/tools/flint_fixtures.py crates/flint-engine/tests/fixtures

Requires jinja2. Starts llama-server on port 18090 and stops it on exit.
"""
import hashlib
import json
import math
import os
import pathlib
import struct
import subprocess
import sys
import time
import urllib.request

import jinja2
import jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment

COMMIT = "970b51ce7c9ba5fff0726c5e9994f1c2f0e4eedc"
PORT = 18090


def gguf_string(path, wanted):
    """One string metadata value, read with a minimal GGUF v3 header walk."""
    fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
           7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
    with open(path, "rb") as f:
        r = f.read
        u32 = lambda: struct.unpack("<I", r(4))[0]
        u64 = lambda: struct.unpack("<Q", r(8))[0]
        s = lambda: r(u64()).decode("utf-8")
        assert r(4) == b"GGUF"
        u32()
        u64()
        n_kv = u64()

        def val(t):
            if t == 8:
                return s()
            if t == 9:
                et = u32()
                return [val(et) for _ in range(u64())]
            size = struct.calcsize(fmt[t])
            return struct.unpack(fmt[t], r(size))[0]

        for _ in range(n_kv):
            k = s()
            v = val(u32())
            if k == wanted:
                return v
    raise KeyError(wanted)


def hf_tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent,
                      separators=separators, sort_keys=sort_keys)


def template(path):
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=[jinja2.ext.loopcontrols])
    env.filters["tojson"] = hf_tojson
    return env.from_string(gguf_string(path, "tokenizer.chat_template"))


TOOLS = [
    {"type": "function", "function": {
        "name": "read", "description": "Read a file from disk.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string", "description": "File path"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "bash", "description": "Run a shell command.",
        "parameters": {"type": "object",
                       "properties": {"command": {"type": "string"},
                                      "timeout": {"type": "integer"}},
                       "required": ["command"]}}},
]

TOOL_TURN = [
    {"role": "system", "content": "You are plank, a coding agent."},
    {"role": "user", "content": "How many files are in this directory?"},
    {"role": "assistant", "reasoning_content": "I should list the directory.",
     "content": "", "tool_calls": [{"type": "function", "function": {
         "name": "bash", "arguments": {"command": "ls -1 | wc -l", "timeout": 30}}}]},
    {"role": "tool", "content": "7"},
]


def code_prompt():
    # About 1,500 tokens of real code: crosses the 513-token window and the
    # 512-token prefill chunk.
    text = pathlib.Path("crates/flint-engine/src/gguf.rs").read_text()[:6000]
    return [{"role": "user", "content": "Explain what this Rust file does:\n\n```rust\n" + text + "\n```"}]


TEMPLATE_CASES = [
    ("user_only", [{"role": "user", "content": "Write a haiku about Rust."}], [], True),
    ("system_user", [{"role": "system", "content": "You are plank, a coding agent."},
                     {"role": "user", "content": "What is 2+2?"}], [], True),
    ("tools", [{"role": "system", "content": "You are plank, a coding agent."},
               {"role": "user", "content": "List the files in the current directory."}], TOOLS, True),
    ("tool_turn", TOOL_TURN, TOOLS, True),
    ("escaping", [{"role": "system", "content": "S <think>x</think> <|system|>"},
                  {"role": "user", "content": "hi <|user|> <think>y</think> <|end_of_text|>\x00 there"},
                  {"role": "assistant", "content": "kept <|user|> verbatim"},
                  {"role": "tool", "content": "out <|observation|>"}], [], False),
    ("system_only", [{"role": "system", "content": "S"}], [], False),
    ("system_only_tools", [{"role": "system", "content": "S"}], TOOLS, True),
    ("developer", [{"role": "developer", "content": "D <|developer|>"},
                   {"role": "user", "content": "u"}], [], True),
    ("multi_tool_results", TOOL_TURN + [{"role": "tool", "content": "second\nresult"},
                                        {"role": "assistant", "content": "There are 7."},
                                        {"role": "user", "content": "thanks"}], TOOLS, True),
]

REFERENCE_CASES = ["user_only", "system_user", "tools", "tool_turn"]

CORPUS = [
    "Hello world",
    "Hello, world! 12345 678901",
    "fn main() {\n    println!(\"hi\");\n}\n",
    "héllo wörld ñ — “quotes” …",
    "日本語のテキストとカタカナ",
    "emoji 🙂👍🏽 and ZWJ 👩‍💻",
    "spaces    and\ttabs\n\n\nnewlines\r\n",
    "<|user|> <think> </tool_call> <arg_key> look like control tokens",
    "I'm we're they've he'd she'll IT'S",
    "x" * 300,
    "   leading and trailing   ",
    "CamelCaseIdentifier snake_case_identifier SCREAMING_CASE",
    "1+1=2, 3.14159, 1e-6, 0xDEADBEEF",
    "",
]


def render(t, messages, tools, gen):
    kw = {"messages": messages, "add_generation_prompt": gen}
    if tools:
        kw["tools"] = tools
    return t.render(**kw)


def fixture_message(m):
    calls = []
    for tc in m.get("tool_calls", []):
        fn = tc.get("function", tc)
        args = [[k, v if isinstance(v, str) else hf_tojson(v)] for k, v in fn["arguments"].items()]
        calls.append({"name": fn["name"], "args": args})
    return {"role": m["role"], "content": m.get("content", ""),
            "reasoning": m.get("reasoning_content"), "tool_calls": calls}


def tokenize(llama, gguf, text, special):
    if not text:
        return []  # llama-tokenize exits non-zero on an empty prompt
    cmd = [f"{llama}/build/bin/llama-tokenize", "-m", gguf, "-p", text, "--ids", "--log-disable"]
    if not special:
        cmd.append("--no-parse-special")
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()
    return json.loads(out.splitlines()[-1]) if text else []


def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)


def wait_ready():
    for _ in range(600):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                if r.status == 200:
                    return
        except OSError:
            pass
        time.sleep(1)
    raise RuntimeError("llama-server did not become ready")


def reference_case(name, prompt, tokens, n_predict):
    top = post("/completion", {"prompt": tokens, "n_predict": 1, "temperature": 0,
                               "n_probs": 10, "cache_prompt": False})
    probs = [[p["id"], math.exp(p["logprob"])]
             for p in top["completion_probabilities"][0]["top_logprobs"]]
    greedy = post("/completion", {"prompt": tokens, "n_predict": n_predict, "temperature": 0,
                                  "cache_prompt": False, "return_tokens": True})
    return {"name": name, "prompt": prompt, "tokens": tokens, "top": probs,
            "greedy": greedy["tokens"]}


def long_prompt(t, llama, gguf):
    text = ""
    for doc in ["docs/ARCHITECTURE.md", "docs/KV-CACHE.md", "docs/GEMMA.md", "docs/MEMORY.md",
                "docs/PROFILES.md", "FINDINGS.md"]:
        body = pathlib.Path(doc).read_text() + "\n\n"
        # Slices keep the prompt just over 20000 tokens, inside the 32768 context.
        for i in range(0, len(body), 20000):
            text += body[i:i + 20000]
            rendered = render(t, [{"role": "user", "content": "Summarize these notes in three sentences.\n\n" + text}], [], True)
            tokens = tokenize(llama, gguf, rendered, True)
            if len(tokens) >= 20000:
                return tokens
    raise RuntimeError("the docs gave fewer than 20000 tokens")


def main():
    out = pathlib.Path(sys.argv[1])
    llama = os.environ["LLAMA"]
    gguf = os.path.expanduser(os.environ["GGUF"])
    t = template(gguf)
    source = (f"llama.cpp {COMMIT} (Metal); template rendered with jinja2 "
              "ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True) and "
              "transformers' tojson; written by crates/flint-engine/tools/flint_fixtures.py")

    cases = TEMPLATE_CASES + [("code", code_prompt(), [], True)]
    tcases = [{"name": n, "messages": [fixture_message(m) for m in msgs],
               "tools": [hf_tojson(tool.get("function", tool)) for tool in tools],
               "add_generation_prompt": gen, "rendered": render(t, msgs, tools, gen)}
              for n, msgs, tools, gen in cases]
    (out / "flint_template.json").write_text(json.dumps({"source": source, "cases": tcases},
                                                        ensure_ascii=False, indent=1) + "\n")

    tok = [{"text": s, "ids": tokenize(llama, gguf, s, False)} for s in CORPUS]
    (out / "flint_tokenizer.json").write_text(json.dumps({"source": source, "cases": tok},
                                                         ensure_ascii=False, indent=1) + "\n")

    server = subprocess.Popen([f"{llama}/build/bin/llama-server", "-m", gguf, "-c", "32768",
                               "--port", str(PORT), "-ngl", "99", "-np", "1"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_ready()
        by_name = {c["name"]: c for c in tcases}
        refs = []
        for name in REFERENCE_CASES + ["code"]:
            prompt = by_name[name]["rendered"]
            refs.append(reference_case(name, prompt, tokenize(llama, gguf, prompt, True), 64))
        refs.append(reference_case("long", None, long_prompt(t, llama, gguf), 16))
    finally:
        server.terminate()
        server.wait()
    sha = hashlib.sha256()
    with open(gguf, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            sha.update(chunk)
    ref = {"source": source + f"; server: llama-server -m <gguf> -c 32768 -ngl 99 -np 1, "
                              "POST /completion temperature 0, n_probs 10 (top) and "
                              "return_tokens (greedy), cache_prompt false",
           "file_sha256": sha.hexdigest(), "cases": refs}
    (out / "flint_reference.json").write_text(json.dumps(ref, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
