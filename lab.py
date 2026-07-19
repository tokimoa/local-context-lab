"""local-context-lab — 長文コンテキスト定石のローカルLLM同一条件比較ハーネス。

タスクA（複数会議のTODO統合）の4アプローチを実装する。設計はREADME参照。
採点は決定的（ゴールド照合）で、LLM-as-a-Judgeを使わない。
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

TODO_JSON_SPEC = (
    '{"todos": [{"item": "内容", "assignee": "担当者名またはnull", '
    '"due": "YYYY-MM-DDまたはnull", "meeting": "会議名"}]}'
)


# ---------------------------------------------------------------- モデル実行

@dataclass
class Tally:
    """アプローチ横断で比較するコスト集計。"""
    calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    wall_s: float = 0.0
    events: list = field(default_factory=list)


class Runner:
    def __init__(self, model_id: str):
        from mlx_lm import load

        self.model_id = model_id
        self.model, self.tokenizer = load(model_id)

    def generate(self, prompt: str, tally: Tally, label: str, max_tokens: int = 4096) -> str:
        from mlx_lm import generate

        chat = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, tokenize=False, enable_thinking=False,
        )
        n_in = len(self.tokenizer.encode(chat))
        t0 = time.monotonic()
        text = generate(self.model, self.tokenizer, prompt=chat,
                        max_tokens=max_tokens, verbose=False)
        dt = time.monotonic() - t0
        text = re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.DOTALL).strip()
        tally.calls += 1
        tally.prompt_tokens += n_in
        tally.output_tokens += len(self.tokenizer.encode(text))
        tally.wall_s += dt
        tally.events.append({"label": label, "in": n_in, "s": round(dt, 1)})
        return text


def parse_todos(text: str) -> tuple[list[dict], bool]:
    """出力からtodos配列を取り出す。返り値は (todos, salvaged)。

    ローカル4bitモデルは2,000トークン級のJSON出力で稀にキーを破損させる実測がある
    （"due" が width" や 食: に化け、厳密パースだと1文字の破損で全損になる。2026-07-20、
    gemma-4-26b-a4b-4bit）。厳密パース失敗時は、壊れたオブジェクトだけを捨てて
    残りを回収する寛容パースにフォールバックし、salvaged=True を返す。"""
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return [], False
    try:
        data = json.loads(text[start:end + 1])
        todos = data.get("todos", []) if isinstance(data, dict) else []
        return [t for t in todos if isinstance(t, dict) and t.get("item")], False
    except json.JSONDecodeError:
        return _salvage(text), True


def _salvage(text: str) -> list[dict]:
    """破損JSONからの部分回収: ネストのないオブジェクトを個別にパースし、壊れたものだけ捨てる。"""
    todos = []
    for m in re.finditer(r"\{[^{}]*\}", text):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("item"):
            todos.append(obj)
    return todos


def meeting_block(item: dict) -> str:
    m = item["meta"]
    return (f"# 会議: {m['title']} ／ 日付: {m['date']} ／ "
            f"参加者: {'、'.join(m['participants'])}\n\n{item['transcript']}")


EXTRACT_PROMPT = """以下の会議書き起こしから、TODO（宿題・アクションアイテム）を抽出してください。

{meeting}

# ルール
- 書き起こしで明示的に割り当てられたTODOのみ。提案・検討中は含めない
- 期日は明示された表現だけを会議日付基準でYYYY-MM-DDに換算。曖昧な表現はnull
- 書き起こしにない情報を補わない

# 出力形式（このJSONのみ）
{spec}"""


# ---------------------------------------------------------------- 4アプローチ

def run_full(runner: Runner, series: list[dict], tally: Tally):
    meetings = "\n\n---\n\n".join(meeting_block(i) for i in series)
    prompt = (f"以下の{len(series)}件の会議書き起こしを読み、全会議のTODO（宿題・アクション"
              f"アイテム）を統合した一覧を作成してください。\n\n{meetings}\n\n# ルール\n"
              "- 書き起こしで明示的に割り当てられたTODOのみ。提案・検討中は含めない\n"
              "- 期日は明示された表現だけを各会議の日付基準でYYYY-MM-DDに換算。曖昧な表現はnull\n"
              f"\n# 出力形式（このJSONのみ）\n{TODO_JSON_SPEC}")
    raw = runner.generate(prompt, tally, "full", max_tokens=8192)
    todos, salvaged = parse_todos(raw)
    return todos, raw, salvaged


def run_compaction(runner: Runner, series: list[dict], tally: Tally,
                   limit_chars: int = 2000):
    summary = "（まだ何もありません）"
    for i, item in enumerate(series):
        prompt = (f"あなたは複数回の会議のTODOを追跡しています。\n\n# これまでのまとめ\n{summary}\n\n"
                  f"# 新しい会議\n{meeting_block(item)}\n\n"
                  f"これまでのまとめに、新しい会議のTODO（担当者・期日つき。明示的に割り当てられた"
                  f"もののみ、期日は会議日付基準でYYYY-MM-DD換算、曖昧ならnull）を統合した"
                  f"「更新版のまとめ」を{limit_chars}字以内で出力してください。TODOの情報を"
                  f"落とさないことを最優先にしてください。")
        summary = runner.generate(prompt, tally, f"compact-{i + 1}", max_tokens=3072)[:limit_chars * 2]
    final = runner.generate(
        f"以下のまとめから、TODO一覧を出力してください。\n\n{summary}\n\n"
        f"# 出力形式（このJSONのみ）\n{TODO_JSON_SPEC}",
        tally, "compact-final", max_tokens=8192)
    todos, salvaged = parse_todos(final)
    return todos, final, salvaged


def _extract_each(runner: Runner, series: list[dict], tally: Tally, label: str) -> list[list[dict]]:
    out = []
    for i, item in enumerate(series):
        text = runner.generate(
            EXTRACT_PROMPT.format(meeting=meeting_block(item), spec=TODO_JSON_SPEC),
            tally, f"{label}-{i + 1}", max_tokens=4096)
        todos, _ = parse_todos(text)
        for t in todos:
            t.setdefault("meeting", item["meta"]["title"])
        out.append(todos)
    return out


def run_note(runner: Runner, series: list[dict], tally: Tally):
    # 構造化ノート: 会議ごとの抽出結果をハーネスが蓄積し、ノート自体が成果物になる
    return [t for todos in _extract_each(runner, series, tally, "note") for t in todos], "", False


def run_subagent(runner: Runner, series: list[dict], tally: Tally):
    # 抽出はnoteと同一。統合をLLM呼び出しで行う（重複整理・表現統一を任せる）
    extracted = _extract_each(runner, series, tally, "sub")
    notes = "\n".join(json.dumps({"todos": todos}, ensure_ascii=False) for todos in extracted)
    final = runner.generate(
        f"以下は各会議から抽出したTODOのJSONです。全体を統合し、重複があれば1件に"
        f"まとめて出力してください。項目の内容・担当者・期日は変更しないこと。\n\n{notes}\n\n"
        f"# 出力形式（このJSONのみ）\n{TODO_JSON_SPEC}",
        tally, "sub-merge", max_tokens=8192)
    todos, salvaged = parse_todos(final)
    return todos, final, salvaged


APPROACHES = {"full": run_full, "compaction": run_compaction,
              "note": run_note, "subagent": run_subagent}


# ---------------------------------------------------------------- 採点（決定的）

def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a or "", b or "").ratio()


def score(candidates: list[dict], series: list[dict]) -> dict:
    gold = [t | {"_meeting": i["meta"]["title"]}
            for i in series for t in i["gold"]["todos"]]
    matched_cand: set[int] = set()
    recall_hits = due_gold = due_correct = due_wrong = 0
    for gt in gold:
        best, best_score = None, 0.0
        for ci, ct in enumerate(candidates):
            s = _sim(gt["item"], str(ct.get("item", "")))
            if ct.get("assignee") and str(ct.get("assignee")) == str(gt.get("assignee") or ""):
                s += 0.15
            if s > best_score:
                best, best_score = ci, s
        if best is not None and best_score >= 0.3:
            recall_hits += 1
            matched_cand.add(best)
            if gt.get("due"):
                due_gold += 1
                cand_due = str(candidates[best].get("due") or "")
                if cand_due == gt["due"]:
                    due_correct += 1
                elif re.match(r"\d{4}-\d{2}-\d{2}", cand_due):
                    due_wrong += 1
        elif gt.get("due"):
            due_gold += 1  # 見つからなかったTODOの期日は未達扱い（分母に含める）
    spurious = len(candidates) - len(matched_cand)
    return {
        "gold_n": len(gold), "cand_n": len(candidates),
        "todo_recall": recall_hits / len(gold) if gold else 0.0,
        "spurious_rate": spurious / len(candidates) if candidates else 0.0,
        "due_gold_n": due_gold,
        "due_acc": due_correct / due_gold if due_gold else None,
        "due_wrong": due_wrong,
    }


# ---------------------------------------------------------------- CLI

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="minutes形式JSONL（gold.todos必須）")
    parser.add_argument("--model", required=True)
    parser.add_argument("--series-size", type=int, default=8)
    parser.add_argument("--approach", required=True, choices=sorted(APPROACHES))
    parser.add_argument("--limit-series", type=int, default=0, help="シリーズ数の上限（0=全部）")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    items = [json.loads(l) for l in Path(args.data).read_text().splitlines() if l.strip()]
    items.sort(key=lambda x: x["id"])
    series_list = [items[i:i + args.series_size]
                   for i in range(0, len(items) - args.series_size + 1, args.series_size)]
    if args.limit_series:
        series_list = series_list[: args.limit_series]

    runner = Runner(args.model)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        done = {json.loads(l)["series"] for l in out_path.read_text().splitlines() if l.strip()}

    with out_path.open("a") as f:
        for si, series in enumerate(series_list):
            if si in done:
                continue
            tally = Tally()
            candidates, final_raw, salvaged = APPROACHES[args.approach](runner, series, tally)
            row = {
                "series": si, "approach": args.approach, "model": args.model,
                "n_meetings": len(series), **score(candidates, series),
                "calls": tally.calls, "prompt_tokens": tally.prompt_tokens,
                "output_tokens": tally.output_tokens, "wall_s": round(tally.wall_s, 1),
                "events": tally.events, "candidates": candidates,
                "parse_salvaged": salvaged, "final_raw": final_raw[:8000],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{args.approach} series {si}] recall={row['todo_recall']:.2f} "
                  f"spurious={row['spurious_rate']:.2f} tokens={row['prompt_tokens']} "
                  f"wall={row['wall_s']}s", flush=True)
    print(f"saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
