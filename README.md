# local-context-lab

長文コンテキスト処理の定石（compaction・structured note-taking・sub-agent・just-in-time retrieval）を、**デスクトップ級ローカルLLMで同一条件比較**するための実験ハーネスです。

Anthropicが[context engineeringの定石](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)として整理したパターン群は、クラウドの大型モデルを前提に語られることがほとんどです。ローカルLLM（7B〜30B・量子化・限られたコンテキスト長）では、次の固有の事情があります。

- コンテキストを伸ばすとプリフィル時間が二乗で伸び、メモリを圧迫する
- 圧縮や要約を担うのも同じ小型モデルであり、要約品質が全体を律速する
- compactionはKVキャッシュを無効化し、再プリフィルのコストが発生する

このリポジトリは「どのパターンが、ローカルの制約下で、どれだけのトークン・時間・品質のトレードオフを持つか」を数字にするためのものです。

## タスク設計

**タスクA: 複数会議のTODO統合**。複数の会議文字起こし（合計2万字超）から、全会議を横断したTODO一覧（項目・担当者・期日）を作成する。採点はゴールドとの照合で決定的に行う（TODO再現率・幽霊TODO率・期日正答率）。

- `full`: 全会議を1プロンプトに連結（ベースライン）
- `compaction`: 会議を1つずつ読み、上限つきの「まとめ」を毎回更新。最後にまとめから回答
- `note`: 会議ごとにTODOを構造化ノートへ抽出し、ハーネスがノートを蓄積。ノートがそのまま成果物
- `subagent`: 会議ごとに独立コンテキストで抽出し、最後に統合役のLLM呼び出しでマージ

（noteとsubagentは抽出が同一で、マージ戦略＝ハーネス連結かLLM統合かが異なります）

**タスクB: シリーズ横断QA**（計画中）: 「第3回で合意された事項は」型の質問で `full` vs `jit`（該当会議のみを都度読み込む）を比較。

## 測るもの

- 品質: TODO再現率／幽霊TODO率（ゴールドに対応しない項目）／期日正答率
- コスト: 合計プロンプトトークン・出力トークン・LLM呼び出し回数・実時間

## データについて

実験に使う会議データ（[日本語 実務LLMランキング](https://tokimoa.jp/llm-benchmark)の評価データセット）は、評価の汚染防止のため非公開です。ハーネスは `--data` に同形式のJSONL（`{"id", "meta": {"title", "date", "participants"}, "transcript", "gold": {"todos": [...]}}`）を渡せば任意のデータで動きます。

## 使い方

```sh
uv run python lab.py --data path/to/minutes.jsonl \
  --model mlx-community/gemma-4-26b-a4b-it-4bit \
  --series-size 8 --approach compaction --out results/compaction.jsonl
```

要件: Apple Silicon + mlx-lm。

## ライセンス

Apache License 2.0. Copyright (c) 2026 tokimoa.

関連: [jitsumu-metrics](https://github.com/tokimoa/jitsumu-metrics)（反復信頼性指標）／[jitsumu-skills](https://github.com/tokimoa/jitsumu-skills)（実測効果つき業務スキル）
