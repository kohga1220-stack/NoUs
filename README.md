# NoUs

分野横断の知識ネットワークから「遠い分野どうしの構造的なつながり」を探し、ローカルLLM（Ollama）で仮説生成・多分野討論を行うCLIツール。

## セットアップ

Google Colab で動かす場合は `notebooks/nous_colab.ipynb` を開いて上から実行（PC のメモリを使わない）。

### ローカル

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ollama pull gemma4:e2b   # 仮説生成・討論に使用
```

## 使い方

```bash
python nous.py collect             # Wikipedia + arXiv を収集
python nous.py fillvoids           # Void Zone を埋めるブリッジ記事を収集
python nous.py embed               # ChromaDB にベクトル索引を構築
python nous.py query "concept"     # 分野横断検索 [text|graph|both]
python nous.py graph [threshold]   # 知識グラフ + Bridge / Void 分析
python nous.py hyp "concept"       # 仮説生成
python nous.py debate "concept"    # 5 Scepter 討論 + NOUS 統合
python nous.py memory | link | network | sync | debates
```

### 構造抽象化（structural_sim）

```bash
python nous.py abstract [limit]    # 各記事を「分野語を使わない構造記述 + モチーフ」に変換
```

一度作ると `query` / `hyp` / `debate` / `graph` が自動で構造類似度を併用する
（`nous_score = 0.4 × 意味類似 + 0.6 × 構造類似`、構造類似 = 0.5 × 構造記述の埋め込み類似 + 0.5 × モチーフJaccard）。

### 仮説の評価

```bash
python nous.py evaluate            # 5つのScepter視点のLLM審査員がルーブリック(1-5)で採点 + 埋め込み新規性
python nous.py rate <name>         # 人間が同じルーブリックで採点
python nous.py scores              # 総合点ランキング
python nous.py reliability         # 一致度：LLM審査員どうし／人間 × LLM平均（r・ICC・甘さの偏り）
python nous.py evaluate --judges gemma4:e2b,llama3.2:3b,qwen2.5:3b   # 別モデルを審査員に
python nous.py litcheck            # OpenAlex 意味検索で最も近い既存論文を探し lit-novelty を記録
python nous.py calibrate           # 有名な理論10個の lit-novelty を測り「既出」の目安を作る（scores に ⚑）
```

LLM 応答の JSON が壊れていた場合は自動で再試行する。

ルーブリック: novelty / testability / specificity / coherence / structural_depth。

### OpenAlex（26分野・約3億件の学術文献）

```bash
export OPENALEX_API_KEY=...        # 無料キー（https://openalex.org/settings/api）で $1/日
python nous.py openalex [n]        # 各分野の被引用上位 + 直近の論文を収集
python nous.py voids --refresh     # 分野ペアの共起 lift → 文献に基づく Void Zone
python nous.py fillvoids --openalex  # Void ペアを実際に架橋している希少な論文を収集
python nous.py trends              # 各分野の年次成長率と加速度
python nous.py migrate-domains     # 旧ドメイン名（physics 等）を OpenAlex の分野名に統一（1回）
```

一通り実行しても API 費用は数セント程度（list/group_by は 1 回 $0.0001）。

データは `data/`（SQLite・ChromaDB・raw テキスト・グラフHTML）に保存され、Git 管理外。

## 構成

| パス | 役割 |
|---|---|
| `nous/config.py` | パス・既定モデル |
| `nous/llm.py` | Ollama 呼び出しと JSON 抽出 |
| `nous/domains.py` | 収集ドメイン（physics 等）→ Scepter の対応表 |
| `nous/collector/` | Wikipedia / arXiv 収集 |
| `nous/engine/` | 埋め込み・分野横断検索・知識グラフ・仮説生成 |
| `nous/scepter/` | 5 つの分野エージェント（H/S/N/A/I） |
| `nous/engine/structure.py` | 構造抽象化・モチーフ語彙・structural_sim |
| `nous/evaluation/` | ルーブリック・LLM審査・新規性・ICC |
| `nous/collector/openalex.py` | OpenAlex 収集・分野共起・分野トレンド |
| `nous/core.py` | 討論オーケストレーション |
| `nous/memory/` | 仮説の保存・リンク・ChromaDB への還流 |
| `nous/output/` | テキスト / pyvis 可視化 |

## テスト

```bash
pip install pytest networkx numpy
python -m pytest
```
