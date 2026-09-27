# NoUs

分野横断の知識ネットワークから「遠い分野どうしの構造的なつながり」を探し、ローカルLLM（Ollama）で仮説生成・多分野討論を行うCLIツール。

## セットアップ

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
| `nous/core.py` | 討論オーケストレーション |
| `nous/memory/` | 仮説の保存・リンク・ChromaDB への還流 |
| `nous/output/` | テキスト / pyvis 可視化 |

## テスト

```bash
pip install pytest networkx numpy
python -m pytest
```
