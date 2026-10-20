# NoUs

分野横断の知識ネットワークから「遠い分野どうしの構造的なつながり」を探し、ローカルLLM（Ollama）で仮説生成・多分野討論を行うCLIツール。

## セットアップ

Google Colab で動かす場合は [`notebooks/nous_colab.ipynb`](https://colab.research.google.com/github/kohga1220-stack/NoUs/blob/main/notebooks/nous_colab.ipynb) を開き、「ランタイム → すべてのセルを実行」（PC のメモリを使わない）。「1. ワンセット準備」が環境をそろえ、「2. 実行フォーム」で、やること（状態確認・自律ループ・採用の再点検・浄土へ書き出す・モデル比較など）を選んで ▶ を押す。結果は画面と Drive の `浄土/実行ログ/` に残る（コードは `nous/colab_run.py`）。手動・個別の操作は `nous_colab_manual.ipynb`。空の `nous.db` で Drive を上書きしないよう、データ読み込みと `save()` に保護が入っている（`nous/colab_setup.py`）。

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

### モデルの指定（Ollama 以外も使う）

`--gen-model` や `--judges` などのモデル名に、接頭辞を付けると呼び出し先を選べる。接頭辞なしは従来どおりローカル Ollama。

| 指定 | 呼び出し先 | 必要な環境変数 |
|---|---|---|
| `gemma4:e2b`（接頭辞なし）/ `ollama:gemma4:e2b` | ローカル Ollama | なし |
| `opencode:<ID>` | OpenCode Console。**使わない方針**（下の注意を参照） | `OPENCODE_API_KEY` |
| `gemini:gemini-3.6-flash` | Gemini API | `GEMINI_API_KEY` |

```bash
python nous.py evaluate --judges gemma4:e2b,gemini:gemini-3.6-flash
python nous.py reliability         # 審査員どうし・人間との一致度で比べる
```

- **OpenCode は使わない方針**。`-free` のモデルは API から使えない（実キーで 403 `FreeTierError` を確認、2026-10-17）。`-free` なしの有料 ID の動作は未確認。Gemini は実キーで動作を確認済み。
- 外部 API に送るのは、プロンプトに含まれる記事・仮説の本文（公開情報）だけ。無料枠のモデルには、入力が学習に使われるものがあるため、使う前に、そのサービスのデータの扱いを確認する。
- Colab では、「ワンセット準備」セルが、シークレットの `OPENCODE_API_KEY` と `GEMINI_API_KEY` を環境変数へ読み込む（未登録なら「未設定」と表示）。`MODELS` に `opencode:...` や `gemini:...` を書いても、`ollama pull` はされない。
- 429（レート制限）・5xx・接続エラーは、5・10・20・40 秒の間隔で最大4回再試行する。
- Gemini の応答に候補や本文がない場合（プロンプトのブロックなど）は、理由（`blockReason` / `finishReason`）を含む `EmptyResponseError` にして、再試行しない。

### 仮説の評価

```bash
python nous.py evaluate            # 5つのScepter視点のLLM審査員がルーブリック(1-5)で採点 + 埋め込み新規性
python nous.py rate <name>         # 人間が同じルーブリックで採点
python nous.py scores              # 総合点ランキング
python nous.py reliability         # 一致度：LLM審査員どうし／人間 × LLM平均（r・ICC・甘さの偏り）
python nous.py evaluate --judges gemma4:e2b,llama3.2:3b,qwen2.5:3b   # 別モデルを審査員に
python nous.py check-llm gemini:gemini-3.6-flash  # 各モデルに極小リクエストを1回送り、キー・モデルIDが使えるか確認（空応答・失敗は NG、1件でも NG なら終了コード1。エラー文中のキーは *** に伏せる。Colab のワンセット準備の最後でも MODELS に対して自動実行）
python nous.py litcheck            # OpenAlex 意味検索で最も近い既存論文を探し lit-novelty を記録
python nous.py calibrate           # 有名な理論10個の lit-novelty を測り「既出」の目安を作る（scores に ⚑）
python nous.py combocheck          # 仮説を2〜4概念に分解し、概念ペアの共起論文数と隣接性で「未踏の組み合わせ」を探す（★=橋渡し候補）
python nous.py combocheck --report # 保存済みの判定を一覧（API 呼び出しなし）。★は隣接性順に並べ、言い換えに強い穴・litcheck との照合・共起論文の題名を表示
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

### 自律ループ（フェーズ4）

```bash
python nous.py autoloop --plan            # 次に探索する分野ペアと指標を表示（API 呼び出しなし）。--min-shared N
python nous.py autoloop 1 --gen-model M   # 仮説の生成だけ別の（大きい）Ollamaモデルで行う（評価・概念分解は既定モデルのまま）
python nous.py autoloop 3                 # 3サイクル：Voidペア → 橋渡し論文収集 → 仮説 → 評価 → 文献チェック → 採否
python nous.py autoloop 3 --fast          # 先に文献・組合せチェックをし、棄却が決まった仮説は評価（judge）を省く（T4 で1サイクルの約半分が judge）。composite が付かない仮説が増える
python nous.py vault                      # 浄土（Obsidian の vault）へ書き出す：採用も棄却も理由つきで残す。docs/JODO.md
python nous.py vault --promote hyp:31     # 人の判断で善見天へ昇格
python nous.py autoloop --report          # 実行履歴
python nous.py autoloop --retract hyp:27 [理由]   # 採用したあとに既知の内容だと分かった仮説を取り消す（記録は残し、知識ベースから外し、以後の同期にも戻さない）
python nous.py knowncheck --controls            # 有名な10例を、近い論文を読んだ LLM（既定 Gemini）が「既知」と判定できるか確かめる
python nous.py knowncheck --controls --show             # 診断: 取得した論文の題名と、「関連」止まりの理由（why）を表示。既定は近い論文10件＋言い換え検索（--n N で件数、--no-rewrite で言い換えなし）
python nous.py knowncheck --controls --strict           # 従来の厳しい基準（主張の全体が述べられている場合のみ「既知」）。既定は「中心の関係が述べられていれば既知」で、引用は2文まで
python nous.py knowncheck --negatives                  # 逆の確認: 「既知にならないはずの」6件（作り話の組み合わせ、有名な原理を新しい対象に当てはめたもの）を、誤って既知と判定しないか
python nous.py knowncheck --controls --model gemma4:e2b     # Gemini の代わりに Colab 内のローカルモデル（無料）で判定。精度を --controls / --negatives で測ってから、autoloop --known-model gemma4:e2b に切り替える
python nous.py knowncheck hyp:26 hyp:20        # 指定した仮説の主張を、近い論文がすでに述べていないか確認（引用文を要旨と機械的に照合）
python nous.py knowncheck --accepted --retract # 採用済みのうち「既知」と確認できたものを取り消す
python -m nous.evaluation.survey [T]   # 棄却ルール案（最も珍しいペア／どれか1ペア／最も近い論文の類似度 T 以上）を、保存済みデータで比べる（API 呼び出しなし）
```

1サイクル: **直接の共起（lift）は低いのに、他の分野との繋がり方が似ている**（隣同士の隙間）未試行の分野ペアを選ぶ
（共有論文が `--min-shared`＝既定1,000本未満のペア、および lift が低い側の半分に入らないペアは除外。過去の報酬による優先度の上乗せは小さい）→ 両分野を架橋する論文を収集・埋め込み →
**その橋渡し論文を材料に**仮説を1つ生成 → LLM 評価 + `litcheck` + `combocheck`（仮説の主張に含まれる概念で判定）→
**combocheck が `studied`、lit-novelty が較正済みの「既出」目安以下、近い論文がその主張をすでに述べていると確認できた（known-work check：Colab に `gemma4:12b` があればそれを使い（無料）、無ければ `GEMINI_API_KEY` の Gemini。どちらも無ければ無効。`--known-model M` で指定、`none` で無効）、または新規性を除く rubric 複合点が保存済み仮説の中央値未満なら棄却**、それ以外は採用して知識ベースへ還流。
報酬 = 採用なら（新規性を除く rubric 複合点）/5、棄却なら 0（LLM の新規性採点は人間評価より甘く、相関もなかったため報酬に使わない）。
サイクルの最後に仮説の本文・分解した概念・共起論文の題名・概念3つ以上の全体の共起数（参考表示。判定には使わない）・litcheck で見つかった最も近い既存論文（題名・年・類似度）を表示する。`debate` は同一の問いの繰り返しを拒否する（`--force` で実行）。
「採用」は「これらのチェックでは既存文献に見つからなかった」の意味で、新発見の主張ではない。

一通り実行しても API 費用は数セント程度（list/group_by は 1 回 $0.0001）。

データは `data/`（SQLite・ChromaDB・raw テキスト・グラフHTML）に保存され、Git 管理外。

## 構成

構成図（Mermaid）は [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

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
