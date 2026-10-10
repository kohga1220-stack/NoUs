# NoUs アーキテクチャ

コード（`main`）を読んで書いた構成図。図は GitHub 上でそのまま表示される（Mermaid）。
実装の詳細は各ファイル先頭の docstring、運用の手順は `README.md` を参照。

## 1. 全体の流れ

```mermaid
flowchart LR
    subgraph SRC["データ源"]
        W["Wikipedia"]
        A["arXiv"]
        O["OpenAlex<br/>26分野・約3億件"]
    end
    subgraph STORE["保存 data/"]
        DB[("SQLite nous.db")]
        CH[("ChromaDB<br/>nous_knowledge<br/>nous_structures")]
    end
    subgraph ENGINE["engine/"]
        EMB["embedder<br/>all-MiniLM-L6-v2"]
        STR["structure<br/>構造抽象化 + モチーフ"]
        CON["connector<br/>nous_score"]
        KG["knowledge_graph<br/>Bridge / Void"]
        HYP["hypothesis<br/>仮説生成"]
    end
    subgraph AGENTS["scepter/ と core.py"]
        SC["5 Scepter<br/>H S N A I"]
        NOUS["NOUS 統合<br/>討論と判定"]
    end
    subgraph EVAL["evaluation/"]
        RUB["rubric<br/>LLM審査"]
        LIT["literature<br/>litcheck"]
        COMBO["combination<br/>combocheck"]
    end
    LOOP["loop.py<br/>autoloop"]
    LLM["Ollama<br/>gemma4"]

    W --> DB
    A --> DB
    O --> DB
    DB --> EMB --> CH
    CH --> STR --> CH
    CH --> CON --> KG
    CON --> HYP
    CON --> SC --> NOUS
    HYP --> DB
    NOUS --> DB
    DB --> RUB
    RUB --> DB
    DB --> LIT
    DB --> COMBO
    LIT -->|"OpenAlex検索"| O
    COMBO -->|"件数・分布"| O
    LOOP --> HYP
    LOOP --> RUB
    LOOP --> LIT
    LOOP --> COMBO
    LOOP -->|"採用分のみ還流"| CH
    LLM -.-> STR
    LLM -.-> HYP
    LLM -.-> SC
    LLM -.-> RUB
    LLM -.-> COMBO
```

## 2. コマンドと担当モジュール

```mermaid
flowchart TB
    CLI["nous.py"]
    CLI --> C1["collect / fillvoids"]
    CLI --> C2["embed / query / graph"]
    CLI --> C3["hyp / debate / memory / link / network / sync"]
    CLI --> C4["abstract"]
    CLI --> C5["evaluate / rate / scores / reliability"]
    CLI --> C6["litcheck / calibrate / combocheck"]
    CLI --> C7["openalex / voids / trends / migrate-domains"]
    CLI --> C8["autoloop"]

    C1 --> M1["collector/*"]
    C2 --> M2["engine/embedder, connector, knowledge_graph"]
    C3 --> M3["engine/hypothesis, core, memory/*"]
    C4 --> M4["engine/structure"]
    C5 --> M5["evaluation/runner, rubric, store, reliability"]
    C6 --> M6["evaluation/literature, combination"]
    C7 --> M7["collector/openalex, migrate"]
    C8 --> M8["loop"]
```

## 3. 知識ベースと接続（`nous_score`）

```mermaid
flowchart LR
    Q["クエリ"] --> E["埋め込み"]
    E --> S1["意味類似"]
    ST["構造類似<br/>構造記述の埋め込み 0.5<br/>モチーフ Jaccard 0.5"]
    S1 --> MIX{"構造抽象化<br/>済みか"}
    ST --> MIX
    MIX -->|"はい"| B["nous_score =<br/>0.4 意味 + 0.6 構造<br/>同分野ペナルティ"]
    MIX -->|"いいえ"| B2["nous_score =<br/>意味 同分野ペナルティ"]
    B --> R["分野横断の候補"]
    B2 --> R
```

## 4. 評価パイプライン

```mermaid
flowchart TB
    T["対象<br/>hyp / scepter / verdict"] --> J["LLM審査員<br/>rubric 5項目 1-5"]
    T --> N["埋め込み新規性"]
    T --> L["litcheck<br/>最近傍論文との距離<br/>lit-novelty"]
    T --> CB["combocheck<br/>概念ペアの共起と隣接性"]
    H["人間評価 rate"] --> EV
    J --> EV[("evaluations")]
    N --> EV
    L --> EV
    CB --> EV
    EV --> AG["aggregate<br/>composite と 新規性を除く composite"]
    EV --> REL["reliability<br/>ICC 人間 対 LLM"]
    CAL["calibrate<br/>既知の理論10件"] --> CUT["既出の目安 q75"]
    CUT --> L
```

rubric の項目: novelty / testability / specificity / coherence / structural_depth。
LLM の novelty は人間評価より甘く相関もなかったため（n=10）、`loop.py` の報酬には使わない。

## 5. combocheck

```mermaid
flowchart TB
    H["仮説"] --> CL["claim_text<br/>Hypothesis H: 以降"]
    CL --> D["LLMで標準的な概念 2〜4個に分解"]
    D --> P["全ての概念ペア A, B"]
    P --> J1["joint: A AND B の論文数<br/>フレーズ一致"]
    P --> LS["loose: 単語を語順不問で AND"]
    P --> AD["adjacency<br/>サブフィールド分布の重なり"]
    J1 --> ST{"joint"}
    ST -->|"10本以上"| S1["studied 研究済み"]
    ST -->|"1〜9本"| S2["few_papers"]
    ST -->|"0本"| S3["unexplored<br/>両概念が500件以上のときだけ有意味"]
    LS --> RB["loose が10未満なら<br/>robust 言い換えに強い穴"]
    AD --> BR["未踏か少数で<br/>adjacency が境界値以上なら ★"]
    REF["参照ペア<br/>有名4組 無関係3組"] --> TH["adjacency の境界値"]
    TH --> BR
```

## 6. 自律ループ `autoloop`

```mermaid
flowchart TB
    S["choose_pair<br/>lift が低い側の半分<br/>共有論文が1,000本以上<br/>neighbour が高い順<br/>過去の報酬は小さく加点"]
    S --> C1["collect_pair_bridges<br/>両分野の被引用上位論文"]
    C1 --> C2["embed_all"]
    C2 --> C3["generate_hypothesis<br/>橋渡し論文から1つ<br/>テンプレートと一般論は禁止"]
    C3 --> C4["judge<br/>rubric 新規性を除く composite"]
    C4 --> C5["check<br/>litcheck と combocheck"]
    C5 --> D{"decide"}
    D -->|"studied"| X["棄却"]
    D -->|"lit-novelty が既出の目安以下"| X
    D -->|"composite が保存済みの中央値未満"| X
    D -->|"それ以外"| OK["採用 sync で知識ベースへ"]
    X --> LG[("loop_runs")]
    OK --> LG
    LG -->|"報酬 採用なら composite/5"| S
```

「採用」は「これらのチェックでは既存文献に見つからなかった」の意味で、新発見の主張ではない。

## 7. データベース

```mermaid
erDiagram
    articles {
        int id PK
        text title
        text domain
        text summary
    }
    explorations ||--o{ hypotheses : "1:n"
    debates ||--o{ scepter_hypotheses : "1:n"
    debates ||--o{ scepter_critiques : "1:n"
    hypotheses ||--o{ hypothesis_links : "n:n"
    evaluations {
        text target_ref
        text rater
        text item
        real score
    }
    combo_checks {
        text target_ref
        text concept_a
        text concept_b
        int joint
        int loose
        real adjacency
    }
    combo_checks ||--o{ combo_titles : "概念ペア"
    literature_checks {
        text target_ref
        text title
        real sim
    }
    loop_runs {
        text field_a
        text field_b
        text hyp_ref
        text status
        real reward
    }
    oa_fields ||--o{ field_links : "分野ペア"
    oa_fields ||--o{ field_trends : "年別"
```

他に `structures`（構造記述）、`literature_controls`（較正用の既知理論）、`combo_meta`（隣接性の境界値）がある。
`target_ref` は `hyp:<id>` / `scepter:<id>` / `verdict:<id>` の形。

## 8. Colab での運用

```mermaid
flowchart LR
    NB["nous_colab.ipynb<br/>ワンセット準備"] --> M["Drive マウント"]
    NB --> G["git clone main"]
    NB --> P["pip install"]
    NB --> OL["Ollama 導入とモデル取得"]
    NB --> DT["colab_setup.prepare_data<br/>ローカルに本物のDBがあれば上書きしない"]
    DT --> RUN["各コマンドを実行"]
    RUN --> SV["save<br/>nous.db が1MB未満なら保存を拒否"]
    SV --> DR[("Drive NoUs_data")]
    DR --> DT
```

## 9. 既知の限界

- 文献チェックは、題名と要旨の語の一致と、埋め込みの距離に基づく。「未踏」は「検索で見つからなかった」の意味で、価値の保証ではない。
- 仮説の生成と評価に使う LLM は小さいモデル（既定は `gemma4:e2b`）。LLM 審査員は同一モデルの視点違いで独立ではない。
- 知識ベースは約570記事と小さく、分野ごとの偏りがある。
