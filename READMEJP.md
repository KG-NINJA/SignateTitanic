# signate-submitter  
### JSON 1つで Signate 用 submission を自動生成

Autokaggler（Kaggle Titanic 自動化）の発展版として、**Signate コンペ用の提出 CSV を自動生成するエージェント**に拡張しました。  
ローカルにある train/test を読み込み、前処理・学習・CV・提出ファイル出力までを一気通貫で行います。

---

## できること

* データ取得: `dataset_dir` を優先し、なければ同梱の Titanic サンプルにフォールバック
* タスク判定: `problem_type="auto"` で分類/回帰を自動判別
* 前処理:
  * 数値: 中央値補完 + 標準化
  * カテゴリ: 最頻値補完 + One-Hot
* モデルプロファイル
  * `fast`: ロジスティック回帰 / Ridge
  * `power`: ランダムフォレスト
  * `boosting`: HistGradientBoosting
* 5-fold CV でスコアを計算し、`submission.csv` を生成
* 標準出力に JSON を返し、`tags` に必ず `#KGNINJA` を含める

---

## リポジトリ構成

```
src/signate_submitter/    エージェント本体
data/sample/              Titanic 互換のサンプルデータ
tests/                    Pytest
.agent_tmp/, .agent_logs/ 実行時に自動生成
```

---

## 使い方

### 1. インストール

```bash
pip install -e ".[test]"
```

### 2. 実行例（ローカルデータを使用）

```bash
echo '{
  "data_source": "local",
  "dataset_dir": "/path/to/signate/dataset",
  "target_column": "y",
  "id_column": "id",
  "profile": "boosting",
  "submission_name": "submission.csv"
}' | python -m signate_submitter
```

`dataset_dir` が無い場合や `data_source="auto"` で失敗した場合はサンプルデータに自動で切り替わります。

### 3. 出力

* `.agent_tmp/signate/submissions/<ファイル名>.csv`
* `.agent_logs/run-<timestamp>.log`
* 標準出力の JSON（例）
  ```json
  {
    "ok": true,
    "meta": {
      "profile": "boosting",
      "problem_type": "classification",
      "tags": ["#KGNINJA"],
      ...
    },
    "result": {
      "cv_mean": 0.79,
      "submission_path": ".agent_tmp/signate/submissions/submission.csv"
    }
  }
  ```

---

## TaskInput パラメータ

| 項目 | デフォルト | 説明 |
|------|-----------|------|
| `profile` | `"fast"` | `fast` / `power` / `boosting` |
| `data_source` | `"auto"` | `auto`（local→sample）、`local`、`sample` |
| `dataset_dir` | `null` | `train.csv` と `test.csv` を含むディレクトリ |
| `train_filename` | `"train.csv"` | 学習データのファイル名 |
| `test_filename` | `"test.csv"` | テストデータのファイル名 |
| `target_column` | `"Survived"` | 目的変数 |
| `id_column` | `"PassengerId"` | 行ID |
| `submission_target` | `target_column` | 提出カラム名 |
| `drop_columns` | `[]` | モデルから除外する列 |
| `problem_type` | `"auto"` | `classification` / `regression` / `auto` |
| `random_seed` | `42` | 乱数シード |
| `submission_name` | 自動生成 | 出力ファイル名 |
| `notes` | `null` | 任意メモ |

---

## 開発

```bash
pytest
```

`.agent_tmp/` と `.agent_logs/` は毎回自動で作成されます。不要になったら削除してください。

---

## License

MIT License
