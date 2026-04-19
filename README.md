# npb-prediction

NPB（日本プロ野球）の選手成績予測・チーム勝率予測プロジェクト。

## 概要

**3段階の予測パイプライン：**

1. **選手レベル（日本人）** — 各選手の翌年成績（OPS・ERA等）を4層アンサンブルで予測
   - Marcel法（過去3年を5:4:3で加重平均 + 年齢調整）
   - Stan ベイズ補正（K%/BB%/BABIP/年齢のRidge補正）
   - 機械学習（XGBoost / LightGBM）
   - BMA（ベイズモデル平均）で統合 + 80%/95% 信頼区間
2. **選手レベル（外国人）** — 前リーグ成績 × Stan v2モデルで NPB初年度予測（全24選手個別予測）
3. **チームレベル** — モンテカルロ10,000回シミュレーション → P(優勝)/P(CS)/勝利数CI

### Marcel法とは
Tom Tangoが考案した統計的予測手法。直近年ほど重くなる比率（直近年5：2年前4：3年前3）で過去成績を加重平均し、「平均へ引き戻す効果（回帰）」と年齢ピーク調整を加えて翌年成績を推定します。セイバーメトリクスコミュニティで予測精度のベースラインとして使われています。

## Streamlit ダッシュボード（ブラウザで見る）

**https://npb-prediction.streamlit.app/**

日本語/英語対応のインタラクティブダッシュボード。インストール不要でブラウザから全機能を操作できます。

## FastAPI（プログラムから呼び出す）

Raspberry Pi 5 + Docker で常時稼働中（Tailscaleネットワーク内）。

- **トップページ**: 打者 TOP3（wRC+順）+ 先発投手 TOP3（FIP順/投球回100以上）+ リリーフ投手 TOP3（FIP順/投球回20〜99）+ レーダーチャート + 用語説明（入力不要）
- **順位表**: セ・パ両リーグの予測順位（ピタゴラス勝率ベース）+ モンテカルロ優勝確率/CS確率
- **チームシミュレーション**: モンテカルロ10,000回の勝数分布ファンチャート + P(優勝)/P(CS)/P(最下位)確率テーブル
- **打者ランキング**: OPS/AVG/HR/RBI/wOBA/wRC+/ベイズOPS でソート
- **投手ランキング**: ERA/WHIP/SO/W/FIP/K%/BB%/K-BB%/K9/BB9/HR9/ベイズERA でソート
- **チーム勝率**: 12球団の予測勝率 + 信頼区間
- **打者予測**: クイックボタンで選手検索 + レーダーチャート + wOBA/wRC+/wRAA + ベイズOPS（80%/95%CI）
- **投手予測**: クイックボタンで選手検索 + レーダーチャート + FIP/K%/BB%/K-BB% + ベイズERA（80%/95%CI）
- **外国人選手予測**: 前リーグ成績 + Stan v2モデルによるNPB初年度予測（チーム別フィルタ + CI表示）

## ファイル構成

| スクリプト | 内容 |
|---|---|
| `fetch_npb_data.py` | baseball-data.com から NPB成績を取得（2015-2025、打者+投手） |
| `fetch_npb_detailed.py` | npb.jp から詳細打撃成績を取得（2B/3B/SF含む、wOBA算出用） |
| `fetch_rosters.py` | baseball-data.com から年別NPB支配下登録選手一覧を取得（2018-2025） |
| `sabermetrics.py` | wOBA/wRC+/wRAA算出（NPBリーグ環境に合わせた係数） |
| `marcel_projection.py` | Marcel法による翌年成績予測（年齢調整付き） |
| `generate_historical_projections.py` | 過去年（2018-2025）のMarcel→ピタゴラス予測勝利数を生成（選手名鑑フィルタ適用済み） |
| `ml_projection.py` | XGBoost/LightGBM による成績予測（年齢+wOBA/wRC+特徴量付き） |
| `pythagorean.py` | ピタゴラス勝率によるチーム勝率予測（NPB最適指数 k=1.72） |
| `api.py` | FastAPI 推論API（全予測をREST APIで提供） |
| `load_to_bq.py` | BigQuery データロード（25テーブル、型変換・日本語カラム名変換付き） |
| `bqml_train.py` | BigQuery ML 学習・評価・予測ラッパー |
| `bayes_projection.py` | ベイズ予測エンジン（日本人Stan補正 + 外国人Stan v2 + BMA + CI） |
| `team_simulation.py` | モンテカルロ10,000回チーム勝率シミュレーション |
| `sql/` | BQML モデル定義 + 分析ビュー（5ファイル） |
| `DATA_SOURCES.md` | 全データソースの取得方法・URL・クレジット詳細 |
| `Dockerfile` | Docker コンテナ定義（RPi5 対応） |
| `docker-compose.yml` | Docker Compose 設定 |

### データ

| ファイル | 内容 |
|---|---|
| `data/raw/npb_hitters_2015_2025.csv` | 打者成績（2015-2025、11シーズン分、3,780行） |
| `data/raw/npb_pitchers_2015_2025.csv` | 投手成績（2015-2025、11シーズン分、3,773行） |
| `data/raw/npb_standings_2015_2025.csv` | 順位表（12球団×11年=132行） |
| `data/raw/npb_player_birthdays.csv` | 選手生年月日（年齢調整に使用、2,479人） |
| `data/raw/npb_batting_detailed_2015_2025.csv` | 詳細打撃成績（2塁打/3塁打/犠飛を含むwOBA算出用、4,538行） |
| `data/raw/npb_rosters_2018_2025.csv` | 支配下登録名鑑（MLB移籍・退団選手の除外判定に使用、7,866行） |
| `data/bayes/posteriors.json` | Stan事後分布パラメータ（beta/sigma/標準化統計/BMA重み） |
| `data/foreign/foreign_players_master.csv` | 外国人選手マスター（24人、英語名・出身リーグ・Web検証済み） |
| `data/foreign/foreign_prev_stats.csv` | 外国人前リーグ成績（全選手Web検証済み） |
| `data/foreign/conversion_factors.csv` | リーグ別MLB→NPB換算係数 |
| `data/projections/npb_sabermetrics_2015_2025.csv` | wOBA/wRC+/wRAA算出結果 |
| `data/projections/bayes_hitters_2026.csv` | 日本人打者ベイズ予測（463人、CI付き） |
| `data/projections/bayes_pitchers_2026.csv` | 日本人投手ベイズ予測（513人、CI付き） |
| `data/projections/foreign_hitters_2026.csv` | 外国人打者予測（8人、全員Stan v2） |
| `data/projections/foreign_pitchers_2026.csv` | 外国人投手予測（16人、全員Stan v2） |
| `data/projections/team_sim_2026.json` | チームモンテカルロ結果（勝率分布 + 確率） |
| `data/projections/` | その他予測結果CSV（Marcel法・ML・ピタゴラス勝率・セイバー） |

## 予測精度（バックテスト）

### Marcel法 vs XGBoost / LightGBM（2025年バックテスト）

同じ選手セット（打者PA≥100 / 投手IP≥30）で統一して比較。MAE = 実績との平均ズレ幅。数値が小さいほど予測が正確。

**打者 OPS MAE（低いほど良い、n=172）**

| モデル | OPS MAE | 評価 |
|---|---|---|
| Marcel法 | .063 | ★ |
| XGBoost | .063 | ★ |
| LightGBM | .066 | |

**投手 ERA MAE（低いほど良い、n=145）**

| モデル | ERA MAE | 評価 |
|---|---|---|
| **Marcel法** | **0.78** | **★ 優位** |
| XGBoost | 0.93 | |
| LightGBM | 0.92 | |

**結論: 打者OPSはMarcel≒ML（差なし）、投手ERAはMarcel優位（MAE 0.78 vs ML 0.92〜0.93）**

### ピタゴラス勝率（チーム順位予測の誤差）

得失点差から予測勝数を算出。NPBに合わせた指数（k=1.72）はMLB標準（k=1.83）より誤差が小さい。

| 指数 | MAE（平均誤差） | 対象 |
|---|---|---|
| NPB最適 (k=1.72) | **3.20勝** | 全12球団 2015-2025 |
| MLB標準 (k=1.83) | 3.32勝 | 同上 |

### wOBA/wRC+（自前算出）

NPB公式データ（npb.jp）からリーグ環境に合わせた係数でwOBA/wRC+を算出。

- **wOBA**（加重出塁率）: 0.310前後=平均、.380以上=優秀、.450以上=エリート
- **wRC+**（得点創出力+）: 100=リーグ平均、150以上=優秀、200以上=MVP級

| 選手 | 年度 | wOBA | wRC+ |
|---|---|---|---|
| 近藤健介 | 2024 | .479 | 249 |
| オースティン | 2024 | .478 | 248 |
| サンタナ | 2024 | .441 | 220 |

## 2026年予測 TOP5（ベイズ統合予測）

> **注意:** ベイズ統合予測（Marcel法 + Stan/Ridge補正 + 機械学習）による自動計算結果です。NPB公式2026ロースターに在籍する選手のみが対象。MLB移籍選手（村上宗隆・岡本和真・今井達也等）は除外済み。外国人選手24人は前リーグ成績から個別予測。

### 打者（ベイズOPS）
| 選手 | チーム | ベイズOPS | Marcel OPS | 80%CI |
|---|---|---|---|---|
| 近藤健介 | ソフトバンク | .880 | .885 | .691〜1.061 |
| 佐藤輝明 | 阪神 | .832 | .832 | .641〜1.014 |
| レイエス | 日本ハム | .825 | .828 | .636〜1.008 |
| 牧秀悟 | DeNA | .802 | .812 | .607〜.985 |
| 細川成也 | 中日 | .806 | .811 | .614〜.993 |

### 投手（ベイズERA）
| 選手 | チーム | ベイズERA | Marcel ERA | 80%CI |
|---|---|---|---|---|
| 才木浩人 | 阪神 | 2.00 | 1.99 | 0.00〜3.97 |
| マルティネス | 巨人 | 2.01 | 2.01 | — |
| モイネロ | ソフトバンク | 1.99 | 2.01 | 0.00〜3.77 |
| 宮城大弥 | オリックス | 2.28 | 2.29 | 0.29〜4.12 |
| 東克樹 | DeNA | 2.31 | 2.35 | 0.24〜4.25 |

### 2026年チーム予測順位（ベイズ統合 + モンテカルロ10,000回）

> MLB移籍選手（村上宗隆・岡本和真・今井達也・オースティン・バウアー等）はロースターフィルタで除外済み。外国人選手24人は前リーグ成績から個別予測。

**セ・リーグ** — 4チーム団子の混戦

| 順位 | チーム | ベイズ勝数 | Marcel勝数 | 差 | 優勝確率 | CS確率 |
|---|---|---|---|---|---|---|
| 1位 | 阪神 | 71.5 | 80.1 | -8.6 | 26.0% | 65.1% |
| 2位 | 巨人 | 71.1 | 70.7 | +0.4 | 20.2% | 64.4% |
| 3位 | 中日 | 71.0 | 68.8 | +2.2 | 21.2% | 62.6% |
| 4位 | DeNA | 70.7 | 71.3 | -0.6 | 20.2% | 59.7% |
| 5位 | 広島 | 69.1 | 70.4 | -1.3 | 12.3% | 45.5% |
| 6位 | ヤクルト | 61.2 | 64.3 | -3.1 | 0.1% | 2.7% |

**パ・リーグ** — ソフトバンク頭一つ抜け

| 順位 | チーム | ベイズ勝数 | Marcel勝数 | 差 | 優勝確率 | CS確率 |
|---|---|---|---|---|---|---|
| 1位 | ソフトバンク | 81.3 | 80.5 | +0.8 | 47.9% | 91.3% |
| 2位 | 日本ハム | 79.1 | 76.8 | +2.3 | 27.2% | 83.5% |
| 3位 | オリックス | 77.5 | 73.8 | +3.7 | 17.6% | 71.6% |
| 4位 | 西武 | 74.9 | 68.6 | +6.3 | 7.1% | 48.4% |
| 5位 | 楽天 | 66.7 | 65.5 | +1.2 | 0.1% | 3.1% |
| 6位 | ロッテ | 64.9 | 67.1 | -2.2 | 0.1% | 2.1% |

> Marcel法のみ → ベイズ統合で変わった主なポイント:
> - **阪神 -8.6勝**: K%/BB%のスキル補正で80.1→71.5勝に
> - **ヤクルト -3.1勝**: 村上宗隆MLB移籍の影響
> - **西武 +6.3勝**: 外国人選手の個別予測（前リーグ成績反映）
> - **セ・リーグが混戦化**: 阪神独走→4チーム70.7-71.5勝で横一線

## MLOps 構成

「予測スクリプトを作る」から「予測システムを運用する」への3本柱。

| 仕組み | 実装 | 効果 |
|---|---|---|
| **モデル保存** | `joblib` → `data/models/*.pkl` | 年度ごとのモデルを永続化。過去のモデルで再予測可能 |
| **精度記録** | `data/metrics/metrics_{year}.json` | Marcel vs ML のMAE推移を記録。「今年は改善したか」が分かる |
| **実験管理** | Weights & Biases (`npb-prediction` プロジェクト) | 毎年の学習ごとにMAE・特徴量重要度・Marcel比改善率を自動記録 |
| **自動実行** | GitHub Actions（毎年11月1日 + 手動） | データ取得→学習→保存→W&B記録→Gitコミットが全自動 |

### CI/CD パイプライン（`annual_update.yml`）

```
Step 1: fetch_npb_data.py       → 打者/投手成績スクレイプ
Step 2: fetch_npb_detailed.py   → 詳細打撃成績（wOBA算出用）
Step 3: pythagorean.py          → 順位表・ピタゴラス勝率
Step 4: sabermetrics.py         → wOBA/wRC+/wRAA算出
Step 5: marcel_projection.py    → Marcel法予測
Step 6: ml_projection.py        → ML予測 + モデル保存 + メトリクスJSON出力
Step 7: git commit & push       → data/ を自動コミット
Step 8: bayes_projection.py     → ベイズ予測（日本人BMA + 外国人Stan v2 + CI）
Step 9: team_simulation.py      → モンテカルロ10,000回チームシミュレーション
Step 10: load_to_bq.py          → BigQuery に全データロード（33テーブル）
Step 11: bqml_train.py          → BQML モデル学習・評価・予測
```

スケジュール: 毎年3月1日（FA・移籍確定後、開幕前）+ `workflow_dispatch`（手動実行）

> **⚠️ 手動実行時の注意**: `workflow_dispatch` で実行する場合は、`data_end_year` に前年（例: `2025`）を明示的に指定してください。空白のままだと当年（`2026`）が設定され、存在しないシーズンデータを取得しようとして予測値が崩壊します。

**CI タイムアウト保護**: ワークフローに `timeout-minutes: 180` を設定。全 ML/ベイズ/シミュレーションスクリプトに `PYTHONUNBUFFERED=1` + ステップ別経過時間ログを追加し、ハング時の原因特定を容易にしている。

### `/metrics` エンドポイント

年次再学習のたびに記録される MAE 推移を返します。

```bash
curl http://localhost:8000/metrics
```

```json
{
  "件数": 1,
  "メトリクス": [{
    "year": 2026,
    "data_end_year": 2025,
    "generated_at": "2026-11-01T09:30:00",
    "hitter": {"lgb": 0.066, "xgb": 0.063, "ensemble": 0.064, "marcel": 0.063},
    "pitcher": {"lgb": 0.92, "xgb": 0.93, "ensemble": 0.92, "marcel": 0.78}
  }]
}
```

`hitter.lgb < hitter.marcel` ならMLがMarcelを上回っている。そうでなければMarcelを採用。

---

## 使い方

```bash
# データ取得（baseball-data.com から）
python fetch_npb_data.py

# 詳細打撃成績取得（npb.jp から）
python fetch_npb_detailed.py

# wOBA/wRC+算出
python sabermetrics.py

# Marcel法で予測
python marcel_projection.py

# XGBoost/LightGBM で予測（wOBA/wRC+特徴量含む）
python ml_projection.py

# ベイズ統合予測（日本人BMA + 外国人Stan v2 + CI）
python bayes_projection.py

# モンテカルロ・チームシミュレーション（10,000回）
python team_simulation.py

# ピタゴラス勝率で予測
python pythagorean.py
```

### API起動

```bash
# ローカル起動
pip install -r requirements.txt
uvicorn api:app --reload

# Docker起動
docker compose up --build
```

APIが起動したら http://localhost:8000/docs でSwagger UIを確認できます。

### APIエンドポイント

| メソッド | パス | 内容 |
|---|---|---|
| GET | `/predict/hitter/{name}` | 打者の翌年成績予測（Marcel + ML + ベイズOPS/CI） |
| GET | `/predict/pitcher/{name}` | 投手の翌年成績予測（Marcel + ML + ベイズERA/CI） |
| GET | `/predict/foreign/{name}` | 外国人選手のNPB初年度予測（Stan v2 + CI） |
| GET | `/predict/team/{name}?year=2024` | チームのピタゴラス勝率 |
| GET | `/standings/simulation` | モンテカルロ順位シミュレーション（P(優勝)/P(CS)/勝数CI） |
| GET | `/sabermetrics/{name}?year=2024` | wOBA/wRC+/wRAA |
| GET | `/rankings/hitters?top=10&sort_by=OPS` | 打者ランキング |
| GET | `/rankings/pitchers?top=10&sort_by=ERA` | 投手ランキング |
| GET | `/pythagorean?year=2024` | 全チームのピタゴラス勝率 |
| GET | `/metrics` | 年次MAE推移（Marcel vs ML） |

### レスポンス例

```bash
# 牧秀悟の翌年予測
curl http://localhost:8000/predict/hitter/牧
```

```json
{
  "query": "牧",
  "count": 1,
  "predictions": [{
    "player": "牧 秀悟",
    "team": "DeNA",
    "marcel": {"OPS": 0.812, "AVG": 0.279, "HR": 20.3, "RBI": 66.2},
    "ml": {"pred_OPS": 0.830},
    "bayes": {"OPS": 0.802, "OPS_80ci": [0.607, 0.985], "method": "bma_jpn"}
  }]
}
```

## 実装済み機能

- [x] Marcel法（年齢調整付き）
- [x] ピタゴラス勝率（NPB最適指数 k=1.72）
- [x] XGBoost / LightGBM（年齢+wOBA/wRC+特徴量付き）
- [x] セイバー指標追加（wOBA/wRC+/wRAA自前算出）
- [x] FastAPI による推論API化 + Docker対応
- [x] Streamlit ダッシュボード（日本語/英語対応）
- [x] 2026ロースター反映（移籍・退団）
- [x] トップページに用語説明（OPS/wOBA/wRC+/FIP/K%/BB%/K-BB%/K9/BB9/HR9）を追加
- [x] 計算対象外選手（新人・新外国人）の可視化（バッジ表示・選手リスト）
- [x] 投手TOP3に総合投球力レーダーチャート追加（防御率/WHIP/奪三振/投球回/勝利の5軸）
- [x] 打者ランキングにwOBA/wRC+ソート追加
- [x] 投手ランキングにFIP/K%/BB%/K-BB%/K9/BB9/HR9ソート追加
- [x] 打者予測にwOBA/wRC+/wRAA + wRC+推移グラフ統合
- [x] 投手予測にFIP/K%/BB%/K-BB%/K9/BB9/HR9 + レーダーチャート追加
- [x] 全指標に計算式expander（式 + 基準値の解説）
- [x] NPBデータ年数バッジ表示（1年/2年の選手は予測精度に注意書き）
- [x] 打者レーダーチャート6軸化（HR/AVG/OBP/SLG/wOBA/wRC+）
- [x] 投手レーダーチャート7軸化（ERA/WHIP/奪三振/K9/BB9/HR9/FIP）
- [x] K/9・BB/9・HR/9にリーグ平均との差分表示追加
- [x] 打者TOP3のソートをwRC+順に変更（OPSから変更）
- [x] 投手TOP3を先発（投球回100以上）・リリーフ（20〜99）に分離しFIP順で表示
- [x] 棒グラフとレーダーチャートの指標を統一（打者: wOBA/wRC+、投手: FIP/K9/BB9/HR9）
- [x] Streamlitダッシュボード日本語/英語切替（サイドバートグル）
- [x] モバイル対応（レスポンシブCSS・縦積み・横スクロール・横向き棒グラフ）
- [x] CI/CD自動再学習（GitHub Actions 毎年3月1日）
- [x] モデルアーティファクト保存（`data/models/*.pkl`）
- [x] 精度メトリクス記録・API公開（`/metrics`）
- [x] W&B実験管理（毎年の学習ごとにMAE・Marcel比改善率を自動記録）
- [x] 歴史的バックテストのチーム割り当てバグ修正（FA・移籍選手が前年チームに計上されていた問題）
- [x] 選手名鑑フィルタ追加（退団・MLB移籍・引退選手を予測から除外、`npb_rosters_2018_2025.csv` 参照）
- [x] GCP分析基盤（BigQuery 25テーブル + BQML 4モデル + 10分析ビュー）
- [x] BigQuery ML でSQLだけで打者OPS/投手ERA予測（Boosted Tree + 線形回帰）
- [x] データ品質チェックビュー（欠損検知・カバレッジ確認）
- [x] GCP Deploy ワークフロー（BigQuery + BQML）
- [x] ベイズ統合予測エンジン（Marcel + Stan Ridge + ML → BMA、80%/95% CI付き）
- [x] 外国人選手Stan v2モデル（前リーグ成績ベース、全24選手Web検証済み個別予測）
- [x] モンテカルロ・チームシミュレーション（10,000回、ピタゴラス k=1.83、パークファクター補正）
- [x] API: ベイズ予測エンドポイント + 外国人予測 + 順位シミュレーション（v0.5.0）
- [x] Streamlit: ベイズCI表示（打者OPS/投手ERA）+ チームシミュレーションページ + 外国人選手ページ
- [x] BigQuery: ベイズ予測・外国人・シミュレーション等8テーブル追加（計33テーブル）

## 予測精度（ベイズ統合の効果）

### 選手レベル: Marcel法 vs ベイズ補正（8年間バックテスト 2018-2025）

| 指標 | n | Marcel MAE | ベイズ MAE | 改善確率 |
|---|---|---|---|---|
| 打者 wOBA | 2,208 | 0.05023 | **0.04980** | 97.1% |
| 投手 ERA | 2,164 | 1.23008 | **1.22241** | 97.1% |

改善幅は小さいが、8年間を通じて97%の確率でMarcel法を上回る安定した改善。

### チームレベル: Marcel法の順位予測精度（2018-2025実績）

| 指標 | 値 |
|---|---|
| 勝数のMAE | **6.4勝** |
| 順位ズレ平均 | 1.42位 |
| 順位ピタリ率 | 18% |
| 1位ズレ以内率 | 65% |

## ベイズ統合の技術詳細

[npb-bayes-projection](https://github.com/yasumorishima/npb-bayes-projection) で検証済みのStan階層ベイズモデルを統合済み（7フェーズ中6フェーズ完了）。

### 予測アーキテクチャ — 4層階層

```
Layer 1: Marcel法（基盤 — 3年加重平均 + 年齢調整）
Layer 2: Stan/Ridge補正（K%/BB%/BABIP/年齢で微調整）
Layer 3: ML（XGBoost/LightGBM）
Layer 4: BMA統合（Marcel 35% + Stan 40% + ML 25%）→ 80%/95% CI付き
  ↓
外国人選手: 前リーグ成績 × Stan v2モデル → 個別NPB予測
  ↓
モンテカルロ10,000回 → チーム勝数分布 → P(優勝)/P(CS)
```

### 設計上の判断
- **Stanはランタイムで動かさない** — posteriors.jsonに事後分布パラメータを保存、推論時はNumPyサンプリングのみ（RPi5 4GB RAM対応）
- **外国人選手は全員Web検証済み** — 24人の英語名・出身リーグ・前リーグ成績を個別に確認
- **MLB移籍選手はロースターフィルタで除外** — roster_current.pyで2026公式ロースターに照合
- **不確実性がエンドツーエンドで伝播** — 選手CI → チームMonte Carlo → 優勝確率

### 実装フェーズ

| Phase | 内容 | 状態 |
|---|---|---|
| 1. 基盤 | 日本人ベイズ推論 | **完了** |
| 2. 外国人 | Stan v2予測（24選手個別） | **完了** |
| 3. チーム | モンテカルロ + パークファクター | **完了** |
| 4. 学習 | Stan学習パイプライン自動化 | 未着手（来シーズン向け） |
| 5. API | ベイズ/外国人/シミュレーション | **完了** |
| 6. Streamlit | CI表示 + 新2ページ | **完了** |
| 7. BigQuery | 8テーブル追加（計33） | **完了** |

## 今後の予定

### その他

- [ ] Marcel重みをNPBデータ最適化値に更新（打者 8/4/3・REG_PA=2000 / 投手 4/5/2・REG_IP=800、ブートストラップ p=0.003 で有意）→ [npb-marcel-weight-study](https://github.com/yasumorishima/npb-marcel-weight-study)
- [x] BQML精度検証 — BQML（BT: OPS MAE 0.0642 / ERA MAE 0.909）は Python ML（OPS ~0.065 / ERA 0.92-0.93）と同等以上。特徴量もBQMLの方が豊富（パークファクター・DIPS・FIP近似・Marcel加重平均等）。両手法とも投手ERAではMarcel法（0.78）に及ばず、これはML共通の課題
- [ ] 精度が悪化したときの自動アラート

### モデルの限界

- **NPB在籍1〜2年の選手**: データが少ないため予測値がリーグ平均に強く引き戻される（1年のみ→約2/3がリーグ平均寄り）。選手名横の「NPB1年/2年」バッジで確認可能
- **若手の急成長**: 年齢調整は+0.3%/年と小さく、ブレイクするような急成長は捉えられない
- **球場改修**: パークファクターは5年移動平均を使用。2026年のバンテリンドーム改修等は未反映
- **データの見落とし**: 個人プロジェクトのため品質保証には限界がある（例: MLB移籍選手がフィルタされていないバグがあり、途中で修正した）

## データソース

**このプロジェクトのNPBデータは以下のサイトから取得しています：**

- [プロ野球データFreak](https://baseball-data.com) — NPB選手成績・生年月日（2015-2025）
- [日本野球機構 NPB](https://npb.jp) — 詳細打撃成績（2B/3B/SF、wOBA/wRC+算出用）

詳細は [DATA_SOURCES.md](DATA_SOURCES.md) を参照。

### ~~GCP 分析基盤（BigQuery + BigQuery ML）~~（2026-04-19 退役）

> BigQuery `npb` データセットと BQML モデル（`bqml_batter_ops` / `bqml_pitcher_era` 等）は Grafana 公開廃止に伴い削除。予測ロジックは Python (`scripts/`) に一本化済み、データは RPi5 Parquet (`/mnt/ssd/npb_shared/`) にバックアップ保持。Streamlit / FastAPI は CSV (`data/projections/`) を直接読むため BQ 非依存で継続稼働。

## License

MIT
