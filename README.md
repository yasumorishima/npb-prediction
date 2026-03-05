# npb-prediction

NPB（日本プロ野球）の選手成績予測・チーム勝率予測プロジェクト。

## 概要

**2段階の予測パイプライン：**

1. **選手レベル** — 各選手の翌年成績（OPS・ERA等）を3手法で予測
   - Marcel法（過去3年を5:4:3で加重平均 + 年齢調整）
   - 機械学習（XGBoost / LightGBM）
2. **チームレベル** — 選手成績の積み上げからピタゴラス勝率でチーム順位を算出

### Marcel法とは
Tom Tangoが考案した統計的予測手法。直近年ほど重くなる比率（直近年5：2年前4：3年前3）で過去成績を加重平均し、「平均へ引き戻す効果（回帰）」と年齢ピーク調整を加えて翌年成績を推定します。MLBで広く使われるベースライン手法です。

## Streamlit ダッシュボード（ブラウザで見る）

**https://npb-prediction.streamlit.app/**

日本語/英語対応のインタラクティブダッシュボード。インストール不要でブラウザから全機能を操作できます。

## FastAPI（プログラムから呼び出す）

**https://raspberrypi.tailb303d6.ts.net/docs**

Raspberry Pi 5 + Docker で常時稼働中。Swagger UIから各APIエンドポイントをブラウザで試せます（止まっている場合はご容赦ください）。

- **トップページ**: 打者 TOP3（wRC+順）+ 先発投手 TOP3（FIP順/投球回100以上）+ リリーフ投手 TOP3（FIP順/投球回20〜99）+ レーダーチャート + 用語説明（入力不要）
- **順位表**: セ・パ両リーグの予測順位（ピタゴラス勝率ベース）
- **打者ランキング**: OPS/AVG/HR/RBI/wOBA/wRC+ でソート
- **投手ランキング**: ERA/WHIP/SO/W/FIP/K%/BB%/K-BB%/K9/BB9/HR9 でソート
- **チーム勝率**: 12球団の予測勝率 + 信頼区間
- **打者予測**: クイックボタンで選手検索 + レーダーチャート + wOBA/wRC+/wRAA + 計算式解説
- **投手予測**: クイックボタンで選手検索 + レーダーチャート + FIP/K%/BB%/K-BB% + 計算式解説

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
| `DATA_SOURCES.md` | 全データソースの取得方法・URL・クレジット詳細 |
| `Dockerfile` | Docker コンテナ定義 |
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
| `data/projections/npb_sabermetrics_2015_2025.csv` | wOBA/wRC+/wRAA算出結果 |
| `data/projections/` | 予測結果CSV（Marcel法・ML・ピタゴラス勝率・セイバー） |

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

## 2026年予測 TOP5（Marcel法）

> **注意:** Marcel法は過去3年間（2023-2025）のNPBデータに基づく純粋な統計予測です。NPB公式2026ロースター（roster_2026.py）に在籍確認された選手のみを対象としています。MLB移籍済み選手・退団選手は除外済みです。

### 打者（OPS）
| 選手 | チーム | OPS | HR | RBI |
|---|---|---|---|---|
| 近藤健介 | ソフトバンク | .885 | 15 | 59 |
| 佐藤輝明 | 阪神 | .832 | 25 | 82 |
| レイエス | 日本ハム | .828 | 24 | 70 |
| 牧秀悟 | DeNA | .812 | 20 | 66 |
| 細川成也 | 中日 | .811 | 20 | 63 |

### 投手（ERA）
| 選手 | チーム | ERA | WHIP | SO |
|---|---|---|---|---|
| 才木浩人 | 阪神 | 1.99 | 1.07 | 123 |
| マルティネス | 巨人 | 2.01 | 1.02 | 54 |
| モイネロ | ソフトバンク | 2.01 | 1.01 | 125 |
| 宮城大弥 | オリックス | 2.29 | 1.01 | 139 |
| 東克樹 | DeNA | 2.35 | 1.12 | 107 |

### 2026年チーム予測順位（NPB 144試合制）

**セ・リーグ**
| 順位 | チーム | 予測勝数 |
|---|---|---|
| 1位 | 阪神 | 80.1 |
| 2位 | DeNA | 71.3 |
| 3位 | 巨人 | 70.7 |
| 4位 | 広島 | 70.4 |
| 5位 | 中日 | 68.8 |
| 6位 | ヤクルト | 64.3 |

**パ・リーグ**
| 順位 | チーム | 予測勝数 |
|---|---|---|
| 1位 | ソフトバンク | 80.5 |
| 2位 | 日本ハム | 76.8 |
| 3位 | オリックス | 73.8 |
| 4位 | 西武 | 68.6 |
| 5位 | ロッテ | 67.1 |
| 6位 | 楽天 | 65.5 |

> 予測フロー: 選手のwOBAを集計 → チーム得点を推定（wRAA） → 得失点差からピタゴラス勝率（k=1.72）で勝数を算出

## MLOps 構成

「予測スクリプトを作る」から「予測システムを運用する」への3本柱。

| 仕組み | 実装 | 効果 |
|---|---|---|
| **モデル保存** | `joblib` → `data/models/*.pkl` | 年度ごとのモデルを永続化。過去のモデルで再予測可能 |
| **精度記録** | `data/metrics/metrics_{year}.json` | Marcel vs ML のMAE推移を記録。「今年は改善したか」が分かる |
| **自動実行** | GitHub Actions（毎年11月1日 + 手動） | データ取得→学習→保存→Gitコミットが全自動 |

### CI/CD パイプライン（`annual_update.yml`）

```
Step 1: fetch_npb_data.py       → 打者/投手成績スクレイプ
Step 2: fetch_npb_detailed.py   → 詳細打撃成績（wOBA算出用）
Step 3: pythagorean.py          → 順位表・ピタゴラス勝率
Step 4: sabermetrics.py         → wOBA/wRC+/wRAA算出
Step 5: marcel_projection.py    → Marcel法予測
Step 6: ml_projection.py        → ML予測 + モデル保存 + メトリクスJSON出力
Step 7: git commit & push       → data/ を自動コミット（models/*.pkl / metrics/*.json 含む）
```

スケジュール: 毎年3月1日（FA・移籍確定後、開幕前）+ `workflow_dispatch`（手動実行）

> **⚠️ 手動実行時の注意**: `workflow_dispatch` で実行する場合は、`data_end_year` に前年（例: `2025`）を明示的に指定してください。空白のままだと当年（`2026`）が設定され、存在しないシーズンデータを取得しようとして予測値が崩壊します。

### `/metrics` エンドポイント

年次再学習のたびに記録される MAE 推移を返します。

```bash
curl https://raspberrypi.tailb303d6.ts.net/metrics
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
| GET | `/predict/hitter/{name}` | 打者の翌年成績予測（Marcel + ML） |
| GET | `/predict/pitcher/{name}` | 投手の翌年成績予測（Marcel + ML） |
| GET | `/predict/team/{name}?year=2024` | チームのピタゴラス勝率 |
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
    "marcel": {"OPS": 0.834, "AVG": 0.295, "HR": 22.9, "RBI": 81.4},
    "ml": {"pred_OPS": 0.874}
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
- [x] 歴史的バックテストのチーム割り当てバグ修正（FA・移籍選手が前年チームに計上されていた問題）
- [x] 選手名鑑フィルタ追加（退団・MLB移籍・引退選手を予測から除外、`npb_rosters_2018_2025.csv` 参照）
## 今後の予定

- [ ] 精度が悪化したときの自動アラート
- [ ] 精度改善（特徴量追加・アンサンブル等）

> **外国人選手のベイズ推定・Monte Carloシミュレーションは [npb-prediction-bayes](https://github.com/yasumorishima/npb-prediction-bayes) に分離しました。**

### Marcel法のもう一つの限界: NPB在籍1〜2年の選手

NPBデータが1〜2年しかない選手（2年目外国人・復帰直後の選手など）は、Marcel法のリーグ平均への回帰が強く働きます。

- データ**1年のみ**（例: 60IPの外国人投手）→ 予測値の約**2/3**がリーグ平均に引き寄せられる
- データ**2年のみ** → 予測値の約**半分**がリーグ平均に引き寄せられる

「計算対象外」ではなく予測値は出ますが、実力の過小/過大評価が起きやすいです。Streamlitアプリでは選手名横に **「NPB1年」「NPB2年」** バッジを表示しています。

### Marcel法のもう一つの限界: 若手の急成長

Marcel法の年齢調整は **+0.3%/年（27歳ピーク基準）** と非常に小さく、急激な成長には追いつきません。

23〜26歳の選手が「殻を破る」ような場合、モデルは過去3年の平均に引き戻すため、実際の成績を大幅に下回る予測になります。新外国人・新人と同様に、**若手ブレイク候補が多いチームほどモデルの予測は保守的に出る**傾向があります。

## ブログ記事

- [Last Place to Champions: What Marcel Projection Reveals About 2021 NPB Yakult and Orix (DEV.to)](https://dev.to/yasumorishima/last-place-to-champions-what-marcel-projection-reveals-about-2021-npb-yakult-and-orix-2eee)
- [Marcel予測と実際の勝利数の乖離から見る2021年ヤクルト・オリックスの優勝要因（Zenn）](https://zenn.dev/shogaku/articles/npb-marcel-lastplace-to-champion)

## データソース

**このプロジェクトのNPBデータは以下のサイトから取得しています：**

- [プロ野球データFreak](https://baseball-data.com) — NPB選手成績・生年月日（2015-2025）
- [日本野球機構 NPB](https://npb.jp) — 詳細打撃成績（2B/3B/SF、wOBA/wRC+算出用）

詳細は [DATA_SOURCES.md](DATA_SOURCES.md) を参照。

## License

MIT
