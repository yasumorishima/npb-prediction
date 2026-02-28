# npb-prediction

NPB（日本プロ野球）の選手成績予測・チーム勝率予測プロジェクト。

## 概要

Marcel法・機械学習（XGBoost/LightGBM）・ピタゴラス勝率を用いて、NPB選手の翌年成績とチーム勝率を予測します。

### Marcel法とは
Tom Tangoが考案した成績予測手法。過去3年の成績を **5/4/3** で加重平均し、リーグ平均への回帰と年齢調整を行います。シンプルながら驚くほど強いベースラインとして知られています。

## Streamlit ダッシュボード

**https://npb-prediction.streamlit.app/**

日本語/英語対応のインタラクティブダッシュボード。ブラウザから全機能を操作できます。

- **トップページ**: TOP3打者/投手 + レーダーチャート（入力不要）
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
| `sabermetrics.py` | wOBA/wRC+/wRAA算出（NPBリーグ環境に合わせた係数） |
| `marcel_projection.py` | Marcel法による翌年成績予測（年齢調整付き） |
| `ml_projection.py` | XGBoost/LightGBM による成績予測（年齢+wOBA/wRC+特徴量付き） |
| `pythagorean.py` | ピタゴラス勝率によるチーム勝率予測（NPB最適指数 k=1.72） |
| `api.py` | FastAPI 推論API（全予測をREST APIで提供） |
| `DATA_SOURCES.md` | 全データソースの取得方法・URL・クレジット詳細 |
| `Dockerfile` | Docker コンテナ定義 |
| `docker-compose.yml` | Docker Compose 設定 |

### データ

| ファイル | 内容 |
|---|---|
| `data/raw/npb_hitters_2015_2025.csv` | 打者成績 3,780行 |
| `data/raw/npb_pitchers_2015_2025.csv` | 投手成績 3,773行 |
| `data/raw/npb_standings_2015_2025.csv` | 順位表 132行（12球団×11年） |
| `data/raw/npb_player_birthdays.csv` | 生年月日 2,479人 |
| `data/raw/npb_batting_detailed_2015_2025.csv` | 詳細打撃成績 4,538行（npb.jp、2B/3B/SF含む） |
| `data/projections/npb_sabermetrics_2015_2025.csv` | wOBA/wRC+/wRAA算出結果 |
| `data/projections/` | 予測結果CSV（Marcel法・ML・ピタゴラス勝率・セイバー） |

## 予測精度（バックテスト）

### Marcel法（年齢調整後）

| 年度 | OPS MAE | OPS RMSE | ERA MAE | ERA RMSE | 打者n | 投手n |
|---|---|---|---|---|---|---|
| 2024 | .055 | .069 | 0.62 | 0.80 | 33 | 24 |
| 2025 | .048 | .057 | 0.63 | 0.78 | 19 | 29 |

### XGBoost / LightGBM（2025年バックテスト）

| モデル | OPS MAE | ERA MAE |
|---|---|---|
| LightGBM | .065 | 0.92 |
| XGBoost | .062 | 0.92 |
| Marcel法 | .063 | 0.78 |

### ピタゴラス勝率

| 指数 | MAE | RMSE | 対象 |
|---|---|---|---|
| NPB最適 (k=1.72) | 3.20勝 | 3.92 | 全12球団（2015-2025） |
| MLB標準 (k=1.83) | 3.32勝 | 4.05 | 同上 |

> **Marcel法がMLを上回る** — 先行研究通り、Marcel法は「驚くほど強いベースライン」であり、NPBでも2年分のバックテストで一貫して確認されました。wOBA/wRC+を特徴量に追加した後も傾向は変わらず。

### wOBA/wRC+（自前算出）

NPB公式データ（npb.jp）からリーグ環境に合わせた係数でwOBA/wRC+を算出。

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

### 2026年チーム予測順位

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

> 予測手法: wOBA回帰（R²=0.9954）→ wRAA → チーム得点推定 → ピタゴラス勝率（k=1.72）

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

## 今後の予定

- [x] Marcel法（年齢調整付き）
- [x] ピタゴラス勝率（NPB最適指数 k=1.72）
- [x] XGBoost / LightGBM（年齢+wOBA/wRC+特徴量付き）
- [x] セイバー指標追加（wOBA/wRC+/wRAA自前算出）
- [x] FastAPI による推論API化 + Docker対応
- [x] Streamlit ダッシュボード（日本語/英語対応）
- [x] 2026ロースター反映（移籍・退団）
- [x] 計算対象外選手（新人・新外国人）の可視化（バッジ表示・選手リスト）
- [x] 投手TOP3に総合投球力レーダーチャート追加（防御率/WHIP/奪三振/投球回/勝利の5軸）
- [x] 予測の不確実性を「幅」で表示（計算外選手が多いチームほど信頼区間を広げる）
- [x] 打者ランキングにwOBA/wRC+ソート追加
- [x] 投手ランキングにFIP/K%/BB%/K-BB%/K9/BB9/HR9ソート追加
- [x] 打者予測にwOBA/wRC+/wRAA + wRC+推移グラフ統合
- [x] 投手予測にFIP/K%/BB%/K-BB%/K9/BB9/HR9 + レーダーチャート追加
- [x] 全指標に計算式expander（式 + 基準値の解説）
- [x] NPBデータ年数バッジ表示（1年/2年の選手は予測精度に注意書き）
- [x] 打者レーダーチャート6軸化（HR/AVG/OBP/SLG/wOBA/wRC+）
- [x] 投手レーダーチャート6軸化（ERA/WHIP/奪三振/K9/BB9/HR9）
- [x] K/9・BB/9・HR/9にリーグ平均との差分表示追加
- [ ] 計算対象外選手への初期値定義（歴代NPB外国人初年度実績・出身リーグ変換係数）
- [ ] 精度改善（特徴量追加・アンサンブル等）

### 予測幅（信頼区間）の考え方

計算対象外選手（新外国人・新人）はMarcel法の予測対象外のため、モデルは **wRAA=0（リーグ平均と同等の貢献）** として計算しています。

しかし実際の初年度成績には大きなばらつきがあります。

| 仮定 | 根拠 |
|---|---|
| 計算外選手1人あたりの不確実性 = **±1.5勝** | NPB外国人選手の初年度wRAAは概ね -15点〜+25点のばらつき、10点≒1勝の経験則を適用 |

```
予測勝数の下限 = 予測勝数 − 計算外人数 × 1.5
予測勝数の上限 = 予測勝数 + 計算外人数 × 1.5
```

計算外選手が多いチームほど予測幅が広く、「下位予測でも新戦力次第で大きく変わりうる」という情報を可視化しています。

### Marcel法のもう一つの限界: NPB在籍1〜2年の選手

NPBデータが1〜2年しかない選手（2年目外国人・復帰直後の選手など）は、Marcel法のリーグ平均への回帰が強く働きます。

- データ**1年のみ**（例: 60IPの外国人投手）→ 予測値の約**2/3**がリーグ平均に引き寄せられる
- データ**2年のみ** → 予測値の約**半分**がリーグ平均に引き寄せられる

「計算対象外」ではなく予測値は出ますが、実力の過小/過大評価が起きやすいです。Streamlitアプリでは選手名横に **「NPB1年」「NPB2年」** バッジを表示しています。

### Marcel法のもう一つの限界: 若手の急成長

Marcel法の年齢調整は **+0.3%/年（27歳ピーク基準）** と非常に小さく、急激な成長には追いつきません。

23〜26歳の選手が「殻を破る」ような場合、モデルは過去3年の平均に引き戻すため、実際の成績を大幅に下回る予測になります。新外国人・新人と同様に、**若手ブレイク候補が多いチームほどモデルの予測は保守的に出る**傾向があります。

## データソース

**このプロジェクトのNPBデータは以下のサイトから取得しています：**

- [プロ野球データFreak](https://baseball-data.com) — NPB選手成績・生年月日（2015-2025）
- [日本野球機構 NPB](https://npb.jp) — 詳細打撃成績（2B/3B/SF、wOBA/wRC+算出用）

詳細は [DATA_SOURCES.md](DATA_SOURCES.md) を参照。

## License

MIT
