# npb-prediction

NPB（日本プロ野球）の選手成績予測・チーム勝率予測プロジェクト。

## 概要

Marcel法・機械学習（XGBoost/LightGBM）・ピタゴラス勝率を用いて、NPB選手の翌年成績とチーム勝率を予測します。

### Marcel法とは
Tom Tangoが考案した成績予測手法。過去3年の成績を **5/4/3** で加重平均し、リーグ平均への回帰と年齢調整を行います。シンプルながら驚くほど強いベースラインとして知られています。

## ファイル構成

| スクリプト | 内容 |
|---|---|
| `fetch_npb_data.py` | baseball-data.com から NPB成績を取得（2015-2025、打者+投手） |
| `fetch_npb_detailed.py` | npb.jp から詳細打撃成績を取得（2B/3B/SF含む、wOBA算出用） |
| `sabermetrics.py` | wOBA/wRC+/wRAA算出（NPBリーグ環境に合わせた係数） |
| `marcel_projection.py` | Marcel法による翌年成績予測（年齢調整付き） |
| `ml_projection.py` | XGBoost/LightGBM による成績予測（年齢+wOBA/wRC+特徴量付き） |
| `pythagorean.py` | ピタゴラス勝率によるチーム勝率予測（NPB最適指数 k=1.72） |
| `DATA_SOURCES.md` | 全データソースの取得方法・URL・クレジット詳細 |

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

## 予測精度（2024年バックテスト）

### Marcel法（年齢調整後）

| 指標 | MAE | RMSE | 対象 |
|---|---|---|---|
| OPS | .055 | .068 | 規定打席以上 |
| 防御率 (ERA) | 0.62 | 0.81 | 100IP以上 |

### XGBoost / LightGBM

| 指標 | MAE | RMSE | 対象 |
|---|---|---|---|
| OPS | .077 | — | 規定打席以上 |
| 防御率 (ERA) | 0.95 | — | 100IP以上 |

### ピタゴラス勝率

| 指数 | MAE | RMSE | 対象 |
|---|---|---|---|
| NPB最適 (k=1.72) | 3.20勝 | 3.92 | 全12球団（2015-2025） |
| MLB標準 (k=1.83) | 3.32勝 | 4.07 | 同上 |

> **Marcel法がMLを上回る** — 先行研究通り、Marcel法は「驚くほど強いベースライン」であり、NPBでも同様の結果が確認されました。wOBA/wRC+を特徴量に追加した後も傾向は変わらず。年齢調整はERA改善に寄与しましたが、OPSへの影響は限定的でした。

### wOBA/wRC+（自前算出）

NPB公式データ（npb.jp）からリーグ環境に合わせた係数でwOBA/wRC+を算出。

| 選手 | 年度 | wOBA | wRC+ |
|---|---|---|---|
| 近藤健介 | 2024 | .479 | 249 |
| オースティン | 2024 | .478 | 248 |
| サンタナ | 2024 | .441 | 220 |

## 2025年予測 TOP5（Marcel法）

### 打者（OPS）
| 選手 | チーム | OPS | HR | RBI |
|---|---|---|---|---|
| 村上宗隆 | ヤクルト | .899 | 34 | 91 |
| 近藤健介 | ソフトバンク | .899 | 17 | 65 |
| 吉田正尚 | オリックス | .854 | 15 | 69 |
| 岡本和真 | 巨人 | .848 | 29 | 81 |
| 牧秀悟 | DeNA | .825 | 23 | 81 |

### 投手（ERA）
| 選手 | チーム | ERA | WHIP | SO |
|---|---|---|---|---|
| 山本由伸 | オリックス | 1.93 | 1.00 | 169 |
| マルティネス | 中日 | 2.01 | 1.01 | 50 |
| モイネロ | ソフトバンク | 2.12 | 1.00 | 89 |
| 才木浩人 | 阪神 | 2.14 | 1.10 | 101 |
| 髙橋宏斗 | 中日 | 2.27 | 1.11 | 128 |

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

## 今後の予定

- [x] Marcel法（年齢調整付き）
- [x] ピタゴラス勝率（NPB最適指数 k=1.72）
- [x] XGBoost / LightGBM（年齢+wOBA/wRC+特徴量付き）
- [x] セイバー指標追加（wOBA/wRC+/wRAA自前算出）
- [ ] FastAPI による推論API化
- [ ] Computer Vision（骨格検知）

## データソース

**このプロジェクトのNPBデータは以下のサイトから取得しています：**

- [プロ野球データFreak](https://baseball-data.com) — NPB選手成績・生年月日（2015-2025）
- [日本野球機構 NPB](https://npb.jp) — 詳細打撃成績（2B/3B/SF、wOBA/wRC+算出用）

詳細は [DATA_SOURCES.md](DATA_SOURCES.md) を参照。データの利用にあたり、提供元への敬意を忘れずに。

## License

MIT
