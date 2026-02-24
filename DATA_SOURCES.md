# NPB予測モデル データソースと取得方法

## データソース一覧

### 1. baseball-data.com（プロ野球データFreak）

**用途**: 基本打撃・投手成績、順位表、選手プロフィール（生年月日）

#### 打者・投手成績
- **スクリプト**: `fetch_npb_data.py`
- **取得方法**: `pandas.read_html()` でHTMLテーブルを直接パース
- **URL**:
  - 打者: `https://baseball-data.com/{yy}/stats/hitter-all/tpa-1.html`
  - 投手: `https://baseball-data.com/{yy}/stats/pitcher-all/era-1.html`
  - `{yy}` = 年度の下2桁（例: 2024→"24"）。当年は `{yy}/` なしで直接アクセス
- **出力**:
  - `data/raw/npb_hitters_2015_2025.csv`（3,780行）
  - `data/raw/npb_pitchers_2015_2025.csv`（3,773行）
- **カラム（打者）**: player, team, AVG, G, PA, AB, H, HR, RBI, SB, BB, HBP, SO, SH, GDP, OBP, SLG, OPS, RC27, XR27, year
- **カラム（投手）**: player, team, ERA, G, W, L, SV, HLD, WPCT, BF, IP, HA, HRA, BB, HBP, SO, R, ER, WHIP, DIPS, year
- **注意**: **二塁打(2B)・三塁打(3B)・犠飛(SF)はこのページにない** → NPB公式で取得

#### 順位表（チーム得点・失点）
- **スクリプト**: `pythagorean.py` 内の取得関数
- **URL**: `https://baseball-data.com/{yy}/stats/team-rankings/`
- **出力**: `data/raw/npb_standings_2015_2025.csv`（132チーム-年）

#### 選手プロフィール（生年月日）
- **スクリプト**: `fetch_npb_data.py` 内の `fetch_player_profiles()` 関数（またはmarcel_projection.pyの生年月日取得部分）
- **URL**: `https://baseball-data.com/{yy}/player/{team_code}/`
  - チームコード: t(阪神), yb(DeNA), g(巨人), d(中日), c(広島), s(ヤクルト), h(ソフトバンク), f(日本ハム), bs(オリックス), e(楽天), l(西武), m(ロッテ)
- **出力**: `data/raw/npb_player_birthdays.csv`（2,479人）

### 2. npb.jp（NPB公式サイト）

**用途**: 詳細打撃成績（二塁打・三塁打・犠飛・故意四球など、baseball-data.comにないカラム）

- **スクリプト**: `fetch_npb_detailed.py`
- **取得方法**: `pandas.read_html()` でHTMLテーブルをパース
- **URL**: `https://npb.jp/bis/{year}/stats/idb1_{team_code}.html`
  - `{year}` = 4桁の年度（例: 2024）
  - `{team_code}` = NPB公式のチームコード（下記参照）
  - **チーム別の個別ページ**（リーグ一括ページは存在しない）
- **チームコード**:
  - セ・リーグ: c(広島), d(中日), db(DeNA), g(巨人), s(ヤクルト), t(阪神)
  - パ・リーグ: b(オリックス), e(楽天), f(日本ハム), h(ソフトバンク), l(西武), m(ロッテ)
  - **注意**: オリックスは **2015-2017は `bs`**、**2018以降は `b`**（球団略称変更に伴うコード変更と推定）
- **カラム（24列）**: flag, player, G, PA, AB, R, H, 2B, 3B, HR, TB, RBI, SB, CS, SH, SF, BB, IBB, HBP, SO, GDP, AVG, SLG, OBP
  - flag列: 左打マーク（*）等 → 取得後に削除
  - **2025年のみ23列**（flag列なし） → スクリプトで両方に対応済み
- **出力**: `data/raw/npb_batting_detailed_2015_2025.csv`（4,538行、2015-2024）
- **レート制限**: npb.jpは連続アクセスでConnectionReset（レート制限）が発生する
  - `time.sleep(0.5)` を各リクエスト間に挟む
  - 全12チーム×11年度 = 132リクエスト → 約2.5分で完了
  - レート制限に当たった場合は数分待ってから再実行

### 3. wOBA/wRC+の算出

- **スクリプト**: `sabermetrics.py`
- **入力**: `data/raw/npb_batting_detailed_2015_2025.csv`（NPB公式データ）
- **出力**: `data/projections/npb_sabermetrics_2015_2025.csv`
- **算出方法**:
  1. MLB標準の得点価値係数（BB=0.69, HBP=0.72, 1B=0.88, 2B=1.27, 3B=1.62, HR=2.10）を基準に
  2. NPBリーグ環境に合わせてスケーリング（リーグOBP / 生wOBAでwOBAスケールを算出）
  3. 年度ごとにリーグ平均wOBA、リーグR/PAを算出し、wRC+を計算
  4. パークファクター補正は省略（NPB公開データがないため）

## 取得実行の手順

```bash
cd C:/Users/fw_ya/Desktop/Claude_code/npb-prediction

# 1. 基本成績取得（baseball-data.com）
PYTHONUTF8=1 python fetch_npb_data.py

# 2. 詳細打撃成績取得（npb.jp）
PYTHONUTF8=1 python fetch_npb_detailed.py

# 3. wOBA/wRC+算出
PYTHONUTF8=1 python sabermetrics.py
```

## クレジット表記

### 必須
- **プロ野球データFreak (baseball-data.com)**: 打者・投手成績、順位表、選手プロフィール
- **NPB公式サイト (npb.jp)**: 詳細打撃成績（二塁打、三塁打、犠飛等）

### 表記例
```
Data sources:
- プロ野球データFreak (https://baseball-data.com)
- 日本野球機構 NPB (https://npb.jp)
```

## 既知の制限

1. **オリックス2015-2017**: チームコード `bs` で再取得が必要（初回実行時は `b` で404エラー）
2. **2025年**: NPB公式のテーブル構造が23列（flag列なし）に変更 → 対応済みだがレート制限で未取得
3. **パークファクター**: NPB公式が公開していないため、wRC+にパークファクター補正を適用できない
4. **FanGraphs NPB**: CSVエクスポートは有料会員（FanGraphs+）限定 → 利用不可
5. **npb_scraping（Pythonライブラリ）**: Baseball-ReferenceがCloudflare 403 → 使用不可
6. **ProEyeKyuu**: CSVダウンロード可能だがJS動的レンダリングで自動取得困難 → 今回は不使用
