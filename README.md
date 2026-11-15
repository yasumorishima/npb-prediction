# npb-prediction

NPB（日本プロ野球）の選手成績予測・チーム勝率予測プロジェクト。

**2026 年の採点（開幕前に凍結した予測を実際のシーズンで採点）**：チーム勝数の MAE 5.833 勝は定数・前年の床には勝ったがピタゴラスの床 5.818 に負け、80% / 95% 区間に入ったのは 12 球団中 6 / 9 球団＝どちらも区間が狭すぎた（[判定](data/backtest/2026_team_result.md)）。選手単位は床には勝ったが Marcel 単独との差はゼロと区別できなかった（2026-09-21 時点の暫定・下の「予測精度」）。

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

ローカル / Docker で起動可能（RPi5 での常時稼働は現在停止中）。

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
| `bayes_projection.py` | ベイズ予測エンジン（日本人Stan補正 + 外国人Stan v2 + BMA + CI） |
| `team_simulation.py` | モンテカルロ10,000回チーム勝率シミュレーション |
| `DATA_SOURCES.md` | 全データソースの取得方法・URL・クレジット詳細 |
| `Dockerfile` | Docker コンテナ定義（RPi5 対応） |
| `docker-compose.yml` | Docker Compose 設定 |

### データ

モデル出力・集計・チーム順位表の正本は repo の `data/`（git 管理）。baseball-data.com / npb.jp から取得した選手単位の成績表（下表の 🔒）は両サイトに再配布の許可が無いため、非公開リポジトリで保管し再配布しない（このリポジトリには置かず、HF への同期からも除外する）。クラウドミラーとして HF Dataset [npb-stats](https://huggingface.co/datasets/yasumorishima/npb-stats) に `data/` ツリー（🔒 を除く）を自動同期（BQ 時代の `npb/` スナップショットも保持。ただし選手単位の成績表＝`npb/raw_*`〈チーム順位表を除く〉・`npb/sabermetrics.parquet`・`npb_bayes/npb_sabermetrics.parquet` は HF から削除済み）。

| ファイル | 内容 |
|---|---|
| `data/raw/npb_hitters_2015_2025.csv` | 打者成績（2015-2025、11シーズン分、3,780行）。🔒 非公開リポジトリで保管（再配布しない） |
| `data/raw/npb_pitchers_2015_2025.csv` | 投手成績（2015-2025、11シーズン分、3,773行）。🔒 非公開リポジトリで保管（再配布しない） |
| `data/raw/npb_standings_2015_2025.csv` | 順位表（12球団×11年=132行） |
| `data/raw/npb_player_birthdays.csv` | 選手生年月日（年齢調整に使用、2,479人）。🔒 非公開リポジトリで保管（再配布しない） |
| `data/raw/npb_batting_detailed_2015_2025.csv` | 詳細打撃成績（2塁打/3塁打/犠飛を含むwOBA算出用、4,538行）。🔒 非公開リポジトリで保管（再配布しない） |
| `data/raw/npb_rosters_2018_2025.csv` | 支配下登録名鑑（MLB移籍・退団選手の除外判定に使用、7,866行）。🔒 非公開リポジトリで保管（再配布しない） |
| `data/bayes/posteriors.json` | Stan事後分布パラメータ（beta/sigma/標準化統計/BMA重み） |
| `data/foreign/foreign_players_master.csv` | 外国人選手マスター（24人、英語名・出身リーグ・Web検証済み） |
| `data/foreign/foreign_prev_stats.csv` | 外国人前リーグ成績（全選手Web検証済み） |
| `data/foreign/conversion_factors.csv` | リーグ別MLB→NPB換算係数 |
| `data/projections/npb_sabermetrics_2015_2025.csv` | wOBA/wRC+/wRAA算出結果（選手ごとの成績行を含む）。🔒 非公開リポジトリで保管（再配布しない）。アプリは集計値だけの `data/projections/league_constants.json` を読む |
| `data/projections/bayes_hitters_2026.csv` | 日本人打者ベイズ予測（463人、CI付き） |
| `data/projections/bayes_pitchers_2026.csv` | 日本人投手ベイズ予測（513人、CI付き） |
| `data/projections/foreign_hitters_2026.csv` | 外国人打者予測（8人、全員Stan v2） |
| `data/projections/foreign_pitchers_2026.csv` | 外国人投手予測（16人、全員Stan v2） |
| `data/projections/team_sim_2026.json` | チームモンテカルロ結果（勝率分布 + 確率） |
| `data/projections/` | その他予測結果CSV（Marcel法・ML・ピタゴラス勝率・セイバー） |

## 予測精度（ベイズ統合の効果）

### 選手レベル: Marcel法 vs ベイズ補正（8年間バックテスト 2018-2025）

| 指標 | n | Marcel MAE | ベイズ MAE | 改善確率 |
|---|---|---|---|---|
| 打者 wOBA | 2,208 | 0.05023 | **0.04980** | 97.1% |
| 投手 ERA | 2,164 | 1.23008 | **1.22241** | 97.1% |

改善幅は小さいが、8年間を通じて97%の確率でMarcel法を上回る安定した改善。

⚠️ **2026-10-08 追記**：上の 97% は 2018-2025 のバックテストの数字。**開幕前に凍結した 2026 予測を実際のシーズンで採点すると（2026-09-21 時点の暫定・シーズン 95% 消化）、Marcel との差はゼロと区別できなかった**（MAE の差 bayes−marcel：打者 OPS +0.00095 ± 0.00086・投手 ERA −0.0244 ± 0.0148）。定数・前年の床には勝っている（打者 OPS MAE 0.0720 対 0.0849 / 0.0792、投手 ERA MAE 0.8182 対 0.8518 / 0.9973）。シーズン終了後に最終値で測り直す。

### チームレベル: Marcel法の順位予測精度（2018-2025実績）

| 指標 | 値 |
|---|---|
| 勝数のMAE | **6.4勝** |
| 順位ズレ平均 | 1.42位 |
| 順位ピタリ率 | 18% |
| 1位ズレ以内率 | 65% |

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

### 信頼区間の幅（2026-09-21 改訂）

区間の σ は 2018-2025 の全出場機会水準で fit した残差（`sigma_residual`）を全選手に同じ値で当てていた。出場機会で絞った選手にはこれが広すぎる。実測で確かめた結果:

| 確認 | 結果 |
|---|---|
| 実測残差 sd ÷ 当てていた σ（2018-2025 の 8 シーズン） | 打者 **0.531〜0.698** / 投手 **0.561〜0.701**（**8 年すべて 1 未満**）。σ は `posteriors.json` の `sigma_residual`（打者は **wOBA 尺度 0.06215 ×2.33 = 0.14481 OPS**・投手は ERA 尺度 1.66495） |
| 出場機会が増えると残差 sd は下がるか | 8 年すべてで Yes（例 2018 投手 IP 30-44 で 1.42 → 95-168 で 0.62） |

そこで σ を出場機会の関数にした（`posteriors.json` の `sigma_model`。式 `sigma_base * exp(gamma * z_log_pt)` は外国人経路が元から使っていた形）。**2018-2024 だけで fit し、2025 と 2026 は fit に一切使っていない**。入力は 3 月時点で分かる Marcel 予測 PA / IP。

**実成果物を測った被覆率**（`predict_*` を回し直して 2026 シーズンと突き合わせ。名目 0.80 / 0.95）。**採点したのは「区間を持つ行」かつ「実績 PA≥100 / IP≥30」の行だけ**＝この母集団の外にも区間は出しているが、そこは測っていない:

⚠️ **2026 は暫定**＝2026-09-21 時点で 143 試合中 129〜136 消化（95%）でシーズンが終わっていない。最終値ではない。
⚠️ **2026 の行はこのリポジトリの中身だけでは再計算できない**＝採点に使った 2026 の実績はシーズン途中なので commit していない（`data/raw/` の選手成績表は 2025 年までで、非公開リポジトリで保管し再配布しない＝このリポジトリには無い。`tools/fit_sigma_model.py` は `--current-*` を渡さないと `HELD-OUT 2026: skipped` と出る）。2025 の行はその成績表を `data/raw/` に置けば `python tools/fit_sigma_model.py` で再現できる。2026 はシーズン終了後に実績を取り込んで測り直す。

| | 旧（平の σ） | 新 |
|---|---|---|
| 打者 2026（暫定・131 行） | 0.954 / 0.992 | **0.840 / 0.947** |
| 投手 2026（暫定・121 行） | 0.950 / 0.992 | **0.851 / 0.975** |

2025 は予測 CSV が残っていないので実成果物では測れない。Marcel 残差から区間を組み直した代理では 打者 0.988/1.000 → **0.888/0.976**、投手 0.993/1.000 → **0.861/0.986**。**上の表とは別の測り方なので並べて比べない。**

区間の作り方で直した不具合 3 件と、残る割り切り・注意は [docs/technical-details.md](docs/technical-details.md) に移した。

再現用: `tools/measure_residual_sd.py`（8 シーズンの残差 sd）と `tools/fit_sigma_model.py`（係数の fit と held-out 検証）。検算は `tests/test_sigma_model.py`。

⚠️ `data/projections/*_2026.csv` は **2026-03-23 に凍結した当時のまま**（区間も旧 σ のまま）。実際に行われたシーズンに対する採点の記録なので再生成していない。次回 `annual_update.yml` を回したときに新しい σ で生成される。

出典: baseball-data.com

### ピタゴラス勝率（チーム順位予測の誤差）

得失点差から予測勝数を算出。NPBに合わせた指数（k=1.72）はMLB標準（k=1.83）より誤差が小さい。

⚠️ ただし**チームシミュレーション（`team_simulation.py`）は k=1.83 を使っている**。2026 のチーム予測はこの値で凍結し、[事前登録](data/backtest/2026_team_prereg.md)の判定に使うので、判定が終わるまで変えない。

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
| **自動実行** | GitHub Actions（毎年3月1日 + 手動） | データ取得→学習→保存→W&B記録→Gitコミット→HF同期が全自動 |

### CI/CD パイプライン（`annual_update.yml`）

```
Step 1: fetch_npb_data.py       → 打者/投手成績スクレイプ
Step 2: fetch_npb_detailed.py   → 詳細打撃成績（wOBA算出用）
Step 3: fetch_rosters.py        → 支配下登録名鑑取得（FA・移籍確定後）
Step 4: pythagorean.py          → 順位表・ピタゴラス勝率
Step 5: sabermetrics.py         → wOBA/wRC+/wRAA算出
Step 6: marcel_projection.py    → Marcel法予測
Step 7: ml_projection.py        → ML予測 + モデル保存 + メトリクスJSON出力
Step 8: bayes_projection.py     → ベイズ予測（日本人BMA + 外国人Stan v2 + CI）
Step 9: team_simulation.py      → モンテカルロ10,000回チームシミュレーション
git commit & push               → data/ を自動コミット（選手単位の成績表は除く＝非公開リポジトリへ push）
HF 同期                          → data/（選手単位の成績表を除く）を HF Dataset (npb-stats) へ upload
```

スケジュール: 毎年3月1日 9:00 UTC（FA・移籍確定後、開幕前）+ `workflow_dispatch`（手動実行）

**`hf_only` モード**: `workflow_dispatch` で `hf_only=true` を指定すると、データ再生成を一切走らせず HF 同期のみを実行します（手動 curate 済みの `data/` を上書きしない安全モード）。季中の全量再実行はロースターフィルタ等の手動 curate を上書きしうるため非推奨。

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

## 詳細と履歴

- [docs/technical-details.md](docs/technical-details.md) — ベイズ統合の技術詳細（4 層の構成・設計上の判断・実装フェーズ）と、信頼区間の作り方で直した不具合・残る注意
- [docs/history.md](docs/history.md) — 実装済み機能の一覧（変更の記録）と、退役した GCP 分析基盤

## 今後の予定

### その他

- [ ] Marcel重みをNPBデータ最適化値に更新（打者 8/4/3・REG_PA=2000 / 投手 4/5/2・REG_IP=800、ブートストラップ p=0.003 で有意）→ [npb-marcel-weight-study](https://github.com/yasumorishima/npb-marcel-weight-study)
- [ ] 精度が悪化したときの自動アラート

### モデルの限界

- **NPB在籍1〜2年の選手**: データが少ないため予測値がリーグ平均に強く引き戻される（1年のみ→約2/3がリーグ平均寄り）。選手名横の「NPB1年/2年」バッジで確認可能
- **若手の急成長**: 年齢調整は+0.3%/年と小さく、ブレイクするような急成長は捉えられない
- **球場改修**: パークファクターは5年移動平均を使用。2026年のバンテリンドーム改修等は未反映
- **データの見落とし**: 個人プロジェクトのため品質保証には限界がある（例: MLB移籍選手がフィルタされていないバグがあり、途中で修正した）

## 研究（統合したリポジトリ）

2026-10-08 に、別リポジトリだった NPB の研究をここへまとめた（履歴ごと）。元のリポジトリは移転先の案内を残して読み取り専用にしてある。

| フォルダ | 中身 | 元のリポジトリ |
|---|---|---|
| [research/stan/](research/stan/) | Stan による日本人・外国人選手モデル（v0〜v5）、パークファクター、チーム順位の検証 | npb-stan-research |
| [research/marcel-weights/](research/marcel-weights/) | Marcel 法の重み（5/4/3）を NPB データで探索 | npb-marcel-weight-study |
| [research/2021-backtest/](research/2021-backtest/) | 2021 年（ヤクルト・オリックスが最下位から優勝）を予測できたか | npb-2021-backtest |

⚠️ **稼働中のモデルと予測はこのリポジトリのルート（`data/projections/`）だけ**。`research/` の中の予測ファイル（例：`research/stan/data/projections/team_sim_2026.csv`）は研究の途中経過で、ルートの予測とは中身が違う。

## データソース

**このプロジェクトのNPBデータは以下のサイトから取得しています：**

- [プロ野球データFreak](https://baseball-data.com) — NPB選手成績・生年月日（2015-2025）
- [日本野球機構 NPB](https://npb.jp) — 詳細打撃成績（2B/3B/SF、wOBA/wRC+算出用）

詳細は [DATA_SOURCES.md](DATA_SOURCES.md) を参照。

## License

MIT
