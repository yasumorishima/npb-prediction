# 履歴

README から移した部分（文面は移した時点のまま）。

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
- [x] HF Dataset ミラー同期（npb-stats、annual_update.yml に HF upload step + `hf_only` モード）
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

## 今後の予定から外した完了済みの項目

- [x] BQML精度検証 — BQML（BT: OPS MAE 0.0642 / ERA MAE 0.909）は Python ML（OPS ~0.065 / ERA 0.92-0.93）と同等以上。特徴量もBQMLの方が豊富（パークファクター・DIPS・FIP近似・Marcel加重平均等）。両手法とも投手ERAではMarcel法（0.78）に及ばず、これはML共通の課題

### ~~GCP 分析基盤（BigQuery + BigQuery ML）~~（2026-04-19 退役）

> BigQuery `npb` データセットと BQML モデル（`bqml_batter_ops` / `bqml_pitcher_era` 等）は Grafana 公開廃止に伴い削除。予測ロジックは Python (`scripts/`) に一本化済み、BQ 時代のデータスナップショットは HF Dataset [npb-stats](https://huggingface.co/datasets/yasumorishima/npb-stats) の `npb/` ディレクトリに保持（選手単位の成績表は再配布の許可が無いため削除済み）。Streamlit / FastAPI は CSV (`data/projections/`) を直接読むため BQ 非依存で継続稼働。
