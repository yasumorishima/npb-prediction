-- BigQuery ML: NPB翌シーズン予測を実行して結果テーブルに保存
--
-- 最新シーズンのデータを入力として、翌年の OPS / ERA を予測する。
-- 結果は bqml_predictions_batter / bqml_predictions_pitcher テーブルに保存。

-- ============================================================
-- 1. 打者 OPS 予測 (Boosted Tree + 線形回帰)
-- ============================================================
CREATE OR REPLACE TABLE `data-platform-490901.npb.bqml_predictions_batter` AS
WITH latest AS (
  SELECT MAX(year) AS max_season
  FROM `data-platform-490901.npb.raw_hitters`
),
predict_input AS (
  SELECT t.*
  FROM `data-platform-490901.npb.v_batter_train` t, latest l
  WHERE t.season = l.max_season
),
bt_preds AS (
  SELECT
    player,
    team,
    season,
    season + 1 AS pred_year,
    predicted_target_ops AS bqml_ops_boosted,
    target_ops AS actual_ops
  FROM ML.PREDICT(
    MODEL `data-platform-490901.npb.bqml_batter_ops`,
    (SELECT * FROM predict_input)
  )
),
linear_preds AS (
  SELECT
    player,
    predicted_target_ops AS bqml_ops_linear
  FROM ML.PREDICT(
    MODEL `data-platform-490901.npb.bqml_batter_ops_linear`,
    (SELECT * FROM predict_input)
  )
),
-- Marcel/ML Python予測もJOINして一覧化
marcel AS (
  SELECT player, OPS AS marcel_ops
  FROM `data-platform-490901.npb.marcel_hitters`
),
python_ml AS (
  SELECT player, pred_OPS AS python_ml_ops
  FROM `data-platform-490901.npb.ml_hitters`
)
SELECT
  b.player,
  b.team,
  b.season AS season_last,
  b.pred_year,
  b.bqml_ops_boosted,
  l.bqml_ops_linear,
  m.marcel_ops,
  p.python_ml_ops,
  b.actual_ops,
  CURRENT_TIMESTAMP() AS predicted_at
FROM bt_preds b
LEFT JOIN linear_preds l ON b.player = l.player
LEFT JOIN marcel m ON b.player = m.player
LEFT JOIN python_ml p ON b.player = p.player
ORDER BY b.bqml_ops_boosted DESC;


-- ============================================================
-- 2. 投手 ERA 予測 (Boosted Tree + 線形回帰)
-- ============================================================
CREATE OR REPLACE TABLE `data-platform-490901.npb.bqml_predictions_pitcher` AS
WITH latest AS (
  SELECT MAX(year) AS max_season
  FROM `data-platform-490901.npb.raw_pitchers`
),
predict_input AS (
  SELECT t.*
  FROM `data-platform-490901.npb.v_pitcher_train` t, latest l
  WHERE t.season = l.max_season
),
bt_preds AS (
  SELECT
    player,
    team,
    season,
    season + 1 AS pred_year,
    predicted_target_era AS bqml_era_boosted,
    target_era AS actual_era
  FROM ML.PREDICT(
    MODEL `data-platform-490901.npb.bqml_pitcher_era`,
    (SELECT * FROM predict_input)
  )
),
linear_preds AS (
  SELECT
    player,
    predicted_target_era AS bqml_era_linear
  FROM ML.PREDICT(
    MODEL `data-platform-490901.npb.bqml_pitcher_era_linear`,
    (SELECT * FROM predict_input)
  )
),
marcel AS (
  SELECT player, ERA AS marcel_era
  FROM `data-platform-490901.npb.marcel_pitchers`
),
python_ml AS (
  SELECT player, pred_ERA AS python_ml_era
  FROM `data-platform-490901.npb.ml_pitchers`
)
SELECT
  b.player,
  b.team,
  b.season AS season_last,
  b.pred_year,
  b.bqml_era_boosted,
  l.bqml_era_linear,
  m.marcel_era,
  p.python_ml_era,
  b.actual_era,
  CURRENT_TIMESTAMP() AS predicted_at
FROM bt_preds b
LEFT JOIN linear_preds l ON b.player = l.player
LEFT JOIN marcel m ON b.player = m.player
LEFT JOIN python_ml p ON b.player = p.player
ORDER BY b.bqml_era_boosted ASC;
