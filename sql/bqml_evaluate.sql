-- BigQuery ML: NPBモデル評価 + Python版との精度比較
-- 各モデルの MAE を一括取得し、Python版 Marcel/ML と比較する

-- ============================================================
-- 1. BQML Boosted Tree 打者評価
-- ============================================================
SELECT
  'bqml_batter_boosted_tree' AS model,
  *
FROM ML.EVALUATE(
  MODEL `data-platform-490901.npb.bqml_batter_ops`,
  (SELECT * FROM `data-platform-490901.npb.v_batter_train` WHERE is_eval = TRUE)
);

-- ============================================================
-- 2. BQML 線形回帰 打者評価
-- ============================================================
SELECT
  'bqml_batter_linear' AS model,
  *
FROM ML.EVALUATE(
  MODEL `data-platform-490901.npb.bqml_batter_ops_linear`,
  (SELECT * FROM `data-platform-490901.npb.v_batter_train` WHERE is_eval = TRUE)
);

-- ============================================================
-- 3. BQML Boosted Tree 投手評価
-- ============================================================
SELECT
  'bqml_pitcher_boosted_tree' AS model,
  *
FROM ML.EVALUATE(
  MODEL `data-platform-490901.npb.bqml_pitcher_era`,
  (SELECT * FROM `data-platform-490901.npb.v_pitcher_train` WHERE is_eval = TRUE)
);

-- ============================================================
-- 4. BQML 線形回帰 投手評価
-- ============================================================
SELECT
  'bqml_pitcher_linear' AS model,
  *
FROM ML.EVALUATE(
  MODEL `data-platform-490901.npb.bqml_pitcher_era_linear`,
  (SELECT * FROM `data-platform-490901.npb.v_pitcher_train` WHERE is_eval = TRUE)
);

-- ============================================================
-- 5. BQML vs Python: 打者 OPS MAE 比較
-- ============================================================
WITH bqml_preds AS (
  SELECT
    player, season, predicted_target_ops AS bqml_ops
  FROM ML.PREDICT(
    MODEL `data-platform-490901.npb.bqml_batter_ops`,
    (SELECT * FROM `data-platform-490901.npb.v_batter_train` WHERE is_eval = TRUE)
  )
),
actual AS (
  SELECT player, year AS season, OPS AS actual_ops
  FROM `data-platform-490901.npb.raw_hitters`
  WHERE year = (SELECT MAX(year) FROM `data-platform-490901.npb.raw_hitters`)
    AND PA >= 100
),
python_ml AS (
  SELECT player, pred_OPS
  FROM `data-platform-490901.npb.ml_hitters`
),
marcel AS (
  SELECT player, OPS AS marcel_ops
  FROM `data-platform-490901.npb.marcel_hitters`
)
SELECT
  'batter_comparison' AS comparison,
  AVG(ABS(a.actual_ops - b.bqml_ops)) AS bqml_mae,
  AVG(ABS(a.actual_ops - p.pred_OPS)) AS python_ml_mae,
  AVG(ABS(a.actual_ops - m.marcel_ops)) AS marcel_mae,
  COUNT(*) AS n_players
FROM actual a
LEFT JOIN bqml_preds b ON a.player = b.player
LEFT JOIN python_ml p ON a.player = p.player
LEFT JOIN marcel m ON a.player = m.player
WHERE b.bqml_ops IS NOT NULL
  AND p.pred_OPS IS NOT NULL
  AND m.marcel_ops IS NOT NULL;

-- ============================================================
-- 6. BQML vs Python: 投手 ERA MAE 比較
-- ============================================================
WITH bqml_preds AS (
  SELECT
    player, season, predicted_target_era AS bqml_era
  FROM ML.PREDICT(
    MODEL `data-platform-490901.npb.bqml_pitcher_era`,
    (SELECT * FROM `data-platform-490901.npb.v_pitcher_train` WHERE is_eval = TRUE)
  )
),
actual AS (
  SELECT player, year AS season, ERA AS actual_era
  FROM `data-platform-490901.npb.raw_pitchers`
  WHERE year = (SELECT MAX(year) FROM `data-platform-490901.npb.raw_pitchers`)
    AND IP >= 20
),
python_ml AS (
  SELECT player, pred_ERA
  FROM `data-platform-490901.npb.ml_pitchers`
),
marcel AS (
  SELECT player, ERA AS marcel_era
  FROM `data-platform-490901.npb.marcel_pitchers`
)
SELECT
  'pitcher_comparison' AS comparison,
  AVG(ABS(a.actual_era - b.bqml_era)) AS bqml_mae,
  AVG(ABS(a.actual_era - p.pred_ERA)) AS python_ml_mae,
  AVG(ABS(a.actual_era - m.marcel_era)) AS marcel_mae,
  COUNT(*) AS n_players
FROM actual a
LEFT JOIN bqml_preds b ON a.player = b.player
LEFT JOIN python_ml p ON a.player = p.player
LEFT JOIN marcel m ON a.player = m.player
WHERE b.bqml_era IS NOT NULL
  AND p.pred_ERA IS NOT NULL
  AND m.marcel_era IS NOT NULL;
