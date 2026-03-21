-- BigQuery ML: NPB打者 OPS 予測モデル
-- Boosted Tree Regressor + 線形回帰 (ベースライン比較用)
--
-- データソース: raw_hitters + raw_batting_detailed + sabermetrics + park_factors + raw_player_birthdays
-- Marcel法と同じ特徴量構造をSQLウインドウ関数で再現
-- 粒度: 選手×シーズン → 翌シーズン OPS を予測
--
-- NPB固有の考慮点:
--   - Statcast (EV/バレル等) なし → 従来成績 + セイバーメトリクス
--   - 年間143試合（MLB 162試合）→ PA/IPの閾値を調整
--   - 外国人選手の入退団が激しい → チーム変更フラグ重要
--   - セ・パ DH制差異 → リーグフラグ

-- ============================================================
-- Step 1: 学習用ビュー（ラグ特徴量 + ターゲット）
-- ============================================================
CREATE OR REPLACE VIEW `data-platform-490901.npb.v_batter_train` AS
WITH base AS (
  SELECT
    h.player,
    h.team,
    h.year AS season,
    -- 基本成績
    h.AVG,
    h.OBP,
    h.SLG,
    h.OPS,
    h.HR,
    h.RBI,
    h.SB,
    h.BB,
    h.SO,
    h.PA,
    h.G,
    h.HBP,
    h.GDP,
    -- 詳細打撃成績（wOBA算出の元データ）
    d.`_2B` AS doubles,
    d.`_3B` AS triples,
    d.SF,
    d.CS,
    d.R,
    d.TB,
    -- セイバーメトリクス
    s.wOBA,
    s.`wRC_plus`,
    s.wRAA,
    -- パークファクター
    p.PF_5yr,
    -- 年齢（開幕時点）
    DATE_DIFF(
      DATE(CONCAT(CAST(h.year AS STRING), '-04-01')),
      PARSE_DATE('%Y-%m-%d', CAST(b.birthday AS STRING)),
      YEAR
    ) AS age,
    -- リーグ（セ/パ）
    st.league
  FROM `data-platform-490901.npb.raw_hitters` h
  LEFT JOIN `data-platform-490901.npb.raw_batting_detailed` d
    ON h.player = d.player AND h.year = d.year
  LEFT JOIN `data-platform-490901.npb.sabermetrics` s
    ON h.player = s.player AND h.year = s.year
  LEFT JOIN `data-platform-490901.npb.park_factors` p
    ON h.team = p.team AND h.year = p.year
  LEFT JOIN `data-platform-490901.npb.raw_player_birthdays` b
    ON h.player = b.player
  LEFT JOIN `data-platform-490901.npb.raw_standings` st
    ON h.team = st.team AND h.year = st.year
  WHERE h.PA >= 100  -- NPBは143試合なのでMLBより閾値を下げる
),
lagged AS (
  SELECT
    player,
    team,
    season,
    league,
    -- ===== y1 (直前シーズン) =====
    LAG(OPS, 1) OVER w AS OPS_y1,
    LAG(AVG, 1) OVER w AS AVG_y1,
    LAG(OBP, 1) OVER w AS OBP_y1,
    LAG(SLG, 1) OVER w AS SLG_y1,
    LAG(HR, 1) OVER w AS HR_y1,
    LAG(RBI, 1) OVER w AS RBI_y1,
    LAG(SB, 1) OVER w AS SB_y1,
    LAG(BB, 1) OVER w AS BB_y1,
    LAG(SO, 1) OVER w AS SO_y1,
    LAG(PA, 1) OVER w AS PA_y1,
    LAG(G, 1) OVER w AS G_y1,
    LAG(HBP, 1) OVER w AS HBP_y1,
    LAG(GDP, 1) OVER w AS GDP_y1,
    LAG(doubles, 1) OVER w AS doubles_y1,
    LAG(triples, 1) OVER w AS triples_y1,
    LAG(R, 1) OVER w AS R_y1,
    LAG(TB, 1) OVER w AS TB_y1,
    LAG(wOBA, 1) OVER w AS wOBA_y1,
    LAG(`wRC_plus`, 1) OVER w AS wRC_plus_y1,
    LAG(wRAA, 1) OVER w AS wRAA_y1,
    LAG(PF_5yr, 1) OVER w AS PF_5yr_y1,
    LAG(age, 1) OVER w AS age_y1,

    -- ===== y2 (2年前) =====
    LAG(OPS, 2) OVER w AS OPS_y2,
    LAG(AVG, 2) OVER w AS AVG_y2,
    LAG(wOBA, 2) OVER w AS wOBA_y2,
    LAG(PA, 2) OVER w AS PA_y2,
    LAG(HR, 2) OVER w AS HR_y2,
    LAG(BB, 2) OVER w AS BB_y2,
    LAG(SO, 2) OVER w AS SO_y2,

    -- ===== y3 (3年前: Marcel法3年加重用) =====
    LAG(OPS, 3) OVER w AS OPS_y3,
    LAG(PA, 3) OVER w AS PA_y3,

    -- ===== ターゲット =====
    OPS AS target_ops,

    -- ===== エンジニアリング特徴量 =====
    -- ラグ差分（トレンド把握）
    LAG(OPS, 1) OVER w - LAG(OPS, 2) OVER w AS OPS_delta_1,
    LAG(OPS, 2) OVER w - LAG(OPS, 3) OVER w AS OPS_delta_2,
    LAG(AVG, 1) OVER w - LAG(AVG, 2) OVER w AS AVG_delta_1,
    LAG(wOBA, 1) OVER w - LAG(wOBA, 2) OVER w AS wOBA_delta_1,
    LAG(HR, 1) OVER w - LAG(HR, 2) OVER w AS HR_delta_1,
    LAG(BB, 1) OVER w - LAG(BB, 2) OVER w AS BB_delta_1,
    LAG(SO, 1) OVER w - LAG(SO, 2) OVER w AS SO_delta_1,

    -- 比率指標（PA正規化）
    SAFE_DIVIDE(LAG(BB, 1) OVER w, LAG(PA, 1) OVER w) AS BB_rate_y1,
    SAFE_DIVIDE(LAG(SO, 1) OVER w, LAG(PA, 1) OVER w) AS SO_rate_y1,
    SAFE_DIVIDE(LAG(HR, 1) OVER w, LAG(PA, 1) OVER w) AS HR_rate_y1,
    SAFE_DIVIDE(LAG(BB, 1) OVER w - LAG(SO, 1) OVER w, LAG(PA, 1) OVER w) AS BB_SO_diff_rate_y1,
    -- ISO (Isolated Power = SLG - AVG)
    LAG(SLG, 1) OVER w - LAG(AVG, 1) OVER w AS ISO_y1,
    -- BABIP近似 (H - HR) / (AB - SO - HR + SF)
    -- ここではシンプルに差分を使う
    LAG(SLG, 2) OVER w - LAG(AVG, 2) OVER w AS ISO_y2,

    -- 年齢系
    LAG(age, 1) OVER w - 29 AS age_from_peak,  -- NPBピーク=29歳
    POW(LAG(age, 1) OVER w - 29, 2) AS age_sq,

    -- 出場率（NPB: 143試合 × ~4.5PA ≒ 640PA がフル出場）
    LAG(PA, 1) OVER w / 600.0 AS pa_rate,
    LAG(PA, 1) OVER w - LAG(PA, 2) OVER w AS PA_trend,

    -- チーム変更（移籍フラグ）
    CASE
      WHEN LAG(team, 1) OVER w != LAG(team, 2) OVER w
        AND LAG(team, 1) OVER w IS NOT NULL
        AND LAG(team, 2) OVER w IS NOT NULL
      THEN 1 ELSE 0
    END AS team_changed,

    -- リーグ変更（セ→パ or パ→セ）
    CASE
      WHEN LAG(league, 1) OVER w != LAG(league, 2) OVER w
        AND LAG(league, 1) OVER w IS NOT NULL
        AND LAG(league, 2) OVER w IS NOT NULL
      THEN 1 ELSE 0
    END AS league_changed,

    -- パリーグフラグ（DH制）
    CASE WHEN league = 'パ・リーグ' THEN 1 ELSE 0 END AS is_pacific,

    -- Marcel加重平均（SQLで再現: 5/4/3 + 回帰）
    SAFE_DIVIDE(
      LAG(OPS, 1) OVER w * 5 * LAG(PA, 1) OVER w
      + COALESCE(LAG(OPS, 2) OVER w * 4 * LAG(PA, 2) OVER w, 0)
      + COALESCE(LAG(OPS, 3) OVER w * 3 * LAG(PA, 3) OVER w, 0),
      LAG(PA, 1) OVER w * 5
      + COALESCE(LAG(PA, 2) OVER w * 4, 0)
      + COALESCE(LAG(PA, 3) OVER w * 3, 0)
    ) AS marcel_ops_raw,

    -- 時系列CV用
    CASE WHEN season >= (SELECT MAX(year) - 1 FROM `data-platform-490901.npb.raw_hitters`)
      THEN TRUE ELSE FALSE
    END AS is_eval

  FROM base
  WINDOW w AS (PARTITION BY player ORDER BY season)
)
SELECT * FROM lagged
WHERE OPS_y1 IS NOT NULL  -- 最低1年の過去データが必要
;


-- ============================================================
-- Step 2: Boosted Tree Regressor
-- ============================================================
CREATE OR REPLACE MODEL `data-platform-490901.npb.bqml_batter_ops`
OPTIONS(
  model_type = 'BOOSTED_TREE_REGRESSOR',
  input_label_cols = ['target_ops'],
  max_iterations = 100,  -- NPBはデータ量がMLBより少ない→過学習対策
  learn_rate = 0.03,     -- 低めの学習率で安定性重視
  max_tree_depth = 4,    -- MLBの6より浅く（特徴量少ない）
  subsample = 0.8,
  min_tree_child_weight = 10,  -- MLBの5より大きく（サンプル少ない対策）
  colsample_bytree = 0.8,
  l1_reg = 0.1,
  l2_reg = 2.0,         -- MLBの1.0より強め（過学習抑制）
  early_stop = TRUE,
  min_rel_progress = 0.001,
  data_split_method = 'CUSTOM',
  data_split_col = 'is_eval'
) AS
SELECT
  -- y1 features
  OPS_y1, AVG_y1, OBP_y1, SLG_y1,
  HR_y1, RBI_y1, SB_y1, BB_y1, SO_y1, PA_y1, G_y1,
  HBP_y1, GDP_y1, doubles_y1, triples_y1, R_y1, TB_y1,
  wOBA_y1, wRC_plus_y1, wRAA_y1, PF_5yr_y1,
  -- y2 features
  OPS_y2, AVG_y2, wOBA_y2, PA_y2, HR_y2, BB_y2, SO_y2,
  -- y3 features
  OPS_y3, PA_y3,
  -- delta features
  OPS_delta_1, OPS_delta_2, AVG_delta_1, wOBA_delta_1,
  HR_delta_1, BB_delta_1, SO_delta_1,
  -- rate features
  BB_rate_y1, SO_rate_y1, HR_rate_y1, BB_SO_diff_rate_y1,
  ISO_y1, ISO_y2,
  -- engineered
  age_from_peak, age_sq, pa_rate, PA_trend,
  team_changed, league_changed, is_pacific,
  marcel_ops_raw,
  -- target
  target_ops,
  -- split
  is_eval
FROM `data-platform-490901.npb.v_batter_train`
;


-- ============================================================
-- Step 3: 線形回帰 (ベースライン比較用)
-- ============================================================
CREATE OR REPLACE MODEL `data-platform-490901.npb.bqml_batter_ops_linear`
OPTIONS(
  model_type = 'LINEAR_REG',
  input_label_cols = ['target_ops'],
  optimize_strategy = 'NORMAL_EQUATION',
  l2_reg = 2.0,
  data_split_method = 'CUSTOM',
  data_split_col = 'is_eval'
) AS
SELECT
  OPS_y1, AVG_y1, wOBA_y1, wRC_plus_y1,
  HR_y1, BB_y1, SO_y1, PA_y1,
  OPS_y2, PA_y2,
  OPS_delta_1, BB_rate_y1, SO_rate_y1, ISO_y1,
  age_from_peak, age_sq, pa_rate, PA_trend,
  team_changed, is_pacific,
  marcel_ops_raw,
  target_ops,
  is_eval
FROM `data-platform-490901.npb.v_batter_train`
;
