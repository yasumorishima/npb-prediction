-- BigQuery ML: NPB投手 ERA 予測モデル
-- Boosted Tree Regressor + 線形回帰 (ベースライン比較用)
--
-- データソース: raw_pitchers + raw_standings + raw_player_birthdays + park_factors
-- 粒度: 選手×シーズン → 翌シーズン ERA を予測
--
-- NPB投手固有の考慮点:
--   - DIPS (Defense Independent Pitching Statistics) が利用可能
--   - 先発/リリーフの分離（IP閾値で判定）
--   - セ・パ DH制差異が投手成績に影響

-- ============================================================
-- Step 1: 学習用ビュー（ラグ特徴量 + ターゲット）
-- ============================================================
CREATE OR REPLACE VIEW `data-platform-490901.npb.v_pitcher_train` AS
WITH base AS (
  SELECT
    p.player,
    p.team,
    p.year AS season,
    -- 基本成績
    p.ERA,
    p.WHIP,
    p.W,
    p.L,
    p.SV,
    p.HLD,
    p.G,
    p.BF,
    p.IP,
    p.HA,
    p.HRA,
    p.BB,
    p.HBP,
    p.SO,
    p.R,
    p.ER,
    p.DIPS,
    p.WPCT,
    -- パークファクター
    pf.PF_5yr,
    -- 年齢
    DATE_DIFF(
      DATE(CONCAT(CAST(p.year AS STRING), '-04-01')),
      PARSE_DATE('%Y-%m-%d', CAST(b.birthday AS STRING)),
      YEAR
    ) AS age,
    -- リーグ
    st.league,
    -- 先発/リリーフ判定（IP >= 50 を先発とみなす）
    CASE WHEN p.IP >= 50 THEN 1 ELSE 0 END AS is_starter
  FROM `data-platform-490901.npb.raw_pitchers` p
  LEFT JOIN `data-platform-490901.npb.park_factors` pf
    ON p.team = pf.team AND p.year = pf.year
  LEFT JOIN `data-platform-490901.npb.raw_player_birthdays` b
    ON p.player = b.player
  LEFT JOIN `data-platform-490901.npb.raw_standings` st
    ON p.team = st.team AND p.year = st.year
  WHERE p.IP >= 20  -- 最低投球回フィルタ
),
lagged AS (
  SELECT
    player,
    team,
    season,
    league,
    -- ===== y1 (直前シーズン) =====
    LAG(ERA, 1) OVER w AS ERA_y1,
    LAG(WHIP, 1) OVER w AS WHIP_y1,
    LAG(W, 1) OVER w AS W_y1,
    LAG(L, 1) OVER w AS L_y1,
    LAG(SV, 1) OVER w AS SV_y1,
    LAG(HLD, 1) OVER w AS HLD_y1,
    LAG(G, 1) OVER w AS G_y1,
    LAG(BF, 1) OVER w AS BF_y1,
    LAG(IP, 1) OVER w AS IP_y1,
    LAG(HA, 1) OVER w AS HA_y1,
    LAG(HRA, 1) OVER w AS HRA_y1,
    LAG(BB, 1) OVER w AS BB_y1,
    LAG(HBP, 1) OVER w AS HBP_y1,
    LAG(SO, 1) OVER w AS SO_y1,
    LAG(R, 1) OVER w AS R_y1,
    LAG(ER, 1) OVER w AS ER_y1,
    LAG(DIPS, 1) OVER w AS DIPS_y1,
    LAG(WPCT, 1) OVER w AS WPCT_y1,
    LAG(PF_5yr, 1) OVER w AS PF_5yr_y1,
    LAG(age, 1) OVER w AS age_y1,
    LAG(is_starter, 1) OVER w AS is_starter_y1,

    -- ===== y2 (2年前) =====
    LAG(ERA, 2) OVER w AS ERA_y2,
    LAG(WHIP, 2) OVER w AS WHIP_y2,
    LAG(IP, 2) OVER w AS IP_y2,
    LAG(SO, 2) OVER w AS SO_y2,
    LAG(BB, 2) OVER w AS BB_y2,
    LAG(HRA, 2) OVER w AS HRA_y2,
    LAG(DIPS, 2) OVER w AS DIPS_y2,

    -- ===== y3 (3年前: Marcel法3年加重用) =====
    LAG(ERA, 3) OVER w AS ERA_y3,
    LAG(IP, 3) OVER w AS IP_y3,

    -- ===== ターゲット =====
    ERA AS target_era,

    -- ===== エンジニアリング特徴量 =====
    -- ラグ差分
    LAG(ERA, 1) OVER w - LAG(ERA, 2) OVER w AS ERA_delta_1,
    LAG(ERA, 2) OVER w - LAG(ERA, 3) OVER w AS ERA_delta_2,
    LAG(WHIP, 1) OVER w - LAG(WHIP, 2) OVER w AS WHIP_delta_1,
    LAG(SO, 1) OVER w - LAG(SO, 2) OVER w AS SO_delta_1,
    LAG(BB, 1) OVER w - LAG(BB, 2) OVER w AS BB_delta_1,
    LAG(DIPS, 1) OVER w - LAG(DIPS, 2) OVER w AS DIPS_delta_1,

    -- 比率指標（IP正規化: /9イニング）
    SAFE_DIVIDE(LAG(SO, 1) OVER w * 9, LAG(IP, 1) OVER w) AS K9_y1,
    SAFE_DIVIDE(LAG(BB, 1) OVER w * 9, LAG(IP, 1) OVER w) AS BB9_y1,
    SAFE_DIVIDE(LAG(HRA, 1) OVER w * 9, LAG(IP, 1) OVER w) AS HR9_y1,
    SAFE_DIVIDE(LAG(HA, 1) OVER w * 9, LAG(IP, 1) OVER w) AS H9_y1,
    -- K-BB%
    SAFE_DIVIDE(LAG(SO, 1) OVER w - LAG(BB, 1) OVER w, LAG(BF, 1) OVER w) AS K_BB_rate_y1,
    -- K%
    SAFE_DIVIDE(LAG(SO, 1) OVER w, LAG(BF, 1) OVER w) AS K_rate_y1,
    -- BB%
    SAFE_DIVIDE(LAG(BB, 1) OVER w, LAG(BF, 1) OVER w) AS BB_rate_y1,
    -- FIP近似 (NPBにはFIP列がないので概算)
    -- FIP = (13*HR + 3*BB - 2*K)/IP + constant(≒3.2 for NPB)
    (13.0 * LAG(HRA, 1) OVER w + 3.0 * LAG(BB, 1) OVER w - 2.0 * LAG(SO, 1) OVER w)
      / NULLIF(LAG(IP, 1) OVER w, 0) + 3.2 AS FIP_approx_y1,
    -- ERA-DIPS乖離（運要素）
    LAG(ERA, 1) OVER w - LAG(DIPS, 1) OVER w AS era_dips_gap,

    -- 年齢系
    LAG(age, 1) OVER w - 27 AS age_from_peak,  -- 投手ピーク=27歳
    POW(LAG(age, 1) OVER w - 27, 2) AS age_sq,

    -- 投球量（NPB: 先発フル=180IP, リリーフ=60IP）
    LAG(IP, 1) OVER w / 180.0 AS ip_rate,
    LAG(IP, 1) OVER w - LAG(IP, 2) OVER w AS IP_trend,

    -- チーム変更
    CASE
      WHEN LAG(team, 1) OVER w != LAG(team, 2) OVER w
        AND LAG(team, 1) OVER w IS NOT NULL
        AND LAG(team, 2) OVER w IS NOT NULL
      THEN 1 ELSE 0
    END AS team_changed,

    -- リーグ変更
    CASE
      WHEN LAG(league, 1) OVER w != LAG(league, 2) OVER w
        AND LAG(league, 1) OVER w IS NOT NULL
        AND LAG(league, 2) OVER w IS NOT NULL
      THEN 1 ELSE 0
    END AS league_changed,

    -- パリーグフラグ
    CASE WHEN league = 'パ・リーグ' THEN 1 ELSE 0 END AS is_pacific,

    -- 役割変更（先発↔リリーフ）
    CASE
      WHEN LAG(is_starter, 1) OVER w != LAG(is_starter, 2) OVER w
        AND LAG(is_starter, 2) OVER w IS NOT NULL
      THEN 1 ELSE 0
    END AS role_changed,

    -- Marcel加重平均（ERA版）
    SAFE_DIVIDE(
      LAG(ERA, 1) OVER w * 4 * LAG(IP, 1) OVER w
      + COALESCE(LAG(ERA, 2) OVER w * 5 * LAG(IP, 2) OVER w, 0)
      + COALESCE(LAG(ERA, 3) OVER w * 2 * LAG(IP, 3) OVER w, 0),
      LAG(IP, 1) OVER w * 4
      + COALESCE(LAG(IP, 2) OVER w * 5, 0)
      + COALESCE(LAG(IP, 3) OVER w * 2, 0)
    ) AS marcel_era_raw,

    -- 時系列CV用
    CASE WHEN season >= (SELECT MAX(year) - 1 FROM `data-platform-490901.npb.raw_pitchers`)
      THEN TRUE ELSE FALSE
    END AS is_eval

  FROM base
  WINDOW w AS (PARTITION BY player ORDER BY season)
)
SELECT * FROM lagged
WHERE ERA_y1 IS NOT NULL
;


-- ============================================================
-- Step 2: Boosted Tree Regressor
-- ============================================================
CREATE OR REPLACE MODEL `data-platform-490901.npb.bqml_pitcher_era`
OPTIONS(
  model_type = 'BOOSTED_TREE_REGRESSOR',
  input_label_cols = ['target_era'],
  max_iterations = 100,
  learn_rate = 0.03,
  max_tree_depth = 4,
  subsample = 0.8,
  min_tree_child_weight = 10,
  colsample_bytree = 0.8,
  l1_reg = 0.1,
  l2_reg = 2.0,
  early_stop = TRUE,
  min_rel_progress = 0.001,
  data_split_method = 'CUSTOM',
  data_split_col = 'is_eval'
) AS
SELECT
  -- y1 features
  ERA_y1, WHIP_y1, W_y1, L_y1, SV_y1, HLD_y1,
  G_y1, BF_y1, IP_y1, HA_y1, HRA_y1, BB_y1, HBP_y1, SO_y1,
  R_y1, ER_y1, DIPS_y1, WPCT_y1, PF_5yr_y1,
  is_starter_y1,
  -- y2 features
  ERA_y2, WHIP_y2, IP_y2, SO_y2, BB_y2, HRA_y2, DIPS_y2,
  -- y3 features
  ERA_y3, IP_y3,
  -- delta features
  ERA_delta_1, ERA_delta_2, WHIP_delta_1,
  SO_delta_1, BB_delta_1, DIPS_delta_1,
  -- rate features
  K9_y1, BB9_y1, HR9_y1, H9_y1,
  K_BB_rate_y1, K_rate_y1, BB_rate_y1,
  FIP_approx_y1, era_dips_gap,
  -- engineered
  age_from_peak, age_sq, ip_rate, IP_trend,
  team_changed, league_changed, is_pacific, role_changed,
  marcel_era_raw,
  -- target
  target_era,
  -- split
  is_eval
FROM `data-platform-490901.npb.v_pitcher_train`
;


-- ============================================================
-- Step 3: 線形回帰 (ベースライン比較用)
-- ============================================================
CREATE OR REPLACE MODEL `data-platform-490901.npb.bqml_pitcher_era_linear`
OPTIONS(
  model_type = 'LINEAR_REG',
  input_label_cols = ['target_era'],
  optimize_strategy = 'NORMAL_EQUATION',
  l2_reg = 2.0,
  data_split_method = 'CUSTOM',
  data_split_col = 'is_eval'
) AS
SELECT
  ERA_y1, WHIP_y1, DIPS_y1,
  IP_y1, SO_y1, BB_y1, HRA_y1,
  ERA_y2, IP_y2,
  ERA_delta_1, K9_y1, BB9_y1, HR9_y1,
  K_BB_rate_y1, FIP_approx_y1, era_dips_gap,
  age_from_peak, age_sq, ip_rate, IP_trend,
  team_changed, is_pacific, is_starter_y1, role_changed,
  marcel_era_raw,
  target_era,
  is_eval
FROM `data-platform-490901.npb.v_pitcher_train`
;
