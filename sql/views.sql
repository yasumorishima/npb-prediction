-- BigQuery 分析用ビュー（NPB版）
-- NPBデータ固有の分析クエリ集 + データ品質チェック

-- ============================================================
-- 1. 選手年度別 OPS トレンド（打者パフォーマンス推移）
-- ============================================================
CREATE OR REPLACE VIEW `data-platform-490901.npb.v_batter_trend` AS
SELECT
  h.player,
  h.team,
  h.year AS season,
  h.OPS,
  h.AVG,
  h.OBP,
  h.SLG,
  h.HR,
  h.PA,
  h.G,
  s.wOBA,
  s.`wRC_plus`,
  s.wRAA,
  p.PF_5yr,
  -- 前年比
  h.OPS - LAG(h.OPS) OVER (PARTITION BY h.player ORDER BY h.year) AS OPS_yoy,
  s.wOBA - LAG(s.wOBA) OVER (PARTITION BY h.player ORDER BY h.year) AS wOBA_yoy,
  h.HR - LAG(h.HR) OVER (PARTITION BY h.player ORDER BY h.year) AS HR_yoy,
  -- 移籍フラグ
  CASE
    WHEN h.team != LAG(h.team) OVER (PARTITION BY h.player ORDER BY h.year) THEN 1
    ELSE 0
  END AS team_changed
FROM `data-platform-490901.npb.raw_hitters` h
LEFT JOIN `data-platform-490901.npb.sabermetrics` s
  ON h.player = s.player AND h.year = s.year
LEFT JOIN `data-platform-490901.npb.park_factors` p
  ON h.team = p.team AND h.year = p.year
ORDER BY h.player, h.year;


-- ============================================================
-- 2. 選手年度別 ERA トレンド（投手パフォーマンス推移）
-- ============================================================
CREATE OR REPLACE VIEW `data-platform-490901.npb.v_pitcher_trend` AS
SELECT
  p.player,
  p.team,
  p.year AS season,
  p.ERA,
  p.WHIP,
  p.DIPS,
  p.W,
  p.L,
  p.SV,
  p.HLD,
  p.SO,
  p.BB,
  p.IP,
  p.G,
  pf.PF_5yr,
  -- 前年比
  p.ERA - LAG(p.ERA) OVER (PARTITION BY p.player ORDER BY p.year) AS ERA_yoy,
  p.WHIP - LAG(p.WHIP) OVER (PARTITION BY p.player ORDER BY p.year) AS WHIP_yoy,
  -- K/9, BB/9
  SAFE_DIVIDE(p.SO * 9, p.IP) AS K9,
  SAFE_DIVIDE(p.BB * 9, p.IP) AS BB9,
  -- FIP近似
  (13.0 * p.HRA + 3.0 * p.BB - 2.0 * p.SO) / NULLIF(p.IP, 0) + 3.2 AS FIP_approx,
  -- ERA-DIPS乖離
  p.ERA - p.DIPS AS era_dips_gap,
  -- 先発/リリーフ
  CASE WHEN p.IP >= 50 THEN 'starter' ELSE 'reliever' END AS role
FROM `data-platform-490901.npb.raw_pitchers` p
LEFT JOIN `data-platform-490901.npb.park_factors` pf
  ON p.team = pf.team AND p.year = pf.year
ORDER BY p.player, p.year;


-- ============================================================
-- 3. チームピタゴラス勝率分析（運要素の定量化）
-- ============================================================
CREATE OR REPLACE VIEW `data-platform-490901.npb.v_team_pythagorean` AS
SELECT
  year,
  league,
  team,
  G,
  W,
  L,
  D,
  actual_WPCT,
  RS,
  RA,
  pyth_WPCT_npb,
  pyth_W_npb,
  -- 実際の勝数とピタゴラス期待勝数の差（正=運が良い）
  W - pyth_W_npb AS luck_wins,
  -- 得失点差
  RS - RA AS run_diff,
  -- 得点効率
  SAFE_DIVIDE(RS, G) AS runs_per_game,
  SAFE_DIVIDE(RA, G) AS runs_allowed_per_game
FROM `data-platform-490901.npb.pythagorean`
ORDER BY year DESC, actual_WPCT DESC;


-- ============================================================
-- 4. wRC+ リーダーボード（年度別）
-- ============================================================
CREATE OR REPLACE VIEW `data-platform-490901.npb.v_sabermetrics_leaders` AS
SELECT
  s.player,
  h.team,
  s.year AS season,
  s.wOBA,
  s.`wRC_plus`,
  s.wRAA,
  h.OPS,
  h.AVG,
  h.HR,
  h.RBI,
  h.PA,
  h.G,
  st.league,
  -- 年度内順位
  ROW_NUMBER() OVER (PARTITION BY s.year ORDER BY s.`wRC_plus` DESC) AS wrc_rank,
  ROW_NUMBER() OVER (PARTITION BY s.year ORDER BY s.wOBA DESC) AS woba_rank,
  ROW_NUMBER() OVER (PARTITION BY s.year ORDER BY h.OPS DESC) AS ops_rank
FROM `data-platform-490901.npb.sabermetrics` s
JOIN `data-platform-490901.npb.raw_hitters` h
  ON s.player = h.player AND s.year = h.year
LEFT JOIN `data-platform-490901.npb.raw_standings` st
  ON h.team = st.team AND h.year = st.year
WHERE h.PA >= 300  -- 規定打席相当
ORDER BY s.year DESC, s.`wRC_plus` DESC;


-- ============================================================
-- 5. Marcel精度検証（予測年vs実績の比較）
-- ============================================================
CREATE OR REPLACE VIEW `data-platform-490901.npb.v_marcel_accuracy` AS
WITH historical AS (
  SELECT
    m.player,
    m.team,
    m.target_year,
    m.OPS AS marcel_ops,
    m.age,
    m.PA AS marcel_pa,
    h.OPS AS actual_ops,
    h.PA AS actual_pa,
    ABS(m.OPS - h.OPS) AS abs_error
  FROM `data-platform-490901.npb.marcel_hitters` m
  LEFT JOIN `data-platform-490901.npb.raw_hitters` h
    ON m.player = h.player AND m.target_year = h.year
  WHERE h.OPS IS NOT NULL
)
SELECT
  *,
  -- エラー水準の分類
  CASE
    WHEN abs_error < 0.030 THEN 'excellent'
    WHEN abs_error < 0.060 THEN 'good'
    WHEN abs_error < 0.100 THEN 'fair'
    ELSE 'poor'
  END AS accuracy_grade
FROM historical
ORDER BY abs_error;


-- ============================================================
-- 6. 年齢カーブ（NPB全体のaging curve）
-- ============================================================
CREATE OR REPLACE VIEW `data-platform-490901.npb.v_age_curve` AS
WITH with_age AS (
  SELECT
    h.player,
    h.year,
    h.OPS,
    h.PA,
    s.wOBA,
    DATE_DIFF(
      DATE(CONCAT(CAST(h.year AS STRING), '-04-01')),
      PARSE_DATE('%Y-%m-%d', CAST(b.birthday AS STRING)),
      YEAR
    ) AS age
  FROM `data-platform-490901.npb.raw_hitters` h
  LEFT JOIN `data-platform-490901.npb.raw_player_birthdays` b
    ON h.player = b.player
  LEFT JOIN `data-platform-490901.npb.sabermetrics` s
    ON h.player = s.player AND h.year = s.year
  WHERE h.PA >= 200
    AND b.birthday IS NOT NULL
)
SELECT
  age,
  COUNT(*) AS n_players,
  AVG(OPS) AS avg_ops,
  AVG(wOBA) AS avg_woba,
  STDDEV(OPS) AS stddev_ops,
  MIN(OPS) AS min_ops,
  MAX(OPS) AS max_ops,
  AVG(PA) AS avg_pa
FROM with_age
WHERE age BETWEEN 18 AND 45
GROUP BY age
ORDER BY age;


-- ============================================================
-- 7. パークファクター影響分析
-- ============================================================
CREATE OR REPLACE VIEW `data-platform-490901.npb.v_park_effects` AS
SELECT
  year,
  team,
  stadium,
  PF_5yr,
  CASE
    WHEN PF_5yr > 105 THEN 'hitter_park'
    WHEN PF_5yr < 95 THEN 'pitcher_park'
    ELSE 'neutral'
  END AS park_type,
  home_G,
  home_RS,
  home_RA,
  away_G,
  away_RS,
  away_RA,
  -- ホーム/アウェイの得点差
  SAFE_DIVIDE(home_RS, home_G) - SAFE_DIVIDE(away_RS, away_G) AS home_away_rs_diff
FROM `data-platform-490901.npb.park_factors`
ORDER BY year DESC, PF_5yr DESC;


-- ============================================================
-- 8. データ品質チェック: シーズン別カバレッジ
-- ============================================================
CREATE OR REPLACE VIEW `data-platform-490901.npb.v_data_coverage` AS
SELECT
  h.year AS season,
  -- 打者
  COUNT(DISTINCT h.player) AS n_hitters,
  COUNT(DISTINCT CASE WHEN h.PA >= 300 THEN h.player END) AS n_regulars,
  AVG(h.PA) AS avg_pa,
  -- 投手
  (SELECT COUNT(DISTINCT player) FROM `data-platform-490901.npb.raw_pitchers` p WHERE p.year = h.year) AS n_pitchers,
  (SELECT COUNT(DISTINCT player) FROM `data-platform-490901.npb.raw_pitchers` p WHERE p.year = h.year AND p.IP >= 100) AS n_starters,
  -- セイバーメトリクス
  COUNT(DISTINCT CASE WHEN s.wOBA IS NOT NULL THEN h.player END) AS n_with_saber,
  COUNT(DISTINCT CASE WHEN s.wOBA IS NULL THEN h.player END) AS n_missing_saber,
  -- 誕生日（年齢計算に必要）
  COUNT(DISTINCT CASE WHEN b.birthday IS NOT NULL THEN h.player END) AS n_with_birthday,
  COUNT(DISTINCT CASE WHEN b.birthday IS NULL THEN h.player END) AS n_missing_birthday,
  -- パークファクター
  COUNT(DISTINCT pf.team) AS n_teams_with_pf,
  -- 試合データ
  (SELECT COUNT(*) FROM `data-platform-490901.npb.raw_standings` st WHERE st.year = h.year) AS n_teams_standings
FROM `data-platform-490901.npb.raw_hitters` h
LEFT JOIN `data-platform-490901.npb.sabermetrics` s
  ON h.player = s.player AND h.year = s.year
LEFT JOIN `data-platform-490901.npb.raw_player_birthdays` b
  ON h.player = b.player
LEFT JOIN `data-platform-490901.npb.park_factors` pf
  ON h.team = pf.team AND h.year = pf.year
GROUP BY h.year
ORDER BY h.year;


-- ============================================================
-- 9. データ品質チェック: 欠損値サマリー
-- ============================================================
CREATE OR REPLACE VIEW `data-platform-490901.npb.v_data_quality` AS
WITH hitter_quality AS (
  SELECT
    'raw_hitters' AS source,
    COUNT(*) AS total_rows,
    COUNTIF(player IS NULL) AS null_player,
    COUNTIF(team IS NULL) AS null_team,
    COUNTIF(OPS IS NULL) AS null_ops,
    COUNTIF(PA IS NULL OR PA = 0) AS zero_pa,
    COUNTIF(AVG IS NULL) AS null_avg,
    COUNTIF(HR IS NULL) AS null_hr
  FROM `data-platform-490901.npb.raw_hitters`
),
pitcher_quality AS (
  SELECT
    'raw_pitchers' AS source,
    COUNT(*) AS total_rows,
    COUNTIF(player IS NULL) AS null_player,
    COUNTIF(team IS NULL) AS null_team,
    COUNTIF(ERA IS NULL) AS null_era,
    COUNTIF(IP IS NULL OR IP = 0) AS zero_ip,
    COUNTIF(WHIP IS NULL) AS null_whip,
    0 AS null_hr
  FROM `data-platform-490901.npb.raw_pitchers`
),
birthday_quality AS (
  SELECT
    'raw_player_birthdays' AS source,
    COUNT(*) AS total_rows,
    COUNTIF(player IS NULL) AS null_player,
    0 AS null_team,
    0 AS null_ops_or_era,
    0 AS zero_pa_or_ip,
    COUNTIF(birthday IS NULL) AS null_birthday,
    0 AS null_hr
  FROM `data-platform-490901.npb.raw_player_birthdays`
),
saber_quality AS (
  SELECT
    'sabermetrics' AS source,
    COUNT(*) AS total_rows,
    COUNTIF(player IS NULL) AS null_player,
    0 AS null_team,
    COUNTIF(wOBA IS NULL) AS null_woba,
    0 AS zero_pa_or_ip,
    COUNTIF(`wRC_plus` IS NULL) AS null_wrc_plus,
    0 AS null_hr
  FROM `data-platform-490901.npb.sabermetrics`
)
SELECT * FROM hitter_quality
UNION ALL
SELECT * FROM pitcher_quality
UNION ALL
SELECT source, total_rows, null_player, null_team, null_ops_or_era, zero_pa_or_ip, null_birthday, null_hr FROM birthday_quality
UNION ALL
SELECT source, total_rows, null_player, null_team, null_woba, zero_pa_or_ip, null_wrc_plus, null_hr FROM saber_quality;


-- ============================================================
-- 10. メトリクス履歴テーブル作成（存在しない場合のみ）
-- model_metrics_history は WRITE_APPEND で蓄積されるが、
-- 初回は空テーブルを作成しておく必要がある
-- ============================================================
CREATE TABLE IF NOT EXISTS `data-platform-490901.npb.model_metrics_history` (
  run_date STRING,
  run_timestamp STRING,
  data_end_year INT64
);
