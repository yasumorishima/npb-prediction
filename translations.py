"""Translation strings for NPB Prediction dashboard (Japanese / English)."""

TEXTS: dict[str, dict[str, str]] = {
    "ja": {
        # --- Sidebar ---
        "sidebar_title": "NPB予測",
        "nav_label": "ページ選択",
        "glossary": "用語の説明",
        "glossary_ops": "**OPS** — 出塁率＋長打率。打者の総合打撃力を示す",
        "glossary_era": "**防御率（ERA）** — 9イニングあたりの平均失点。低いほど優秀",
        "glossary_whip": "**WHIP** — 1イニングあたりに許した走者数。低いほど優秀",
        "glossary_woba": "**wOBA** — 打席あたりの得点貢献度。四球・単打・本塁打等を重みづけ",
        "glossary_wrcplus": "**wRC+** — リーグ平均を100とした打撃力。120なら平均より2割上",
        "data_source": "データソース: [プロ野球データFreak](https://baseball-data.com) / [日本野球機構 NPB](https://npb.jp)",

        # --- Page names ---
        "page_top": "トップ",
        "page_standings": "予測順位表",
        "page_hitter": "打者予測",
        "page_pitcher": "投手予測",
        "page_hitter_rank": "打者ランキング",
        "page_pitcher_rank": "投手ランキング",
        "page_vs": "VS対決",
        "page_team_wpct": "チーム勝率",
        "page_saber": "選手の実力指標",

        # --- Common ---
        "no_data": "データが読み込めませんでした",
        "central_league": "セ・リーグ",
        "pacific_league": "パ・リーグ",
        "year_label": "年度",
        "team_label": "チーム",
        "all_years": "全年度",
        "wins_suffix": "勝",
        "losses_suffix": "敗",
        "wpct_label": "勝率",
        "rs_label": "得点",
        "ra_label": "失点",
        "stat_pred": "Marcel法",
        "ai_pred": "機械学習",
        "search_by_name": "選手名で検索（部分一致）",
        "search_prompt_btn": "選手名を入力するか、上のボタンをタップしてください",
        "no_player_found": "「{name}」に該当する選手が見つかりません",
        "no_data_team_year": "{team} ({year}) のデータがありません",
        "search_label": "選手名で検索",
        "search_prompt": "選手名を入力してください",
        "no_match": "「{name}」に該当するデータがありません",

        # --- Stat column names ---
        "col_player": "選手名",
        "col_avg": "打率",
        "col_hr": "本塁打",
        "col_rbi": "打点",
        "col_h": "安打",
        "col_bb": "四球",
        "col_sb": "盗塁",
        "col_obp": "出塁率",
        "col_slg": "長打率",
        "col_era": "防御率",
        "col_w": "勝利",
        "col_so": "奪三振",
        "col_ip": "投球回",

        # --- Bar / card labels ---
        "bar_hr": "本塁打",
        "bar_avg": "打率",
        "bar_obp": "出塁率",
        "bar_slg": "長打率",
        "bar_so": "奪三振",
        "bar_w": "勝利",
        "bar_ip": "投球回",
        "bar_era": "防御率",

        # --- Radar chart categories ---
        "radar_hr": "本塁打",
        "radar_avg": "打率",
        "radar_obp": "出塁率",
        "radar_slg": "長打率",
        "radar_era": "防御率",
        "radar_so": "奪三振",
        "radar_ip": "投球回",
        "radar_w": "勝利",

        # --- Player kind ---
        "foreign_player": "外国人",
        "rookie_no_data": "データ不足",
        "wraa_zero_note": "リーグ平均の貢献として計算",
        "wraa_zero_inline": "wRAA=0で計算中",

        # --- Top page ---
        "top_title": "NPB 2026 予測",
        "top_subtitle": "過去の成績データ × AI予測",
        "top_warning": (
            "⚠️ **ご注意 — これは統計モデルの自動計算結果です**\n\n"
            "Marcel法が「過去3年のNPB成績データ」だけをもとに算出した参考値です。"
            "好きなチームや選手が低く出ていても、それはモデルが過去の数字をそう計算したというだけで、"
            "作者の見解・応援・願望とは一切関係ありません。\n\n"
            "**このモデルには捉えられない要素がたくさんあります** —— "
            "新外国人・新人・復帰選手など、NPBでの過去データがない選手の貢献はすべて「平均」として扱われています。"
            "記録のない選手たちが活躍すれば、どのチームの順位も大きく変わりえます。"
            "シーズンが始まってみないとわからない部分が必ずあります。\n\n"
            "2025–2026オフの移籍・退団は反映済みです。"
        ),
        "btn_all_top3": "全体TOP3",
        "team_batters_title": "{team} 打者一覧（2026年予測）",
        "team_pitchers_title": "{team} 投手一覧（2026年予測）",
        "batter_pred_caption": "過去3年の成績から予測した2026年の成績です。小数が出るのは統計的な予測値のためです。",
        "pitcher_pred_caption": "過去3年の成績から予測した2026年の成績です。",
        "how_to_read": "指標の見方",
        "batter_stats_help": (
            "- **打率** — ヒットを打つ確率。.300以上なら一流\n"
            "- **本塁打** — ホームラン数\n"
            "- **打点** — 自分の打撃でホームに返した走者の数\n"
            "- **安打** — ヒット数\n"
            "- **四球** — フォアボールの数。多いほど選球眼が良い\n"
            "- **盗塁** — 走力の指標\n"
            "- **出塁率** — 打席でアウトにならずに塁に出る確率。.380以上なら一流\n"
            "- **長打率** — 1打数あたりの塁打数。二塁打・本塁打が多いほど高い\n"
            "- **OPS** — 出塁率＋長打率。打者の総合打撃力。.800以上なら主力級、.900超はスター"
        ),
        "pitcher_stats_help": (
            "- **防御率** — 9イニング投げたら何点取られるか。2点台なら一流\n"
            "- **勝利** — 勝ち投手になった回数\n"
            "- **奪三振** — 三振を奪った数。多いほど支配力が高い\n"
            "- **投球回** — 投げたイニング数。多いほどスタミナがある\n"
            "- **WHIP** — 1イニングに許した走者数。1.00以下ならエース級"
        ),
        "missing_expander_team": "⚠️ {team}の計算対象外選手 ({n}名)",
        "missing_caption_team": "以下の選手はNPBでの過去3年データがないためMarcel予測の対象外です（リーグ平均の貢献として計算）。",
        "no_data_pa": "{team}の打者データがありません（PA >= 100）",
        "no_data_ip": "{team}の投手データがありません（IP >= 30）",
        "top3_batters": "打者 TOP3（総合打撃力予測）",
        "top3_pitchers": "投手 TOP3（総合投球力予測）",
        "featured_matchup": "注目対決",

        # --- Hitter prediction ---
        "hitter_pred_title": "打者予測（2026年）",
        "search_hint_hitter": "例: 牧、近藤、細川",
        "ops_chart_label": "総合打撃力（OPS）",

        # --- Pitcher prediction ---
        "pitcher_pred_title": "投手予測（2026年）",
        "search_hint_pitcher": "例: 才木、モイネロ、宮城",
        "era_chart_label": "防御率（ERA）",

        # --- VS battle ---
        "vs_title": "VS 対決",
        "player1_label": "プレイヤー1",
        "player2_label": "プレイヤー2",

        # --- Team Wpct ---
        "team_wpct_title": "チーム勝率予測",
        "actual_wpct": "実際の勝率",
        "pred_wpct": "予測勝率",
        "actual_record": "実際の成績",
        "expected_wins": "期待勝数",
        "record_fmt": "{w}勝{l}敗",

        # --- Sabermetrics ---
        "saber_title": "選手の実力指標",
        "search_hint_saber": "例: 近藤、牧",
        "wrc_trend_title": "{player} 打撃力（wRC+）の推移",
        "league_average": "リーグ平均",
        "year_axis": "年度",
        "woba_desc": "打席あたりの得点貢献",
        "wrcplus_desc": "リーグ平均=100の打撃力",
        "wraa_desc": "平均より何点多く稼いだか",

        # --- Rankings ---
        "hitter_rank_title": "打者ランキング（2026予測）",
        "pitcher_rank_title": "投手ランキング（2026予測）",
        "show_n": "表示人数",
        "sort_by": "ソート",
        "sort_ops": "総合打撃力(OPS)",
        "sort_avg": "打率(AVG)",
        "sort_hr": "本塁打(HR)",
        "sort_rbi": "打点(RBI)",
        "sort_era": "防御率(ERA)",
        "sort_whip": "走者許容率(WHIP)",
        "sort_so": "奪三振(SO)",
        "sort_w": "勝利数(W)",

        # --- Standings ---
        "standings_title": "予測順位表",
        "standings_info": (
            "⚠️ **これは統計モデルの自動計算結果です。作者の予想・応援とは無関係です。**\n\n"
            "Marcel法は「過去3年のNPBデータ」だけを見ています。"
            "つまり、**このモデルが知らないことが必ずあります**。\n\n"
            "- **データなし選手**: 新外国人・新人・復帰選手の貢献は計算に含まれていません（wRAA=0として扱い、予測幅で可視化）\n"
            "- **若手の急成長**: 23〜26歳の選手が殻を破るような場合、Marcel法は過去3年の平均に引っ張られ、"
            "実際の成績を大きく下回る予測になることがあります。年齢調整（+0.3%/年）は小さく、急激な成長には追いつきません\n\n"
            "下位に予測されたチームでも、記録のない選手・殻を破りかけている若手次第で、状況は十分に変わりえます。"
        ),
        "standings_2026_title": "2026年 順位予測",
        "standings_2026_caption": "各チームの打者成績予測（得点）と投手成績予測（失点）からピタゴラス勝率で算出",
        "missing_badge": "計算外{n}名",
        "pred_range": "幅: {lo}〜{hi}勝",
        "wpct_prefix": "勝率 ",
        "pred_wins_label": "予測勝数",
        "chart_annotation": "オレンジの縦線 = 計算外選手による予測幅（±1.5勝/人）",
        "missing_expander_all": "⚠️ チームごとの計算対象外選手（新人・新外国人等）— wRAA=0で計算中",
        "missing_expander_content": (
            "**以下の選手はNPBでの過去3年データがないためMarcel予測の対象外です。**\n\n"
            "モデルはこれらの選手を **wRAA=0（リーグ平均と同等の貢献）** として自動的に計算しています。\n\n"
            "- 活躍すれば実際の勝利数はモデルの上限（オレンジ線）を上回る可能性があります\n"
            "- 不振の場合は下限を下回る可能性があります\n"
            "- 計算外選手が多いチームほど、予測幅（グラフのオレンジ縦線）が広くなります"
        ),
        "all_projected": "全員Marcel予測対象 ✅",
        "missing_team_detail": "{n}名 → 予測幅 **±{unc:.0f}勝**: {names}",
        "method_expander": "予測方法の説明",
        "method_content": (
            "- **得点の推定**: チーム所属打者の予測wRAA（打者の得点貢献）を合計し、リーグ平均得点に加算\n"
            "- **失点の推定**: チーム所属投手の予測ERA×投球回÷9でリーグ平均からの超過失点を算出\n"
            "- **勝率の計算**: ピタゴラス勝率（得点^1.72 ÷ (得点^1.72 + 失点^1.72)）\n"
            "- **試合数**: 143試合（NPBレギュラーシーズン）\n"
            "- 選手の予測はMarcel法（過去3年の成績を5:4:3で加重平均し、年齢で調整）に基づく\n\n"
            "**予測幅（信頼区間）の考え方**\n\n"
            "- 計算外選手（新外国人・新人等）はNPBデータ不足のためwRAA=0（リーグ平均貢献）と仮定\n"
            "- 歴史的にNPB外国人選手の初年度wRAAは -15点〜+25点 のばらつきがある\n"
            "- この不確実性を 1人あたり ±1.5勝 に換算（±15点÷10点≒1勝 の野球統計の経験則を適用）\n"
            "- グラフのオレンジ縦線が予測幅。計算外が多いチームほど幅が広く、実際の順位との差が出やすい\n\n"
            "**若手の急成長について（Marcel法の構造的な限界）**\n\n"
            "Marcel法の年齢調整は「27歳基準で±0.3%/年」と非常に小さく、急激な成長は捉えられません。\n"
            "23〜26歳の選手がブレイクするケースでは、過去3年の平均に引き戻されるため実際を大きく下回る予測になります。\n"
            "「殻を破りかけている若手が多いチーム」の実力はモデルが示す数字より高い可能性があります。"
        ),
        "historical_title": "過去の順位表（実績 vs ピタゴラス期待値）",
        "actual_wins_bar": "実際の勝数",
        "expected_wins_bar": "期待勝数",
        "wins_y": "勝数",
        "expected_prefix": "期待",
    },

    "en": {
        # --- Sidebar ---
        "sidebar_title": "NPB Predictions",
        "nav_label": "Navigation",
        "glossary": "Glossary",
        "glossary_ops": "**OPS** — On-base Plus Slugging. Overall offensive effectiveness",
        "glossary_era": "**ERA** — Earned Run Average per 9 innings. Lower is better",
        "glossary_whip": "**WHIP** — Walks + Hits per Inning Pitched. Lower is better",
        "glossary_woba": "**wOBA** — Weighted On-Base Average. Weights each outcome by run value",
        "glossary_wrcplus": "**wRC+** — Weighted Runs Created Plus. League average = 100. 120 = 20% above average",
        "data_source": "Data: [Baseball Data Freak](https://baseball-data.com) / [NPB Official](https://npb.jp)",

        # --- Page names ---
        "page_top": "Home",
        "page_standings": "Predicted Standings",
        "page_hitter": "Batter Predictions",
        "page_pitcher": "Pitcher Predictions",
        "page_hitter_rank": "Batter Rankings",
        "page_pitcher_rank": "Pitcher Rankings",
        "page_vs": "Head-to-Head",
        "page_team_wpct": "Team Win%",
        "page_saber": "Advanced Stats",

        # --- Common ---
        "no_data": "Failed to load data",
        "central_league": "Central League",
        "pacific_league": "Pacific League",
        "year_label": "Year",
        "team_label": "Team",
        "all_years": "All Years",
        "wins_suffix": "W",
        "losses_suffix": "L",
        "wpct_label": "Win%",
        "rs_label": "RS",
        "ra_label": "RA",
        "stat_pred": "Marcel",
        "ai_pred": "ML",
        "search_by_name": "Search by player name (partial match)",
        "search_prompt_btn": "Enter a name or tap a quick-button above",
        "no_player_found": '"{name}" not found',
        "no_data_team_year": "No data for {team} ({year})",
        "search_label": "Search by player name",
        "search_prompt": "Enter a player name",
        "no_match": 'No data found for "{name}"',

        # --- Stat column names ---
        "col_player": "Player",
        "col_avg": "AVG",
        "col_hr": "HR",
        "col_rbi": "RBI",
        "col_h": "H",
        "col_bb": "BB",
        "col_sb": "SB",
        "col_obp": "OBP",
        "col_slg": "SLG",
        "col_era": "ERA",
        "col_w": "W",
        "col_so": "SO",
        "col_ip": "IP",

        # --- Bar / card labels ---
        "bar_hr": "HR",
        "bar_avg": "AVG",
        "bar_obp": "OBP",
        "bar_slg": "SLG",
        "bar_so": "SO",
        "bar_w": "W",
        "bar_ip": "IP",
        "bar_era": "ERA",

        # --- Radar chart categories ---
        "radar_hr": "HR",
        "radar_avg": "AVG",
        "radar_obp": "OBP",
        "radar_slg": "SLG",
        "radar_era": "ERA",
        "radar_so": "SO",
        "radar_ip": "IP",
        "radar_w": "W",

        # --- Player kind ---
        "foreign_player": "Foreign Player",
        "rookie_no_data": "Insufficient Data",
        "wraa_zero_note": "projected as league-average contribution",
        "wraa_zero_inline": "wRAA=0",

        # --- Top page ---
        "top_title": "NPB 2026 Predictions",
        "top_subtitle": "Historical Stats × AI Projections",
        "top_warning": (
            "⚠️ **Note — These are automated statistical model outputs.**\n\n"
            "Marcel method calculates reference values based solely on past 3 years of NPB performance data. "
            "If your favorite team or player ranks low, this reflects only what the model calculated from historical numbers — "
            "it does not represent the author's opinion, support, or wishes.\n\n"
            "**This model cannot capture everything** —— "
            "New foreign players, rookies, and returning players with no NPB history are all treated as 'average' contributions. "
            "These players' performances could significantly change any team's standing. "
            "There will always be uncertainties that only the season itself can reveal.\n\n"
            "Transactions from the 2025–2026 offseason are reflected."
        ),
        "btn_all_top3": "Top 3 Overall",
        "team_batters_title": "{team} Batters (2026 Projections)",
        "team_pitchers_title": "{team} Pitchers (2026 Projections)",
        "batter_pred_caption": "2026 projections based on past 3 years of performance. Decimal values are expected due to statistical modeling.",
        "pitcher_pred_caption": "2026 projections based on past 3 years of performance.",
        "how_to_read": "How to Read the Stats",
        "batter_stats_help": (
            "- **AVG** — Batting average. .300+ is elite\n"
            "- **HR** — Home runs\n"
            "- **RBI** — Runs batted in\n"
            "- **H** — Hits\n"
            "- **BB** — Walks. More = better plate discipline\n"
            "- **SB** — Stolen bases. Measure of speed\n"
            "- **OBP** — On-base percentage. .380+ is elite\n"
            "- **SLG** — Slugging percentage. Higher with extra-base hits\n"
            "- **OPS** — OBP + SLG. Overall offensive value. .800+ = starter, .900+ = star"
        ),
        "pitcher_stats_help": (
            "- **ERA** — Earned runs per 9 innings. Sub-2.00 is elite\n"
            "- **W** — Wins\n"
            "- **SO** — Strikeouts. More = more dominant\n"
            "- **IP** — Innings pitched. More = better stamina\n"
            "- **WHIP** — Baserunners per inning. Under 1.00 is ace-level"
        ),
        "missing_expander_team": "⚠️ {team}: Players Not Projected ({n})",
        "missing_caption_team": "These players lack 3 years of NPB data and are excluded from Marcel projections (treated as league-average contribution).",
        "no_data_pa": "No batter data for {team} (PA ≥ 100)",
        "no_data_ip": "No pitcher data for {team} (IP ≥ 30)",
        "top3_batters": "Top 3 Batters (Overall Batting Projections)",
        "top3_pitchers": "Top 3 Pitchers (Overall Pitching Projections)",
        "featured_matchup": "Featured Matchup",

        # --- Hitter prediction ---
        "hitter_pred_title": "Batter Predictions (2026)",
        "search_hint_hitter": "e.g. Maki, Kondo, Hosokawa",
        "ops_chart_label": "Overall Batting (OPS)",

        # --- Pitcher prediction ---
        "pitcher_pred_title": "Pitcher Predictions (2026)",
        "search_hint_pitcher": "e.g. Saiki, Moineló, Miyagi",
        "era_chart_label": "ERA",

        # --- VS battle ---
        "vs_title": "Head-to-Head",
        "player1_label": "Player 1",
        "player2_label": "Player 2",

        # --- Team Wpct ---
        "team_wpct_title": "Team Win% Prediction",
        "actual_wpct": "Actual Win%",
        "pred_wpct": "Predicted Win%",
        "actual_record": "Actual Record",
        "expected_wins": "Expected Wins",
        "record_fmt": "{w}W {l}L",

        # --- Sabermetrics ---
        "saber_title": "Advanced Metrics",
        "search_hint_saber": "e.g. Kondo, Maki",
        "wrc_trend_title": "{player} wRC+ Trend",
        "league_average": "League Average",
        "year_axis": "Year",
        "woba_desc": "Run value per plate appearance",
        "wrcplus_desc": "Batting vs. league avg (100 = avg)",
        "wraa_desc": "Runs above average",

        # --- Rankings ---
        "hitter_rank_title": "Batter Rankings (2026 Projections)",
        "pitcher_rank_title": "Pitcher Rankings (2026 Projections)",
        "show_n": "Show N players",
        "sort_by": "Sort by",
        "sort_ops": "Overall Batting (OPS)",
        "sort_avg": "Batting Avg (AVG)",
        "sort_hr": "Home Runs (HR)",
        "sort_rbi": "RBI",
        "sort_era": "ERA",
        "sort_whip": "WHIP",
        "sort_so": "Strikeouts (SO)",
        "sort_w": "Wins (W)",

        # --- Standings ---
        "standings_title": "Predicted Standings",
        "standings_info": (
            "⚠️ **These are automated statistical model outputs — not the author's predictions.**\n\n"
            "The Marcel method looks only at the past 3 years of NPB data. "
            "**There are things this model simply cannot know.**\n\n"
            "- **Players with no NPB data**: Contributions of new foreign players, rookies, and returning players "
            "are excluded (set to wRAA=0, visualized as prediction ranges)\n"
            "- **Young player breakouts**: When players aged 23–26 break out, Marcel is anchored to a 3-year average "
            "and will significantly underestimate their actual performance. "
            "The age adjustment (+0.3%/year) is too small to capture rapid growth\n\n"
            "Even lower-ranked teams can move significantly depending on untracked players and breakout youngsters."
        ),
        "standings_2026_title": "2026 Season Projections",
        "standings_2026_caption": "Calculated using Pythagorean Win% from projected runs scored/allowed per team",
        "missing_badge": "{n} not projected",
        "pred_range": "Range: {lo}–{hi}W",
        "wpct_prefix": "Win% ",
        "pred_wins_label": "Projected Wins",
        "chart_annotation": "Orange bars = uncertainty from untracked players (±1.5W/player)",
        "missing_expander_all": "⚠️ Players Not Projected by Team (rookies/new imports) — treated as wRAA=0",
        "missing_expander_content": (
            "**These players lack 3 years of NPB data and are excluded from Marcel projections.**\n\n"
            "The model automatically treats them as **wRAA=0 (league-average contribution)**.\n\n"
            "- If they outperform expectations, actual wins could exceed the upper bound (orange bar)\n"
            "- If they underperform, actual wins could fall below the lower bound\n"
            "- Teams with more untracked players have wider prediction ranges"
        ),
        "all_projected": "All players projected ✅",
        "missing_team_detail": "{n} players → Range **±{unc:.0f}W**: {names}",
        "method_expander": "Methodology",
        "method_content": (
            "- **Runs Scored estimate**: Sum of projected wRAA for each team's batters, added to league-average RS\n"
            "- **Runs Allowed estimate**: Sum of each pitcher's ERA-vs-league × IP/9, added to league-average RA\n"
            "- **Win% calculation**: Pythagorean Win% (RS^1.72 / (RS^1.72 + RA^1.72))\n"
            "- **Games**: 143 (NPB regular season)\n"
            "- Player projections use Marcel method (3-year weighted average with age adjustment)\n\n"
            "**Prediction Range (Uncertainty)**\n\n"
            "- Untracked players (new imports, rookies) are set to wRAA=0 (league-average)\n"
            "- Historically, first-year NPB foreign player wRAA ranges from -15 to +25 runs\n"
            "- Uncertainty converted to ±1.5W per untracked player (±15 runs ÷ 10 runs/win)\n"
            "- Orange bars show prediction range. More untracked players = wider range\n\n"
            "**Young Player Breakouts (Marcel Limitation)**\n\n"
            "Marcel's age adjustment (+0.3%/year from age 27) is very small and cannot capture rapid improvement. "
            "When players aged 23–26 break out, Marcel is anchored to the 3-year average and will significantly "
            "underestimate their performance. Teams with potential breakout youngsters may be stronger than the model suggests."
        ),
        "historical_title": "Historical Standings (Actual vs. Expected Wins)",
        "actual_wins_bar": "Actual Wins",
        "expected_wins_bar": "Expected Wins",
        "wins_y": "Wins",
        "expected_prefix": "Exp.",
    },
}
