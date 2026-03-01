"""Translation strings for NPB Prediction dashboard (Japanese / English)."""

TEXTS: dict[str, dict[str, str]] = {
    "ja": {
        # --- Sidebar ---
        "sidebar_title": "NPB予測",
        "nav_label": "ページ選択",
        "glossary": "用語の説明",
        "top_glossary_batters": "打者指標",
        "top_glossary_pitchers": "投手指標",
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
            "**球場補正（パークファクター）は考慮していません** —— "
            "2026年のバンテリンドームへのホームランテラス設置など、球場改修による影響は反映されていません。"
            "球場ごとの打者有利・投手有利の差も未補正です。\n\n"
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
        "top3_batters": "打者 TOP3（wRC+ 順）",
        "top3_batters_caption": "wRC+（Weighted Runs Created Plus）= リーグ平均を100とした打撃力。高いほど優秀",
        "top3_pitchers": "先発投手 TOP3（FIP 順 / 投球回 100以上）",
        "top3_pitchers_caption": "FIP（Fielding Independent Pitching）= 守備の影響を除いた防御率。被本塁打・四球・奪三振のみで算出",
        "top3_relievers": "主にリリーフ TOP3（FIP 順 / 投球回 20〜99）",
        "top3_relievers_caption": "投球回（IP）20〜99 の投手を対象にFIP順で表示。※ 怪我等で投球回が少ない先発投手も含まれる場合があります",
        "reliever_rank_title": "主にリリーフのランキング（投球回 20〜99）",
        "reliever_rank_caption": "投球回（IP）20〜99 の投手が対象。※ 怪我等で投球回が少ない先発投手も含まれる場合があります",
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
        "team_wpct_title": "チーム勝率分析",
        "actual_wpct": "実際の勝率",
        "pred_wpct": "ピタゴラス期待勝率（実績RS/RA）",
        "actual_record": "実際の成績",
        "expected_wins": "ピタゴラス期待勝数",
        "record_fmt": "{w}勝{l}敗",

        # --- Sabermetrics ---
        "saber_title": "選手の実力指標",
        "search_hint_saber": "例: 近藤、牧",
        "wrc_trend_title": "{player} 打撃力（wRC+）の推移",
        "league_average": "リーグ平均",
        "avg_legend": "白線 = リーグ平均",
        "avg_short": "平均",
        "above_avg": "平均以上",
        "below_avg": "平均以下",
        "year_axis": "年度",
        "woba_desc": "打席あたりの得点貢献",
        "wrcplus_desc": "リーグ平均=100の打撃力",
        "wraa_desc": "平均より何点多く稼いだか",
        "woba_value_desc": "wOBA — 打席あたりの得点への貢献度。.350超なら一流",
        "wrcplus_value_desc": "wRC+ — リーグ平均を100とした打撃力。120ならリーグ平均の2割増し",
        "wraa_value_desc": "wRAA — リーグ平均の打者より何点多く稼いだか",
        "fip_value_desc": "FIP — 被本塁打・四球・三振だけで評価した防御率。味方の守備に左右されない",
        "k_pct_desc": "K% — 対戦打者のうち三振を取った割合。20%超なら優秀",
        "bb_pct_desc": "BB% — 対戦打者のうち四球を出した割合。7%未満なら優秀",
        "k_bb_pct_desc": "K-BB% — 三振率から四球率を引いた値。15%超ならエース級",
        "k9_desc": "K/9 — 9イニングあたり奪三振数。9.0超なら奪三振マシン",
        "bb9_desc": "BB/9 — 9イニングあたり与四球数。2.5未満なら制球力◎",
        "hr9_desc": "HR/9 — 9イニングあたり被本塁打数。0.8未満なら優秀",
        "formula_hitter": "計算式の説明",
        "formula_hitter_content": (
            "**wOBA（加重出塁率）**\n\n"
            "```\n"
            "wOBA = (0.69×四球 + 0.73×死球 + 0.89×単打\n"
            "      + 1.27×二塁打 + 1.62×三塁打 + 2.10×本塁打)\n"
            "      ÷ 打席数\n"
            "```\n"
            "各打撃結果を「得点への貢献度」で重みづけした出塁率。\n"
            "ホームランは四球の約3倍の価値。**.350超なら一流、.400超ならMVP級**。\n\n"
            "---\n"
            "**wRC+（加重得点創出力+）**\n\n"
            "```\n"
            "wRC+ = ((wOBA − リーグ平均wOBA) ÷ スケール\n"
            "       + リーグ平均得点率)\n"
            "       ÷ リーグ平均得点率 × 100\n"
            "```\n"
            "リーグ平均 = **100**。**120なら平均の2割増し、80なら2割減**。\n"
            "球場やリーグの違いを補正できるため、異なる環境の打者を比較可能。\n\n"
            "---\n"
            "**wRAA（平均比得点貢献）**\n\n"
            "```\n"
            "wRAA = (wOBA − リーグ平均wOBA) ÷ スケール × 打席数\n"
            "```\n"
            "リーグ平均の打者と比べて**何点多く稼いだか**。\n"
            "プラスなら平均以上、マイナスなら平均以下。**+20なら主力、+40ならMVP候補**。"
        ),
        "formula_pitcher": "計算式の説明",
        "formula_pitcher_content": (
            "**FIP（守備から独立した防御率）**\n\n"
            "```\n"
            "FIP = (13×被本塁打 + 3×(四球+死球) − 2×奪三振)\n"
            "      ÷ 投球回 + 定数\n"
            "```\n"
            "味方の守備や運に左右されない「投手自身の実力」を測る指標。\n"
            "本塁打（×13）のペナルティが最も重く、三振（×2）で相殺する構造。\n"
            "**ERAとFIPの差が大きい投手は、翌年ERAがFIPに近づく傾向あり**。\n\n"
            "---\n"
            "**K%（三振率）/ BB%（四球率）/ K-BB%**\n\n"
            "```\n"
            "K%    = 奪三振 ÷ 対戦打者数 × 100\n"
            "BB%   = 与四球 ÷ 対戦打者数 × 100\n"
            "K-BB% = K% − BB%\n"
            "```\n"
            "K%が高く、BB%が低いほど優秀。**K-BB%が15%超ならエース級**。\n"
            "K/9（9回あたり三振数）と違い、対戦打者数ベースなので投球効率を正確に反映。\n\n"
            "---\n"
            "**K/9・BB/9・HR/9**\n\n"
            "```\n"
            "K/9  = 奪三振 × 9 ÷ 投球回\n"
            "BB/9 = 与四球 × 9 ÷ 投球回\n"
            "HR/9 = 被本塁打 × 9 ÷ 投球回\n"
            "```\n"
            "9イニングあたりの三振・四球・被本塁打。\n"
            "**K/9 9.0超 = 奪三振マシン、BB/9 2.5未満 = 制球力◎、HR/9 0.8未満 = 被弾少**。"
        ),

        # --- Rankings ---
        "hitter_rank_title": "打者ランキング（2026予測）",
        "pitcher_rank_title": "投手ランキング（2026予測）",
        "show_n": "表示人数",
        "sort_by": "ソート",
        "sort_ops": "OPS — 出塁率+長打率",
        "sort_avg": "打率 — ヒットの確率",
        "sort_hr": "本塁打 — ホームラン数",
        "sort_rbi": "打点 — 走者を返した数",
        "sort_woba": "wOBA — 打席あたりの得点貢献度",
        "sort_wrcplus": "wRC+ — リーグ平均=100の打撃力",
        "sort_era": "防御率 — 9回あたり失点",
        "sort_whip": "WHIP — 1回あたり出した走者数",
        "sort_so": "奪三振 — 三振を取った数",
        "sort_w": "勝利数",
        "sort_fip": "FIP — 投手の真の実力",
        "sort_k_pct": "K% — 三振を取る割合",
        "sort_bb_pct": "BB% — 四球を出す割合",
        "sort_k_bb_pct": "K-BB% — 三振率−四球率",
        "sort_k9": "K/9 — 9回あたり奪三振",
        "sort_bb9": "BB/9 — 9回あたり与四球",
        "sort_hr9": "HR/9 — 9回あたり被本塁打",

        # --- Standings ---
        "standings_title": "予測順位表",
        "standings_info": (
            "⚠️ **これは統計モデルの自動計算結果です。作者の予想・応援とは無関係です。**\n\n"
            "Marcel法は「過去3年のNPBデータ」だけを見ています。"
            "つまり、**このモデルが知らないことが必ずあります**。\n\n"
            "- **データなし選手**: 新外国人・新人・復帰選手の貢献は計算に含まれていません（wRAA=0として扱い、予測幅で可視化）\n"
            "- **NPB 1〜2年目選手**: データはあるが少ないため予測値がリーグ平均に強く補正されます。"
            "実力の過小/過大評価が起きやすく、特に2年目外国人選手の移籍初年度実績は参考程度にしてください（選手名横の「NPB1年/2年」バッジで確認できます）\n"
            "- **若手の急成長**: 23〜26歳の選手が殻を破るような場合、Marcel法は過去3年の平均に引っ張られ、"
            "実際の成績を大きく下回る予測になることがあります。年齢調整（+0.3%/年）は小さく、急激な成長には追いつきません\n\n"
            "下位に予測されたチームでも、記録のない選手・殻を破りかけている若手次第で、状況は十分に変わりえます。"
        ),
        "standings_2026_title": "2026年 順位予測",
        "standings_2026_caption": "各チームの打者成績予測（得点）と投手成績予測（失点）からピタゴラス勝率で算出",
        "missing_badge": "計算外{n}名",
        "data_years_badge": "NPB{n}年",
        "data_years_note_1": "⚠️ 直近3年のうちNPBデータが1年のみのため、予測値はリーグ平均に強く補正されています（約2/3がリーグ平均寄り）。実力の過小/過大評価に注意してください。",
        "data_years_note_2": "📊 直近3年のうちNPBデータが2年のみのため、予測値はリーグ平均にやや補正されています。参考値としてご覧ください。",
        "data_years_legend": "注欄の見方：⚠️直近1年のみ＝直近3年中1年分のみデータあり（予測値はリーグ平均寄りになりやすい）　📊直近2年のみ＝直近3年中2年分のみデータあり（やや補正あり）",
        "pred_range": "幅: {lo}〜{hi}勝",
        "wpct_prefix": "勝率 ",
        "pred_wins_label": "予測勝数",
        "chart_annotation": "オレンジの縦線 = 計算外選手による予測幅（±1.5勝/人）",
        "pred_range_brief": "オレンジの縦線 = 予測幅。新外国人・新人など予測できない選手が多いチームほど幅が広くなります（1人あたり±1.5勝）",
        "pred_range_explain_title": "予測幅（オレンジの縦線）の詳しい説明",
        "pred_range_explain": (
            "新外国人・新人などNPBデータが3年未満の選手は、Marcel法では予測できません。"
            "これらの選手は「リーグ平均の貢献（wRAA=0）」として計算していますが、"
            "実際の成績は大きくばらつきます。\n\n"
            "- 歴史的にNPB外国人選手の初年度wRAAは **-15点〜+25点** のばらつき\n"
            "- 野球統計の経験則: **10点の得失点差 ≈ 1勝**\n"
            "- → 計算外選手 **1人あたり ±1.5勝** の不確実性\n\n"
            "例: 計算外3名のチーム → 予測70勝なら「67〜74勝」の範囲になります"
        ),
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
        "historical_title": "過去の順位表（実績 vs Marcel予測）",
        "actual_wins_bar": "実際の勝数",
        "expected_wins_bar": "Marcel予測勝数",
        "pyth_wins_bar": "ピタゴラス期待値（実績RS/RA）",
        "wins_y": "勝数",
        "expected_prefix": "Marcel予測",
        "pyth_prefix": "期待値",
    },

    "en": {
        # --- Sidebar ---
        "sidebar_title": "NPB Predictions",
        "nav_label": "Navigation",
        "glossary": "Glossary",
        "top_glossary_batters": "Batting Stats",
        "top_glossary_pitchers": "Pitching Stats",
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
            "**Park factors are not accounted for** —— "
            "Changes such as the 2026 home run terrace addition at Vantelin Dome Nagoya are not reflected. "
            "Batter-friendly vs. pitcher-friendly park differences are also unadjusted.\n\n"
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
        "top3_batters": "Top 3 Batters (by wRC+)",
        "top3_batters_caption": "wRC+ (Weighted Runs Created Plus) = Batting performance normalized to league average of 100. Higher is better",
        "top3_pitchers": "Top 3 Starters (by FIP / IP ≥ 100)",
        "top3_pitchers_caption": "FIP (Fielding Independent Pitching) = Defense-independent ERA based solely on HR, BB, and K",
        "top3_relievers": "Mainly Relievers TOP3 (by FIP / IP 20–99)",
        "top3_relievers_caption": "Pitchers with IP 20–99 ranked by FIP. ※ Starters with limited innings may also appear.",
        "reliever_rank_title": "Mainly Relievers Rankings (IP 20–99)",
        "reliever_rank_caption": "Pitchers with IP 20–99. ※ Starters with limited innings may also appear.",
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
        "team_wpct_title": "Team Win% Analysis",
        "actual_wpct": "Actual Win%",
        "pred_wpct": "Pythagorean Expected Win% (Actual RS/RA)",
        "actual_record": "Actual Record",
        "expected_wins": "Pythagorean Expected Wins",
        "record_fmt": "{w}W {l}L",

        # --- Sabermetrics ---
        "saber_title": "Advanced Metrics",
        "search_hint_saber": "e.g. Kondo, Maki",
        "wrc_trend_title": "{player} wRC+ Trend",
        "league_average": "League Average",
        "avg_legend": "White line = League Avg",
        "avg_short": "Avg",
        "above_avg": "Above Avg",
        "below_avg": "Below Avg",
        "year_axis": "Year",
        "woba_desc": "Run value per plate appearance",
        "wrcplus_desc": "Batting vs. league avg (100 = avg)",
        "wraa_desc": "Runs above average",
        "woba_value_desc": "wOBA — Run value per plate appearance. .350+ is elite",
        "wrcplus_value_desc": "wRC+ — League average = 100. 120 means 20% above average",
        "wraa_value_desc": "wRAA — Runs above league-average batter",
        "fip_value_desc": "FIP — ERA based only on HR, BB, K. Defense-independent",
        "k_pct_desc": "K% — Percentage of batters struck out. 20%+ is excellent",
        "bb_pct_desc": "BB% — Percentage of batters walked. Under 7% is excellent",
        "k_bb_pct_desc": "K-BB% — Strikeout rate minus walk rate. 15%+ is ace-level",
        "k9_desc": "K/9 — Strikeouts per 9 innings. 9.0+ is elite",
        "bb9_desc": "BB/9 — Walks per 9 innings. Under 2.5 is excellent",
        "hr9_desc": "HR/9 — Home runs per 9 innings. Under 0.8 is excellent",
        "formula_hitter": "How These Stats Work",
        "formula_hitter_content": (
            "**wOBA (Weighted On-Base Average)**\n\n"
            "```\n"
            "wOBA = (0.69×BB + 0.73×HBP + 0.89×1B\n"
            "      + 1.27×2B + 1.62×3B + 2.10×HR)\n"
            "      ÷ PA\n"
            "```\n"
            "Weights each outcome by run value. A HR is worth ~3× a walk.\n"
            "**.350+ is elite, .400+ is MVP-caliber.**\n\n"
            "---\n"
            "**wRC+ (Weighted Runs Created Plus)**\n\n"
            "```\n"
            "wRC+ = ((wOBA − lgwOBA) ÷ Scale\n"
            "       + lgRunsPerPA)\n"
            "       ÷ lgRunsPerPA × 100\n"
            "```\n"
            "League average = **100**. **120 = 20% above avg, 80 = 20% below.**\n"
            "Adjusts for league/park, allowing cross-era comparison.\n\n"
            "---\n"
            "**wRAA (Weighted Runs Above Average)**\n\n"
            "```\n"
            "wRAA = (wOBA − lgwOBA) ÷ Scale × PA\n"
            "```\n"
            "Runs contributed above a league-average batter.\n"
            "**+20 = solid regular, +40 = MVP candidate.**"
        ),
        "formula_pitcher": "How These Stats Work",
        "formula_pitcher_content": (
            "**FIP (Fielding Independent Pitching)**\n\n"
            "```\n"
            "FIP = (13×HR + 3×(BB+HBP) − 2×K)\n"
            "      ÷ IP + constant\n"
            "```\n"
            "Measures pitcher skill independent of defense and luck.\n"
            "HR penalty (×13) is heaviest; strikeouts (×2) offset it.\n"
            "**Large ERA−FIP gap → ERA likely regresses toward FIP next year.**\n\n"
            "---\n"
            "**K% / BB% / K-BB%**\n\n"
            "```\n"
            "K%    = K ÷ BF × 100\n"
            "BB%   = BB ÷ BF × 100\n"
            "K-BB% = K% − BB%\n"
            "```\n"
            "High K%, low BB% = dominant. **K-BB% 15%+ = ace-level.**\n"
            "More accurate than K/9 since it's per batter faced, not per inning.\n\n"
            "---\n"
            "**K/9 · BB/9 · HR/9**\n\n"
            "```\n"
            "K/9  = K × 9 ÷ IP\n"
            "BB/9 = BB × 9 ÷ IP\n"
            "HR/9 = HR × 9 ÷ IP\n"
            "```\n"
            "Per-9-inning rates.\n"
            "**K/9 9.0+ = strikeout machine, BB/9 < 2.5 = pinpoint control, HR/9 < 0.8 = stingy.**"
        ),

        # --- Rankings ---
        "hitter_rank_title": "Batter Rankings (2026 Projections)",
        "pitcher_rank_title": "Pitcher Rankings (2026 Projections)",
        "show_n": "Show N players",
        "sort_by": "Sort by",
        "sort_ops": "OPS — On-base + Slugging",
        "sort_avg": "AVG — Batting Average",
        "sort_hr": "HR — Home Runs",
        "sort_rbi": "RBI — Runs Batted In",
        "sort_woba": "wOBA — Weighted On-Base Average",
        "sort_wrcplus": "wRC+ — Batting vs. League Avg (100=avg)",
        "sort_era": "ERA — Earned Runs per 9 Inn.",
        "sort_whip": "WHIP — Baserunners per Inning",
        "sort_so": "SO — Strikeouts",
        "sort_w": "W — Wins",
        "sort_fip": "FIP — Fielding Independent Pitching",
        "sort_k_pct": "K% — Strikeout Rate",
        "sort_bb_pct": "BB% — Walk Rate",
        "sort_k_bb_pct": "K-BB% — Strikeout minus Walk Rate",
        "sort_k9": "K/9 — Strikeouts per 9 Inn.",
        "sort_bb9": "BB/9 — Walks per 9 Inn.",
        "sort_hr9": "HR/9 — Home Runs per 9 Inn.",

        # --- Standings ---
        "standings_title": "Predicted Standings",
        "standings_info": (
            "⚠️ **These are automated statistical model outputs — not the author's predictions.**\n\n"
            "The Marcel method looks only at the past 3 years of NPB data. "
            "**There are things this model simply cannot know.**\n\n"
            "- **Players with no NPB data**: Contributions of new foreign players, rookies, and returning players "
            "are excluded (set to wRAA=0, visualized as prediction ranges)\n"
            "- **Players with 1–2 years of NPB data**: These players appear in projections but their stats are "
            "heavily anchored to league average due to limited data. Projections may under- or over-estimate actual ability. "
            "Look for the 'NPB1yr / NPB2yr' badge next to player names\n"
            "- **Young player breakouts**: When players aged 23–26 break out, Marcel is anchored to a 3-year average "
            "and will significantly underestimate their actual performance. "
            "The age adjustment (+0.3%/year) is too small to capture rapid growth\n\n"
            "Even lower-ranked teams can move significantly depending on untracked players and breakout youngsters."
        ),
        "standings_2026_title": "2026 Season Projections",
        "standings_2026_caption": "Calculated using Pythagorean Win% from projected runs scored/allowed per team",
        "missing_badge": "{n} not projected",
        "data_years_badge": "{n}yr NPB",
        "data_years_note_1": "⚠️ Only 1 of the last 3 seasons has NPB data — projection is heavily anchored to league average (~2/3 regression). May under- or over-estimate actual ability.",
        "data_years_note_2": "📊 Only 2 of the last 3 seasons have NPB data — projection is moderately anchored to league average. Treat as a rough estimate.",
        "data_years_legend": "Column guide: ⚠️last 1yr only = data in only 1 of last 3 seasons (projection anchored to league avg)　📊last 2yrs only = data in only 2 of last 3 seasons (moderate regression)",
        "pred_range": "Range: {lo}–{hi}W",
        "wpct_prefix": "Win% ",
        "pred_wins_label": "Projected Wins",
        "chart_annotation": "Orange bars = uncertainty from untracked players (±1.5W/player)",
        "pred_range_brief": "Orange bars = prediction range. Teams with more untracked players (new imports/rookies) have wider ranges (±1.5W per player)",
        "pred_range_explain_title": "How prediction ranges (orange bars) work",
        "pred_range_explain": (
            "New foreign players, rookies, and others with less than 3 years of NPB data cannot be projected by Marcel. "
            "These players are treated as \"league-average contributors (wRAA=0)\", but their actual performance varies widely.\n\n"
            "- Historically, first-year NPB foreign players show wRAA ranging from roughly **-15 to +25 runs**\n"
            "- Baseball rule of thumb: **10 runs ≈ 1 win**\n"
            "- → Each untracked player adds **±1.5 wins** of uncertainty\n\n"
            "Example: 3 untracked players, 70 projected wins → displayed as a range of 67–74 wins"
        ),
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
        "historical_title": "Historical Standings (Actual vs. Marcel Prediction)",
        "actual_wins_bar": "Actual Wins",
        "expected_wins_bar": "Marcel Predicted Wins",
        "pyth_wins_bar": "Pythagorean Expected (Actual RS/RA)",
        "wins_y": "Wins",
        "expected_prefix": "Marcel",
        "pyth_prefix": "Expected",
    },
}
