"""
NPB成績予測 Streamlitダッシュボード

Marcel法・ピタゴラス勝率・wOBA/wRC+/FIPの予測結果をブラウザで閲覧。

Data sources:
- プロ野球データFreak (https://baseball-data.com)
- 日本野球機構 NPB (https://npb.jp)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from translations import TEXTS


def t(key: str) -> str:
    """Return translated string for the current language."""
    lang = st.session_state.get("lang", "日本語")
    dict_key = "en" if lang == "English" else "ja"
    return TEXTS.get(dict_key, TEXTS["ja"]).get(key, key)


BASE_URL = "https://raw.githubusercontent.com/yasumorishima/npb-prediction/main/"

NPB_TEAM_COLORS = {
    "DeNA": "#0055A5",
    "巨人": "#F97709",
    "阪神": "#FFE201",
    "広島": "#EE1C25",
    "中日": "#00468B",
    "ヤクルト": "#006AB6",
    "ソフトバンク": "#F5C70E",
    "日本ハム": "#004B97",
    "楽天": "#860029",
    "ロッテ": "#000000",
    "オリックス": "#C4A400",
    "西武": "#102A6F",
}

NPB_TEAM_GLOW = {
    "DeNA": "#00aaff",
    "巨人": "#ff9933",
    "阪神": "#ffe44d",
    "広島": "#ff4444",
    "中日": "#4488ff",
    "ヤクルト": "#44aaff",
    "ソフトバンク": "#ffdd33",
    "日本ハム": "#4488ff",
    "楽天": "#cc3366",
    "ロッテ": "#888888",
    "オリックス": "#ddcc33",
    "西武": "#4466cc",
}

TEAMS = list(NPB_TEAM_COLORS.keys())


def _norm_hr(hr: float) -> float:
    return max(0.0, min(100.0, hr / 50.0 * 100.0))


def _norm_avg(avg: float) -> float:
    return max(0.0, min(100.0, (avg - 0.200) / 0.150 * 100.0))


def _norm_obp(obp: float) -> float:
    return max(0.0, min(100.0, (obp - 0.250) / 0.200 * 100.0))


def _norm_slg(slg: float) -> float:
    return max(0.0, min(100.0, (slg - 0.300) / 0.350 * 100.0))


def _norm_ops(ops: float) -> float:
    return max(0.0, min(100.0, (ops - 0.500) / 0.600 * 100.0))


def _norm_era_r(era: float) -> float:
    """ERA → 0-100 (低いほど高スコア: ERA 1.0→100, 5.0→0)"""
    return max(0.0, min(100.0, (5.0 - era) / 4.0 * 100.0))


def _norm_whip_r(whip: float) -> float:
    """WHIP → 0-100 (低いほど高スコア: 0.8→100, 1.6→0)"""
    return max(0.0, min(100.0, (1.6 - whip) / 0.8 * 100.0))


def _norm_so_p(so: float) -> float:
    """投手SO → 0-100 (200K→100)"""
    return max(0.0, min(100.0, so / 200.0 * 100.0))


def _norm_ip(ip: float) -> float:
    """投球回 → 0-100 (200IP→100)"""
    return max(0.0, min(100.0, ip / 200.0 * 100.0))


def _norm_w(w: float) -> float:
    """勝利数 → 0-100 (20W→100)"""
    return max(0.0, min(100.0, w / 20.0 * 100.0))


# --- リーグ平均 ---

# NPBリーグ平均（規定打席以上の代表的な水準）
HITTER_AVG = {"HR": 15, "AVG": 0.260, "OBP": 0.320, "SLG": 0.400, "OPS": 0.720}
PITCHER_AVG = {"ERA": 3.50, "WHIP": 1.30, "SO": 120, "IP": 140, "W": 9}


# --- データ読み込み ---


def _norm(name: str) -> str:
    return name.replace("\u3000", " ").strip()


@st.cache_data(ttl=3600)
def load_csv(path: str) -> pd.DataFrame:
    url = BASE_URL + path
    try:
        df = pd.read_csv(url, encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame()
    if "player" in df.columns:
        df["player"] = df["player"].apply(_norm)
    if "team" in df.columns:
        df["team"] = df["team"].apply(_norm)
    return df


def load_all():
    from roster_2026 import get_all_roster_names, get_team_for_player

    result = {
        "marcel_hitters": load_csv("data/projections/marcel_hitters_2026.csv"),
        "marcel_pitchers": load_csv("data/projections/marcel_pitchers_2026.csv"),
        "sabermetrics": load_csv("data/projections/npb_sabermetrics_2015_2025.csv"),
        "pythagorean": load_csv("data/projections/pythagorean_2015_2025.csv"),
    }
    # NPB公式2026ロースターに在籍する選手のみ残し、チーム名も公式に合わせる
    roster_names = get_all_roster_names()
    for key in ("marcel_hitters", "marcel_pitchers"):
        df = result[key]
        if df.empty or "player" not in df.columns:
            continue
        # ロースターにいる選手だけ残す
        df = df[df["player"].apply(_fuzzy).isin(roster_names)].copy()
        # チーム名を公式ロースターに合わせる（移籍反映）
        for idx, row in df.iterrows():
            new_team = get_team_for_player(row["player"])
            if new_team:
                df.at[idx, "team"] = new_team
        result[key] = df

    _enrich_projections(result)
    return result


def _enrich_projections(data: dict) -> None:
    """打者にwOBA/wRC+/wRAA、投手にFIP/K%/BB%/K-BB%を追加"""
    mh = data["marcel_hitters"]
    mp = data["marcel_pitchers"]
    saber = data.get("sabermetrics", pd.DataFrame())

    # --- 打者: wOBA, wRC+, wRAA ---
    if not mh.empty and not saber.empty:
        df_fit = saber[saber["PA"] >= 100].dropna(subset=["wOBA", "OBP", "SLG"])
        if len(df_fit) >= 10:
            X = np.column_stack([df_fit["OBP"].values, df_fit["SLG"].values, np.ones(len(df_fit))])
            coeffs, _, _, _ = np.linalg.lstsq(X, df_fit["wOBA"].values, rcond=None)
            a_obp, b_slg, intercept_w = coeffs

            recent_s = saber[saber["year"] >= 2022]
            lg_woba = recent_s[recent_s["PA"] >= 50]["wOBA"].mean()
            woba_scale = 1.15

            mh["wOBA"] = (a_obp * mh["OBP"] + b_slg * mh["SLG"] + intercept_w).round(3)
            mh["wRAA"] = ((mh["wOBA"] - lg_woba) / woba_scale * mh["PA"]).round(1)
            lg_r_per_pa = lg_woba / woba_scale
            mh["wRC+"] = (((mh["wOBA"] - lg_woba) / woba_scale + lg_r_per_pa) / lg_r_per_pa * 100).round(0).astype(int)

            data["_lg_woba"] = lg_woba

    # --- 投手: FIP, K%, BB%, K-BB% ---
    if not mp.empty and mp["IP"].sum() > 0:
        has_fip_cols = all(c in mp.columns for c in ["BB", "HBP", "HRA", "BF"])
        if has_fip_cols:
            ip_safe = mp["IP"].replace(0, np.nan)
            bf_safe = mp["BF"].replace(0, np.nan)
            mp["K_pct"] = (mp["SO"] / bf_safe * 100).round(1)
            mp["BB_pct"] = (mp["BB"] / bf_safe * 100).round(1)
            mp["K_BB_pct"] = (mp["K_pct"] - mp["BB_pct"]).round(1)
            mp["K9"] = (mp["SO"] * 9 / ip_safe).round(2)
            mp["BB9"] = (mp["BB"] * 9 / ip_safe).round(2)
            mp["HR9"] = (mp["HRA"] * 9 / ip_safe).round(2)

            lg_ip = mp["IP"].sum()
            lg_era = (mp["ERA"] * mp["IP"]).sum() / lg_ip
            lg_hra = mp["HRA"].sum()
            lg_bb = mp["BB"].sum()
            lg_hbp = mp["HBP"].sum()
            lg_so = mp["SO"].sum()
            fip_c = lg_era - (13 * lg_hra + 3 * (lg_bb + lg_hbp) - 2 * lg_so) / lg_ip
            mp["FIP"] = ((13 * mp["HRA"] + 3 * (mp["BB"] + mp["HBP"]) - 2 * mp["SO"]) / mp["IP"] + fip_c).round(2)


_VARIANT_MAP = str.maketrans("﨑髙濵澤邊齋齊國島嶋櫻", "崎高浜沢辺斎斉国島島桜")


def _fuzzy(s: str) -> str:
    """スペース除去（全角・半角両方） + 異体字を統一"""
    return s.replace(" ", "").replace("\u3000", "").translate(_VARIANT_MAP)


def _is_foreign_player(name: str) -> bool:
    """カタカナ文字が名前の半分超 → 外国人選手と判定"""
    cleaned = name.replace("\u3000", "").replace(" ", "")
    if not cleaned:
        return False
    katakana = sum(1 for c in cleaned if "\u30A0" <= c <= "\u30FF")
    return katakana / len(cleaned) > 0.5


def _data_years_badge(years: int) -> str:
    """data_years が 1 or 2 のときオレンジ/黄バッジHTMLを返す。3以上は空文字。"""
    if years == 1:
        return (
            '<span style="color:#ff9944;font-size:10px;background:#2a1500;'
            f'padding:1px 5px;border-radius:3px;margin-left:5px;">'
            f'{t("data_years_badge").format(n=1)}</span>'
        )
    if years == 2:
        return (
            '<span style="color:#ffcc44;font-size:10px;background:#2a2000;'
            f'padding:1px 5px;border-radius:3px;margin-left:5px;">'
            f'{t("data_years_badge").format(n=2)}</span>'
        )
    return ""


def _get_missing_players(data: dict) -> dict:
    """ロースター登録済みだがMarcel予測対象外の選手をチーム別に返す。
    返り値: {team: [{"name": str, "kind": "外国人" | "新人/データなし"}, ...]}
    """
    from roster_2026 import ROSTER_2026

    mh = data["marcel_hitters"]
    mp = data["marcel_pitchers"]
    if mh.empty or mp.empty:
        return {}
    calculated = (
        set(mh["player"].apply(_fuzzy))
        | set(mp["player"].apply(_fuzzy))
    )
    result = {}
    for team, players in ROSTER_2026.items():
        missing = []
        for p in players:
            if _fuzzy(p) not in calculated:
                kind = "foreign" if _is_foreign_player(p) else "rookie"
                display = p.replace("\u3000", " ").strip()
                missing.append({"name": display, "kind": kind})
        result[team] = missing
    return result


def _search(df: pd.DataFrame, name: str) -> pd.DataFrame:
    q = _fuzzy(_norm(name))
    return df[df["player"].apply(lambda p: q in _fuzzy(p))]


def _pythagorean_wpct(rs: float, ra: float, k: float = 1.72) -> float:
    if ra == 0:
        return 1.0
    return rs**k / (rs**k + ra**k)


# --- HTML/CSSカード描画 ---


def _bar_html(label: str, value: float, max_val: float, display: str, color: str = "#00e5ff",
              avg_val: float | None = None) -> str:
    pct = max(0, min(100, value / max_val * 100))
    avg_marker = ""
    if avg_val is not None:
        avg_pct = max(0, min(100, avg_val / max_val * 100))
        avg_marker = (
            f'<div style="position:absolute;left:{avg_pct:.0f}%;top:0;width:2px;height:100%;'
            f'background:#fff;opacity:0.5;"></div>'
        )
    return f"""
    <div style="display:flex;align-items:center;margin:4px 0;gap:8px;">
      <span style="width:60px;font-size:13px;color:#aaa;">{label}</span>
      <div style="flex:1;height:16px;background:#1a1a2e;border-radius:8px;overflow:hidden;position:relative;">
        <div style="width:{pct:.0f}%;height:100%;background:linear-gradient(90deg,{color},{color}88);border-radius:8px;transition:width 0.5s;"></div>
        {avg_marker}
      </div>
      <span style="width:50px;text-align:right;font-size:13px;font-weight:bold;color:#e0e0e0;">{display}</span>
    </div>"""


def render_hitter_card(row: pd.Series, ml_ops: float | None = None, glow: str = "#00e5ff") -> str:
    """打者ステータスカードをHTMLで生成"""
    team = row.get("team", "")
    dy_badge = _data_years_badge(int(row.get("data_years", 3)))

    ha = HITTER_AVG
    bars = ""
    bars += _bar_html(t("bar_hr"), row["HR"], 50, f"{row['HR']:.0f}", "#ff4466", avg_val=ha["HR"])
    bars += _bar_html(t("bar_avg"), row["AVG"], 0.350, f"{row['AVG']:.3f}", "#44ff88", avg_val=ha["AVG"])
    bars += _bar_html(t("bar_obp"), row["OBP"], 0.450, f"{row['OBP']:.3f}", "#44aaff", avg_val=ha["OBP"])
    bars += _bar_html(t("bar_slg"), row["SLG"], 0.650, f"{row['SLG']:.3f}", "#ffaa44", avg_val=ha["SLG"])
    bars += _bar_html("OPS", row["OPS"], 1.100, f"{row['OPS']:.3f}", "#00e5ff", avg_val=ha["OPS"])

    avg_legend = t("avg_legend")
    return f"""
    <div style="background:linear-gradient(135deg,#0d0d24,#1a1a3a);border:1px solid {glow}44;
                border-radius:12px;padding:16px;margin:8px 0;box-shadow:0 0 15px {glow}22;
                font-family:'Segoe UI',sans-serif;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
        <div>
          <span style="color:#e0e0e0;font-size:18px;font-weight:bold;">{row['player']}</span>{dy_badge}
        </div>
        <span style="color:{glow};font-size:12px;border:1px solid {glow}66;padding:2px 8px;border-radius:4px;">{team}</span>
      </div>
      {bars}
      <div style="text-align:right;margin-top:4px;">
        <span style="color:#888;font-size:10px;">▏{avg_legend}</span>
      </div>
    </div>"""


def render_pitcher_card(row: pd.Series, ml_era: float | None = None, glow: str = "#00e5ff") -> str:
    """投手ステータスカードをHTMLで生成"""
    team = row.get("team", "")
    dy_badge = _data_years_badge(int(row.get("data_years", 3)))

    pa = PITCHER_AVG
    bars = ""
    bars += _bar_html("WHIP", 1.0 / max(row["WHIP"], 0.5), 1.0 / 0.8, f"{row['WHIP']:.2f}", "#44aaff",
                      avg_val=1.0 / pa["WHIP"])
    bars += _bar_html(t("bar_so"), row["SO"], 250, f"{row['SO']:.0f}", "#ff4466", avg_val=pa["SO"])
    bars += _bar_html(t("bar_w"), row["W"], 20, f"{row['W']:.0f}", "#44ff88", avg_val=pa["W"])
    bars += _bar_html(t("bar_ip"), row["IP"], 200, f"{row['IP']:.0f}", "#ffaa44", avg_val=pa["IP"])
    era_pct = max(0, min(100, (6.0 - row["ERA"]) / 5.0 * 100))
    era_avg_pct = max(0, min(100, (6.0 - pa["ERA"]) / 5.0 * 100))
    bars += f"""
    <div style="display:flex;align-items:center;margin:4px 0;gap:8px;">
      <span style="width:60px;font-size:13px;color:#aaa;">{t("bar_era")}</span>
      <div style="flex:1;height:16px;background:#1a1a2e;border-radius:8px;overflow:hidden;position:relative;">
        <div style="width:{era_pct:.0f}%;height:100%;background:linear-gradient(90deg,#00e5ff,#00e5ff88);border-radius:8px;"></div>
        <div style="position:absolute;left:{era_avg_pct:.0f}%;top:0;width:2px;height:100%;background:#fff;opacity:0.5;"></div>
      </div>
      <span style="width:50px;text-align:right;font-size:13px;font-weight:bold;color:#e0e0e0;">{row['ERA']:.2f}</span>
    </div>"""

    avg_legend = t("avg_legend")
    return f"""
    <div style="background:linear-gradient(135deg,#0d0d24,#1a1a3a);border:1px solid {glow}44;
                border-radius:12px;padding:16px;margin:8px 0;box-shadow:0 0 15px {glow}22;
                font-family:'Segoe UI',sans-serif;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
        <div>
          <span style="color:#e0e0e0;font-size:18px;font-weight:bold;">{row['player']}</span>{dy_badge}
        </div>
        <span style="color:{glow};font-size:12px;border:1px solid {glow}66;padding:2px 8px;border-radius:4px;">{team}</span>
      </div>
      {bars}
      <div style="text-align:right;margin-top:4px;">
        <span style="color:#888;font-size:10px;">▏{avg_legend}</span>
      </div>
    </div>"""


def _safe_float(val, default: float = 0.0) -> float:
    try:
        v = float(val)
        return default if pd.isna(v) else v
    except (TypeError, ValueError):
        return default


def render_radar_chart(row: pd.Series, title: str = "", color: str = "#00e5ff") -> go.Figure:
    """打者レーダーチャート（5軸）+ リーグ平均"""
    categories = [t("radar_hr"), t("radar_avg"), t("radar_obp"), t("radar_slg"), "OPS"]
    values = [
        _norm_hr(_safe_float(row["HR"])),
        _norm_avg(_safe_float(row["AVG"])),
        _norm_obp(_safe_float(row["OBP"])),
        _norm_slg(_safe_float(row["SLG"])),
        _norm_ops(_safe_float(row["OPS"])),
    ]
    ha = HITTER_AVG
    avg_values = [
        _norm_hr(ha["HR"]), _norm_avg(ha["AVG"]), _norm_obp(ha["OBP"]),
        _norm_slg(ha["SLG"]), _norm_ops(ha["OPS"]),
    ]

    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=avg_values + [avg_values[0]],
        theta=categories + [categories[0]],
        fill="none",
        line=dict(color="#ffffff", width=1, dash="dash"),
        name=t("league_average"),
    ))
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor=f"rgba({r},{g},{b},0.15)",
        line=dict(color=color, width=2),
        name=title if title else str(row["player"]),
    ))
    layout_kwargs = dict(
        polar=dict(
            bgcolor="#0d0d24",
            radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, gridcolor="#333"),
            angularaxis=dict(gridcolor="#333", linecolor="#444"),
        ),
        showlegend=True,
        legend=dict(orientation="h", y=-0.05, font=dict(size=10, color="#aaa")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
        height=320,
        margin=dict(l=30, r=30, t=30, b=40),
    )
    if title:
        layout_kwargs["title"] = dict(text=title, font=dict(size=14, color="#e0e0e0"))
    fig.update_layout(**layout_kwargs)
    return fig


def render_pitcher_radar_chart(row: pd.Series, title: str = "", color: str = "#00e5ff") -> go.Figure:
    """投手レーダーチャート（5軸: 防御率・WHIP・奪三振・投球回・勝利）+ リーグ平均"""
    categories = [t("radar_era"), "WHIP", t("radar_so"), t("radar_ip"), t("radar_w")]
    values = [
        _norm_era_r(_safe_float(row["ERA"])),
        _norm_whip_r(_safe_float(row["WHIP"])),
        _norm_so_p(_safe_float(row["SO"])),
        _norm_ip(_safe_float(row["IP"])),
        _norm_w(_safe_float(row["W"])),
    ]
    pa = PITCHER_AVG
    avg_values = [
        _norm_era_r(pa["ERA"]), _norm_whip_r(pa["WHIP"]), _norm_so_p(pa["SO"]),
        _norm_ip(pa["IP"]), _norm_w(pa["W"]),
    ]

    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=avg_values + [avg_values[0]],
        theta=categories + [categories[0]],
        fill="none",
        line=dict(color="#ffffff", width=1, dash="dash"),
        name=t("league_average"),
    ))
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor=f"rgba({r},{g},{b},0.15)",
        line=dict(color=color, width=2),
        name=title if title else str(row["player"]),
    ))
    layout_kwargs = dict(
        polar=dict(
            bgcolor="#0d0d24",
            radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, gridcolor="#333"),
            angularaxis=dict(gridcolor="#333", linecolor="#444"),
        ),
        showlegend=True,
        legend=dict(orientation="h", y=-0.05, font=dict(size=10, color="#aaa")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
        height=320,
        margin=dict(l=30, r=30, t=30, b=40),
    )
    if title:
        layout_kwargs["title"] = dict(text=title, font=dict(size=14, color="#e0e0e0"))
    fig.update_layout(**layout_kwargs)
    return fig



# --- ページ実装 ---


CENTRAL_TEAMS = ["DeNA", "巨人", "阪神", "広島", "中日", "ヤクルト"]
PACIFIC_TEAMS = ["ソフトバンク", "日本ハム", "楽天", "ロッテ", "オリックス", "西武"]


def page_top(data: dict):
    """トップページ — 入力不要・1画面完結"""
    st.markdown(f"""
    <div style="text-align:center;padding:10px 0;">
      <h2 style="color:#00e5ff;margin:0;">{t("top_title")}</h2>
      <p style="color:#888;font-size:14px;margin:4px 0;">{t("top_subtitle")}</p>
    </div>
    """, unsafe_allow_html=True)

    st.warning(t("top_warning"))

    mh = data["marcel_hitters"]
    mp = data["marcel_pitchers"]

    if mh.empty or mp.empty:
        st.error(t("no_data"))
        return

    # チーム選択ボタン
    if st.button(t("btn_all_top3"), key="top_reset", type="primary" if not st.session_state.get("selected_team") else "secondary"):
        st.session_state["selected_team"] = None

    st.markdown(f"<div style='color:#888;font-size:12px;margin-bottom:4px;'>{t('central_league')}</div>",
                unsafe_allow_html=True)
    cl_row1 = st.columns(3)
    for i, team in enumerate(CENTRAL_TEAMS[:3]):
        is_selected = st.session_state.get("selected_team") == team
        if cl_row1[i].button(team, key=f"team_{team}",
                             type="primary" if is_selected else "secondary"):
            st.session_state["selected_team"] = team
            st.rerun()
    cl_row2 = st.columns(3)
    for i, team in enumerate(CENTRAL_TEAMS[3:]):
        is_selected = st.session_state.get("selected_team") == team
        if cl_row2[i].button(team, key=f"team_{team}",
                             type="primary" if is_selected else "secondary"):
            st.session_state["selected_team"] = team
            st.rerun()

    st.markdown(f"<div style='color:#888;font-size:12px;margin-bottom:4px;'>{t('pacific_league')}</div>",
                unsafe_allow_html=True)
    pl_row1 = st.columns(3)
    for i, team in enumerate(PACIFIC_TEAMS[:3]):
        is_selected = st.session_state.get("selected_team") == team
        if pl_row1[i].button(team, key=f"team_{team}",
                             type="primary" if is_selected else "secondary"):
            st.session_state["selected_team"] = team
            st.rerun()
    pl_row2 = st.columns(3)
    for i, team in enumerate(PACIFIC_TEAMS[3:]):
        is_selected = st.session_state.get("selected_team") == team
        if pl_row2[i].button(team, key=f"team_{team}",
                             type="primary" if is_selected else "secondary"):
            st.session_state["selected_team"] = team
            st.rerun()

    selected_team = st.session_state.get("selected_team")

    if selected_team:
        # チーム選手一覧表示
        team_glow = NPB_TEAM_GLOW.get(selected_team, "#00e5ff")

        # 打者一覧
        team_hitters = mh[(mh["team"] == selected_team) & (mh["PA"] >= 100)].sort_values("OPS", ascending=False)
        st.markdown(f"### {t('team_batters_title').format(team=selected_team)}")
        st.caption(t("batter_pred_caption"))
        if team_hitters.empty:
            st.info(t("no_data_pa").format(team=selected_team))
        else:
            display_h = team_hitters[["player", "AVG", "HR", "RBI", "H", "BB", "SB", "OBP", "SLG", "OPS"]].copy()
            if "data_years" in team_hitters.columns:
                display_h["注"] = team_hitters["data_years"].apply(
                    lambda v: "⚠️1年" if int(v) == 1 else ("📊2年" if int(v) == 2 else "")
                ).values
            display_h.columns = (
                [t("col_player"), t("col_avg"), t("col_hr"), t("col_rbi"), t("col_h"),
                 t("col_bb"), t("col_sb"), t("col_obp"), t("col_slg"), "OPS", "注"]
                if "注" in display_h.columns else
                [t("col_player"), t("col_avg"), t("col_hr"), t("col_rbi"), t("col_h"),
                 t("col_bb"), t("col_sb"), t("col_obp"), t("col_slg"), "OPS"]
            )
            display_h["打率"] = display_h["打率"].apply(lambda x: f".{int(x*1000):03d}")
            display_h["本塁打"] = display_h["本塁打"].apply(lambda x: f"{x:.0f}")
            display_h["打点"] = display_h["打点"].apply(lambda x: f"{x:.0f}")
            display_h["安打"] = display_h["安打"].apply(lambda x: f"{x:.0f}")
            display_h["四球"] = display_h["四球"].apply(lambda x: f"{x:.0f}")
            display_h["盗塁"] = display_h["盗塁"].apply(lambda x: f"{x:.0f}")
            display_h["出塁率"] = display_h["出塁率"].apply(lambda x: f"{x:.3f}")
            display_h["長打率"] = display_h["長打率"].apply(lambda x: f"{x:.3f}")
            display_h["OPS"] = display_h["OPS"].apply(lambda x: f"{x:.3f}")
            display_h = display_h.reset_index(drop=True)
            display_h.index = display_h.index + 1
            st.dataframe(display_h, use_container_width=True, height=min(400, len(display_h) * 40 + 60))
            if "data_years" in team_hitters.columns and (team_hitters["data_years"] <= 2).any():
                st.caption(t("data_years_legend"))
            with st.expander(t("how_to_read")):
                st.markdown(t("batter_stats_help"))

        # 投手一覧
        team_pitchers = mp[(mp["team"] == selected_team) & (mp["IP"] >= 30)].sort_values("ERA", ascending=True)
        st.markdown(f"### {t('team_pitchers_title').format(team=selected_team)}")
        st.caption(t("pitcher_pred_caption"))
        if team_pitchers.empty:
            st.info(t("no_data_ip").format(team=selected_team))
        else:
            display_p = team_pitchers[["player", "ERA", "W", "SO", "IP", "WHIP"]].copy()
            if "data_years" in team_pitchers.columns:
                display_p["注"] = team_pitchers["data_years"].apply(
                    lambda v: "⚠️1年" if int(v) == 1 else ("📊2年" if int(v) == 2 else "")
                ).values
            display_p.columns = (
                [t("col_player"), t("col_era"), t("col_w"), t("col_so"), t("col_ip"), "WHIP", "注"]
                if "注" in display_p.columns else
                [t("col_player"), t("col_era"), t("col_w"), t("col_so"), t("col_ip"), "WHIP"]
            )
            display_p["防御率"] = display_p["防御率"].apply(lambda x: f"{x:.2f}")
            display_p["勝利"] = display_p["勝利"].apply(lambda x: f"{x:.0f}")
            display_p["奪三振"] = display_p["奪三振"].apply(lambda x: f"{x:.0f}")
            display_p["投球回"] = display_p["投球回"].apply(lambda x: f"{x:.0f}")
            display_p["WHIP"] = display_p["WHIP"].apply(lambda x: f"{x:.2f}")
            display_p = display_p.reset_index(drop=True)
            display_p.index = display_p.index + 1
            st.dataframe(display_p, use_container_width=True, height=min(400, len(display_p) * 40 + 60))
            if "data_years" in team_pitchers.columns and (team_pitchers["data_years"] <= 2).any():
                st.caption(t("data_years_legend"))
            with st.expander(t("how_to_read")):
                st.markdown(t("pitcher_stats_help"))

        # 計算対象外選手
        missing_for_team = _get_missing_players(data).get(selected_team, [])
        if missing_for_team:
            with st.expander(t("missing_expander_team").format(team=selected_team, n=len(missing_for_team))):
                st.caption(t("missing_caption_team"))
                for m in missing_for_team:
                    kind_label = t("foreign_player") if m["kind"] == "foreign" else t("rookie_no_data")
                    st.markdown(f"- **{m['name']}** — {kind_label}（{t('wraa_zero_note')}）")
    else:
        # デフォルト: TOP3表示
        # TOP3 打者
        st.markdown(f"### {t('top3_batters')}")
        top_hitters = mh[mh["PA"] >= 200].nlargest(3, "OPS")

        medals = ["🥇", "🥈", "🥉"]
        for i, (_, row) in enumerate(top_hitters.iterrows()):
            glow = NPB_TEAM_GLOW.get(row["team"], "#00e5ff")
            st.markdown(f"<div style='text-align:center;font-size:24px;'>{medals[i]}</div>",
                        unsafe_allow_html=True)
            components.html(render_hitter_card(row, glow=glow), height=260)
            st.plotly_chart(render_radar_chart(row, title=row["player"], color=glow), use_container_width=True, config={"staticPlot": True})

        # TOP3 投手
        st.markdown(f"### {t('top3_pitchers')}")
        top_pitchers = mp[mp["IP"] >= 100].nsmallest(3, "ERA")

        for i, (_, row) in enumerate(top_pitchers.iterrows()):
            glow = NPB_TEAM_GLOW.get(row["team"], "#00e5ff")
            st.markdown(f"<div style='text-align:center;font-size:24px;'>{medals[i]}</div>",
                        unsafe_allow_html=True)
            components.html(render_pitcher_card(row, glow=glow), height=260)
            st.plotly_chart(render_pitcher_radar_chart(row, title=row["player"], color=glow), use_container_width=True, config={"staticPlot": True})


QUICK_HITTERS = ["牧", "近藤", "サンタナ", "宮崎", "佐藤輝", "細川", "坂倉", "万波"]
QUICK_PITCHERS = ["才木", "モイネロ", "宮城", "戸郷", "東", "高橋宏", "伊藤大", "山下"]


def page_hitter_prediction(data: dict):
    st.markdown(f"### {t('hitter_pred_title')}")

    # クイックボタン（4列×2行）
    st.markdown('<div style="margin-bottom:10px;">', unsafe_allow_html=True)
    qh_row1 = st.columns(4)
    for i, qname in enumerate(QUICK_HITTERS[:4]):
        if qh_row1[i].button(qname, key=f"qh_{qname}"):
            st.session_state["hitter_search"] = qname
    qh_row2 = st.columns(4)
    for i, qname in enumerate(QUICK_HITTERS[4:]):
        if qh_row2[i].button(qname, key=f"qh_{qname}"):
            st.session_state["hitter_search"] = qname
    st.markdown('</div>', unsafe_allow_html=True)

    name = st.text_input(t("search_by_name"), key="hitter_search",
                         placeholder=t("search_hint_hitter"))
    if not name:
        st.info(t("search_prompt_btn"))
        return

    mh = _ensure_hitter_saber(data["marcel_hitters"], data)
    marcel = _search(mh, name)
    if marcel.empty:
        st.warning(t("no_player_found").format(name=name))
        return

    saber = data.get("sabermetrics", pd.DataFrame())

    for _, row in marcel.iterrows():
        glow = NPB_TEAM_GLOW.get(row["team"], "#00e5ff")
        dy = int(row.get("data_years", 3))
        if dy <= 2:
            note_key = "data_years_note_1" if dy == 1 else "data_years_note_2"
            st.warning(t(note_key))

        components.html(render_hitter_card(row, glow=glow), height=280)
        st.plotly_chart(render_radar_chart(row, title=row["player"], color=glow),
                        use_container_width=True)

        # wOBA / wRC+ / wRAA カード（2列+単独行）
        if "wOBA" in row.index and not pd.isna(row.get("wOBA")):
            woba_avg = 0.320
            m1, m2 = st.columns(2)
            m1.metric("wOBA", f"{row['wOBA']:.3f}", delta=f"{row['wOBA'] - woba_avg:+.3f} vs {t('avg_short')}")
            m1.markdown(f"<span style='color:#888;font-size:11px;'>{t('woba_value_desc')}</span>", unsafe_allow_html=True)
            m2.metric("wRC+", f"{int(row['wRC+'])}", delta=f"{int(row['wRC+']) - 100:+d} vs 100")
            m2.markdown(f"<span style='color:#888;font-size:11px;'>{t('wrcplus_value_desc')}</span>", unsafe_allow_html=True)
            st.metric("wRAA", f"{row['wRAA']:+.1f}", delta=f"{t('above_avg') if row['wRAA'] > 0 else t('below_avg')}")
            st.markdown(f"<span style='color:#888;font-size:11px;'>{t('wraa_value_desc')}</span>", unsafe_allow_html=True)

            # 計算式の説明
            with st.expander(t("formula_hitter")):
                st.markdown(t("formula_hitter_content"))

        # wRC+推移グラフ（sabermetrics履歴データから）
        if not saber.empty:
            player_saber = _search(saber, row["player"])
            if len(player_saber) > 1:
                player_name = player_saber.iloc[0]["player"]
                trend = player_saber[player_saber["player"] == player_name].sort_values("year")
                if len(trend) > 1:
                    st.markdown(f"**{t('wrc_trend_title').format(player=player_name)}**")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=trend["year"], y=trend["wRC+"],
                        mode="lines+markers", line=dict(color=glow, width=2),
                        marker=dict(size=8, color=glow),
                    ))
                    fig.add_hline(y=100, line_dash="dash", line_color="#666",
                                  annotation_text=t("league_average"), annotation_font_color="#888")
                    fig.update_layout(
                        height=300, xaxis_title=t("year_axis"), yaxis_title="wRC+",
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#e0e0e0"),
                        xaxis=dict(gridcolor="#222"), yaxis=dict(gridcolor="#222"),
                    )
                    st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True})

        st.markdown("---")


def page_pitcher_prediction(data: dict):
    st.markdown(f"### {t('pitcher_pred_title')}")

    qp_row1 = st.columns(4)
    for i, qname in enumerate(QUICK_PITCHERS[:4]):
        if qp_row1[i].button(qname, key=f"qp_{qname}"):
            st.session_state["pitcher_search"] = qname
    qp_row2 = st.columns(4)
    for i, qname in enumerate(QUICK_PITCHERS[4:]):
        if qp_row2[i].button(qname, key=f"qp_{qname}"):
            st.session_state["pitcher_search"] = qname

    name = st.text_input(t("search_by_name"), key="pitcher_search",
                         placeholder=t("search_hint_pitcher"))
    if not name:
        st.info(t("search_prompt_btn"))
        return

    mp = _ensure_pitcher_saber(data["marcel_pitchers"])
    marcel = _search(mp, name)
    if marcel.empty:
        st.warning(t("no_player_found").format(name=name))
        return

    for _, row in marcel.iterrows():
        glow = NPB_TEAM_GLOW.get(row["team"], "#00e5ff")
        dy = int(row.get("data_years", 3))
        if dy <= 2:
            note_key = "data_years_note_1" if dy == 1 else "data_years_note_2"
            st.warning(t(note_key))

        components.html(render_pitcher_card(row, glow=glow), height=280)
        st.plotly_chart(render_pitcher_radar_chart(row, title=row["player"], color=glow),
                        use_container_width=True)

        # FIP / K% / BB% / K-BB% カード（2列×2行）
        has_fip = "FIP" in row.index and not pd.isna(row.get("FIP"))
        has_k_pct = "K_pct" in row.index and not pd.isna(row.get("K_pct"))
        if has_fip or has_k_pct:
            fip_avg, k_pct_avg, bb_pct_avg = 3.80, 18.0, 8.0
            r1a, r1b = st.columns(2)
            if has_fip:
                fip_delta = row['FIP'] - fip_avg
                r1a.metric("FIP", f"{row['FIP']:.2f}",
                           delta=f"{fip_delta:+.2f} vs {t('avg_short')}", delta_color="inverse")
                r1a.markdown(f"<span style='color:#888;font-size:11px;'>{t('fip_value_desc')}</span>", unsafe_allow_html=True)
            if has_k_pct:
                r1b.metric("K%", f"{row['K_pct']:.1f}%",
                           delta=f"{row['K_pct'] - k_pct_avg:+.1f}% vs {t('avg_short')}")
                r1b.markdown(f"<span style='color:#888;font-size:11px;'>{t('k_pct_desc')}</span>", unsafe_allow_html=True)
                r2a, r2b = st.columns(2)
                r2a.metric("BB%", f"{row['BB_pct']:.1f}%",
                           delta=f"{row['BB_pct'] - bb_pct_avg:+.1f}% vs {t('avg_short')}", delta_color="inverse")
                r2a.markdown(f"<span style='color:#888;font-size:11px;'>{t('bb_pct_desc')}</span>", unsafe_allow_html=True)
                r2b.metric("K-BB%", f"{row['K_BB_pct']:.1f}%",
                           delta=f"{row['K_BB_pct'] - (k_pct_avg - bb_pct_avg):+.1f}% vs {t('avg_short')}")
                r2b.markdown(f"<span style='color:#888;font-size:11px;'>{t('k_bb_pct_desc')}</span>", unsafe_allow_html=True)

        # K/9, BB/9, HR/9（2列+単独行）
        has_k9 = "K9" in row.index and not pd.isna(row.get("K9"))
        if has_k9:
            r3a, r3b = st.columns(2)
            r3a.metric("K/9", f"{row['K9']:.2f}")
            r3a.markdown(f"<span style='color:#888;font-size:11px;'>{t('k9_desc')}</span>", unsafe_allow_html=True)
            r3b.metric("BB/9", f"{row['BB9']:.2f}")
            r3b.markdown(f"<span style='color:#888;font-size:11px;'>{t('bb9_desc')}</span>", unsafe_allow_html=True)
            st.metric("HR/9", f"{row['HR9']:.2f}")
            st.markdown(f"<span style='color:#888;font-size:11px;'>{t('hr9_desc')}</span>", unsafe_allow_html=True)

        # 計算式の説明
        with st.expander(t("formula_pitcher")):
            st.markdown(t("formula_pitcher_content"))

        st.markdown("---")



def page_team_wpct(data: dict):
    st.markdown(f"### {t('team_wpct_title')}")
    pyth = data["pythagorean"]
    if pyth.empty:
        st.error(t("no_data"))
        return

    col1, col2 = st.columns(2)
    team = col1.selectbox(t("team_label"), TEAMS, key="team_wpct")
    year = col2.slider(t("year_label"), 2015, 2025, 2025, key="team_year")

    mask = pyth["team"].str.contains(_norm(team), na=False) & (pyth["year"] == year)
    matched = pyth[mask]
    if matched.empty:
        st.warning(t("no_data_team_year").format(team=team, year=year))
        return

    row = matched.iloc[0]
    glow = NPB_TEAM_GLOW.get(team, "#00e5ff")

    card_html = f"""
    <div style="background:linear-gradient(135deg,#0d0d24,#1a1a3a);border:1px solid {glow}44;
                border-radius:12px;padding:16px;margin:8px 0;box-shadow:0 0 15px {glow}22;
                font-family:'Segoe UI',sans-serif;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
        <span style="color:{glow};font-size:20px;font-weight:bold;">{team}</span>
        <span style="color:#888;font-size:14px;">{year}年</span>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
        <div style="text-align:center;padding:8px;background:#1a1a2e;border-radius:8px;">
          <div style="color:#888;font-size:11px;">{t("actual_wpct")}</div>
          <div style="color:#e0e0e0;font-size:22px;font-weight:bold;">{row['actual_WPCT']:.3f}</div>
        </div>
        <div style="text-align:center;padding:8px;background:#1a1a2e;border-radius:8px;">
          <div style="color:#888;font-size:11px;">{t("pred_wpct")}</div>
          <div style="color:#00e5ff;font-size:22px;font-weight:bold;">{row['pyth_WPCT_npb']:.3f}</div>
        </div>
        <div style="text-align:center;padding:8px;background:#1a1a2e;border-radius:8px;">
          <div style="color:#888;font-size:11px;">{t("actual_record")}</div>
          <div style="color:#e0e0e0;font-size:18px;font-weight:bold;">{t("record_fmt").format(w=int(row['W']), l=int(row['L']))}</div>
        </div>
        <div style="text-align:center;padding:8px;background:#1a1a2e;border-radius:8px;">
          <div style="color:#888;font-size:11px;">{t("expected_wins")}</div>
          <div style="color:#ffaa44;font-size:18px;font-weight:bold;">{row['pyth_W_npb']:.1f}
            <span style="font-size:12px;color:{'#4CAF50' if row['diff_W_npb']>=0 else '#ff4466'};">({row['diff_W_npb']:+.1f})</span>
          </div>
        </div>
      </div>
    </div>"""
    components.html(card_html, height=220)

    fig = go.Figure(data=[
        go.Bar(name=t("rs_label"), x=[f"{t('rs_label')} / {t('ra_label')}"], y=[row["RS"]], marker_color="#4CAF50"),
        go.Bar(name=t("ra_label"), x=[f"{t('rs_label')} / {t('ra_label')}"], y=[row["RA"]], marker_color="#F44336"),
    ])
    fig.update_layout(
        barmode="group", height=300,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
        xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True})


def _leaderboard_card(rank: int, row: pd.Series, stat_key: str, fmt: str, glow: str) -> str:
    """ランキングカード1行"""
    medal = {1: "👑", 2: "🥈", 3: "🥉"}.get(rank, "")
    border_color = {1: "#ffd700", 2: "#c0c0c0", 3: "#cd7f32"}.get(rank, "#333")
    val = row[stat_key]
    dy_badge = _data_years_badge(int(row.get("data_years", 3)))

    return f"""
    <div style="overflow-x:auto;-webkit-overflow-scrolling:touch;">
      <div style="display:flex;align-items:center;gap:8px;padding:8px 12px;margin:4px 0;
                  background:#0d0d24;border:1px solid {border_color}88;border-radius:8px;
                  font-family:'Segoe UI',sans-serif;min-width:max-content;">
        <span style="min-width:24px;font-size:16px;text-align:center;">{medal or rank}</span>
        <span style="flex:1;color:#e0e0e0;font-weight:bold;white-space:nowrap;">{row['player']}{dy_badge}</span>
        <span style="color:#888;font-size:12px;white-space:nowrap;">{row['team']}</span>
        <span style="min-width:50px;text-align:right;color:#00e5ff;font-size:16px;font-weight:bold;">{val:{fmt}}</span>
      </div>
    </div>"""


def _ensure_hitter_saber(mh: pd.DataFrame, data: dict) -> pd.DataFrame:
    """wOBA/wRC+/wRAA列がなければ計算して追加"""
    if "wOBA" in mh.columns:
        return mh
    saber = data.get("sabermetrics", pd.DataFrame())
    if saber.empty:
        return mh
    df_fit = saber[saber["PA"] >= 100].dropna(subset=["wOBA", "OBP", "SLG"])
    if len(df_fit) < 10:
        return mh
    X = np.column_stack([df_fit["OBP"].values, df_fit["SLG"].values, np.ones(len(df_fit))])
    coeffs, _, _, _ = np.linalg.lstsq(X, df_fit["wOBA"].values, rcond=None)
    a, b, c = coeffs
    recent_s = saber[saber["year"] >= 2022]
    lg_woba = recent_s[recent_s["PA"] >= 50]["wOBA"].mean()
    ws = 1.15
    mh["wOBA"] = (a * mh["OBP"] + b * mh["SLG"] + c).round(3)
    mh["wRAA"] = ((mh["wOBA"] - lg_woba) / ws * mh["PA"]).round(1)
    lg_r = lg_woba / ws
    mh["wRC+"] = (((mh["wOBA"] - lg_woba) / ws + lg_r) / lg_r * 100).round(0).astype(int)
    return mh


def _ensure_pitcher_saber(mp: pd.DataFrame) -> pd.DataFrame:
    """FIP/K%/BB%/K-BB%/K9/BB9/HR9列がなければ計算して追加"""
    if "FIP" in mp.columns:
        return mp
    if not all(c in mp.columns for c in ["BB", "HBP", "HRA", "BF"]):
        return mp
    ip_safe = mp["IP"].replace(0, np.nan)
    bf_safe = mp["BF"].replace(0, np.nan)
    mp["K_pct"] = (mp["SO"] / bf_safe * 100).round(1)
    mp["BB_pct"] = (mp["BB"] / bf_safe * 100).round(1)
    mp["K_BB_pct"] = (mp["K_pct"] - mp["BB_pct"]).round(1)
    mp["K9"] = (mp["SO"] * 9 / ip_safe).round(2)
    mp["BB9"] = (mp["BB"] * 9 / ip_safe).round(2)
    mp["HR9"] = (mp["HRA"] * 9 / ip_safe).round(2)
    lg_ip = mp["IP"].sum()
    if lg_ip > 0:
        lg_era = (mp["ERA"] * mp["IP"]).sum() / lg_ip
        fip_c = lg_era - (13 * mp["HRA"].sum() + 3 * (mp["BB"].sum() + mp["HBP"].sum()) - 2 * mp["SO"].sum()) / lg_ip
        mp["FIP"] = ((13 * mp["HRA"] + 3 * (mp["BB"] + mp["HBP"]) - 2 * mp["SO"]) / mp["IP"] + fip_c).round(2)
    return mp


def page_hitter_rankings(data: dict):
    st.markdown(f"### {t('hitter_rank_title')}")
    mh = data["marcel_hitters"]
    if mh.empty:
        st.error(t("no_data"))
        return

    mh = _ensure_hitter_saber(mh, data)

    col1, col2 = st.columns(2)
    top_n = col1.slider(t("show_n"), 5, 50, 20, key="hitter_rank_n")
    sort_labels = {
        t("sort_ops"): "OPS", t("sort_avg"): "AVG",
        t("sort_hr"): "HR", t("sort_rbi"): "RBI",
    }
    if "wOBA" in mh.columns:
        sort_labels[t("sort_woba")] = "wOBA"
        sort_labels[t("sort_wrcplus")] = "wRC+"
    sort_label = col2.selectbox(t("sort_by"), list(sort_labels.keys()), key="hitter_rank_sort")
    sort_by = sort_labels[sort_label]

    df = mh[mh["PA"] >= 200].sort_values(sort_by, ascending=False).head(top_n).reset_index(drop=True)

    fmt_map = {"OPS": ".3f", "AVG": ".3f", "HR": ".0f", "RBI": ".0f", "wOBA": ".3f", "wRC+": ".0f"}
    fmt = fmt_map.get(sort_by, ".3f")

    cards = ""
    for i, (_, row) in enumerate(df.iterrows()):
        glow = NPB_TEAM_GLOW.get(row["team"], "#00e5ff")
        cards += _leaderboard_card(i + 1, row, sort_by, fmt, glow)

    components.html(f"""
    <div style="max-height:600px;overflow-y:auto;padding:4px;">
      {cards}
    </div>""", height=min(650, top_n * 50 + 20))


def page_pitcher_rankings(data: dict):
    st.markdown(f"### {t('pitcher_rank_title')}")
    mp = data["marcel_pitchers"]
    if mp.empty:
        st.error(t("no_data"))
        return

    mp = _ensure_pitcher_saber(mp)

    col1, col2 = st.columns(2)
    top_n = col1.slider(t("show_n"), 5, 50, 20, key="pitcher_rank_n")
    sort_labels = {
        t("sort_era"): "ERA", t("sort_whip"): "WHIP",
        t("sort_so"): "SO", t("sort_w"): "W",
    }
    if "FIP" in mp.columns:
        sort_labels[t("sort_fip")] = "FIP"
        sort_labels[t("sort_k_pct")] = "K_pct"
        sort_labels[t("sort_bb_pct")] = "BB_pct"
        sort_labels[t("sort_k_bb_pct")] = "K_BB_pct"
        sort_labels[t("sort_k9")] = "K9"
        sort_labels[t("sort_bb9")] = "BB9"
        sort_labels[t("sort_hr9")] = "HR9"
    sort_label = col2.selectbox(t("sort_by"), list(sort_labels.keys()), key="pitcher_rank_sort")
    sort_by = sort_labels[sort_label]

    ascending = sort_by in ("ERA", "WHIP", "FIP", "BB_pct", "BB9", "HR9")
    df = mp[mp["IP"] >= 50].sort_values(sort_by, ascending=ascending).head(top_n).reset_index(drop=True)

    fmt_map = {"ERA": ".2f", "WHIP": ".2f", "SO": ".0f", "W": ".0f",
               "FIP": ".2f", "K_pct": ".1f", "BB_pct": ".1f", "K_BB_pct": ".1f",
               "K9": ".2f", "BB9": ".2f", "HR9": ".2f"}
    fmt = fmt_map.get(sort_by, ".2f")

    # 表示ラベル（K_pct → K% のように変換）
    display_suffix = {"K_pct": "%", "BB_pct": "%", "K_BB_pct": "%"}
    suffix = display_suffix.get(sort_by, "")

    cards = ""
    for i, (_, row) in enumerate(df.iterrows()):
        glow = NPB_TEAM_GLOW.get(row["team"], "#00e5ff")
        cards += _leaderboard_card(i + 1, row, sort_by, fmt, glow)

    components.html(f"""
    <div style="max-height:600px;overflow-y:auto;padding:4px;">
      {cards}
    </div>""", height=min(650, top_n * 50 + 20))


def _build_2026_standings(data: dict) -> pd.DataFrame:
    """2026年の予測順位表。

    RS推定: wOBA = a×OBP + b×SLG の回帰 → wRAA → リーグ平均+wRAA合計
    RA推定: (ERA - lgERA) × IP/9 → リーグ平均+超過失点合計
    両方を歴史的リーグ平均にスケーリングして絶対水準を揃える。
    """
    mh = data["marcel_hitters"]
    mp = data["marcel_pitchers"]
    saber = data["sabermetrics"]
    pyth = data["pythagorean"]
    if mh.empty or mp.empty or saber.empty or pyth.empty:
        return pd.DataFrame()

    # 歴史的リーグ平均得点・失点（1チームあたり）
    recent_p = pyth[pyth["year"] >= 2022]
    lg_avg_rs = recent_p.groupby("year")["RS"].mean().mean()
    lg_avg_ra = recent_p.groupby("year")["RA"].mean().mean()

    # Marcel投手全体の加重平均ERA（リーグ基準ERA）
    lg_era = (mp["ERA"] * mp["IP"]).sum() / mp["IP"].sum() if mp["IP"].sum() > 0 else 3.5

    # --- 選手ごとのwOBA・wRAA推定 ---
    mh = mh.copy()
    if "wOBA" in mh.columns and "wRAA" in mh.columns:
        mh["wOBA_est"] = mh["wOBA"]
        mh["wRAA_est"] = mh["wRAA"]
    else:
        # フォールバック: 回帰係数を計算
        df_fit = saber[saber["PA"] >= 100].dropna(subset=["wOBA", "OBP", "SLG"])
        X = np.column_stack([df_fit["OBP"].values, df_fit["SLG"].values, np.ones(len(df_fit))])
        coeffs, _, _, _ = np.linalg.lstsq(X, df_fit["wOBA"].values, rcond=None)
        a_obp, b_slg, intercept_w = coeffs
        recent_s = saber[saber["year"] >= 2022]
        lg_woba = recent_s[recent_s["PA"] >= 50]["wOBA"].mean()
        woba_scale = 1.15
        mh["wOBA_est"] = a_obp * mh["OBP"] + b_slg * mh["SLG"] + intercept_w
        mh["wRAA_est"] = (mh["wOBA_est"] - lg_woba) / woba_scale * mh["PA"]

    mp = mp.copy()
    mp["era_above_avg"] = mp["ERA"] - lg_era  # 正=平均より悪い（失点多い）

    # --- チームごとにRS/RA算出 ---
    # ※ ロースター登録済みだがMarcel対象外の選手（新人・新外国人等）は
    #    wRAA=0（リーグ平均貢献）として暗黙的に扱われる。
    #    実際の戦力との差（過小/過大評価）は missing_count が多いほど不確実。
    missing_all = _get_missing_players(data)
    rows = []
    for team in TEAMS:
        h = mh[mh["team"] == team]
        p = mp[mp["team"] == team]
        rs_raw = lg_avg_rs + (h["wRAA_est"].sum() if not h.empty else 0)
        ra_raw = lg_avg_ra + ((p["era_above_avg"] * p["IP"] / 9.0).sum() if not p.empty else 0)
        league = "CL" if team in CENTRAL_TEAMS else "PL"
        rows.append({
            "league": league, "team": team, "rs_raw": rs_raw, "ra_raw": ra_raw,
            "missing_count": len(missing_all.get(team, [])),
        })

    df = pd.DataFrame(rows)

    # スケーリング: 全12チーム平均をリーグ平均RS/RAに合わせる（選択バイアス除去）
    rs_scale = lg_avg_rs / df["rs_raw"].mean()
    ra_scale = lg_avg_ra / df["ra_raw"].mean()
    df["pred_RS"] = df["rs_raw"] * rs_scale
    df["pred_RA"] = df["ra_raw"] * ra_scale

    df["pred_WPCT"] = df.apply(
        lambda r: _pythagorean_wpct(r["pred_RS"], r["pred_RA"], k=1.72), axis=1
    )
    df["pred_W"] = df["pred_WPCT"] * 143
    df["pred_L"] = 143 - df["pred_W"]

    # 予測幅: 計算外選手1人 ≈ ±1.5勝の不確実性（wRAA±15点 / 10点≒1勝）
    # 根拠: NPB外国人選手の初年度wRAAは-15〜+25点のばらつきがあり、
    #       リーグ平均（wRAA=0）との差が最大±15点程度と仮定。
    _UNC = 1.5  # 1選手あたりの不確実性（勝）
    df["pred_W_low"]  = (df["pred_W"] - df["missing_count"] * _UNC).clip(lower=0)
    df["pred_W_high"] = (df["pred_W"] + df["missing_count"] * _UNC).clip(upper=143)

    return df[["league", "team", "pred_RS", "pred_RA", "pred_WPCT",
               "pred_W", "pred_L", "missing_count", "pred_W_low", "pred_W_high"]]


def page_pythagorean_standings(data: dict):
    st.markdown(f"### {t('standings_title')}")
    st.info(t("standings_info"), icon=None)

    # --- 2026年予測 ---
    standings_2026 = _build_2026_standings(data)
    if not standings_2026.empty:
        st.markdown(f"## {t('standings_2026_title')}")
        st.caption(t("standings_2026_caption"))

        for league, label in [("CL", t("central_league")), ("PL", t("pacific_league"))]:
            lg = standings_2026[standings_2026["league"] == league].sort_values(
                "pred_WPCT", ascending=False).reset_index(drop=True)
            if lg.empty:
                continue

            st.markdown(f"**{label}**")
            cards = ""
            for i, (_, row) in enumerate(lg.iterrows()):
                glow = NPB_TEAM_GLOW.get(row["team"], "#00e5ff")
                rank = i + 1
                medal = {1: "👑", 2: "🥈", 3: "🥉"}.get(rank, "")
                mc = int(row.get("missing_count", 0))
                badge = (
                    f'<span style="color:#ff9944;font-size:11px;background:#2a1500;'
                    f'padding:2px 6px;border-radius:4px;margin-left:4px;">{t("missing_badge").format(n=mc)}</span>'
                    if mc > 0 else ""
                )
                # 計算外選手がいるチームは予測幅（±1.5勝/人）を表示
                if mc > 0:
                    w_lo = int(row.get("pred_W_low", row["pred_W"] - mc * 1.5))
                    w_hi = int(row.get("pred_W_high", row["pred_W"] + mc * 1.5))
                    w_cell = (
                        f'<div style="min-width:110px;display:flex;flex-direction:column;align-items:flex-start;">'
                        f'<span style="color:#00e5ff;font-size:18px;font-weight:bold;">{row["pred_W"]:.0f}{t("wins_suffix")}</span>'
                        f'<span style="color:#ff9944;font-size:10px;">{t("pred_range").format(lo=w_lo, hi=w_hi)}</span>'
                        f'</div>'
                    )
                else:
                    w_cell = f'<span style="color:#00e5ff;font-size:18px;font-weight:bold;min-width:70px;">{row["pred_W"]:.0f}{t("wins_suffix")}</span>'
                cards += f"""
                <div style="overflow-x:auto;-webkit-overflow-scrolling:touch;margin:4px 0;">
                  <div style="display:flex;align-items:center;gap:6px;padding:10px 14px;
                              background:#0d0d24;border-left:4px solid {glow};border-radius:6px;
                              font-family:'Segoe UI',sans-serif;min-width:max-content;">
                    <span style="min-width:24px;font-size:16px;text-align:center;">{medal or rank}</span>
                    <span style="min-width:70px;color:{glow};font-weight:bold;font-size:15px;">{row['team']}</span>
                    {w_cell}
                    <span style="color:#888;font-size:13px;white-space:nowrap;">{row['pred_L']:.0f}{t("losses_suffix")}</span>
                    <span style="color:#aaa;font-size:11px;white-space:nowrap;">{t("wpct_prefix")}{row['pred_WPCT']:.3f}</span>
                    <span style="color:#666;font-size:11px;white-space:nowrap;">{t("rs_label")}{row['pred_RS']:.0f} / {t("ra_label")}{row['pred_RA']:.0f}</span>{badge}
                  </div>
                </div>"""

            components.html(f"<div>{cards}</div>", height=len(lg) * 55 + 10)

            fig = go.Figure()
            err_plus  = (lg["pred_W_high"] - lg["pred_W"]).tolist() if "pred_W_high" in lg else None
            err_minus = (lg["pred_W"] - lg["pred_W_low"]).tolist()  if "pred_W_low"  in lg else None
            teams_reversed = lg["team"].tolist()[::-1]
            wins_reversed = lg["pred_W"].tolist()[::-1]
            colors_reversed = [NPB_TEAM_COLORS.get(t, "#333") for t in teams_reversed]
            err_plus_r = err_plus[::-1] if err_plus else None
            err_minus_r = err_minus[::-1] if err_minus else None
            fig.add_trace(go.Bar(
                name=t("pred_wins_label"), y=teams_reversed, x=wins_reversed,
                orientation="h",
                marker_color=colors_reversed,
                error_x=dict(
                    type="data", array=err_plus_r, arrayminus=err_minus_r,
                    visible=True, color="#ff9944", thickness=2, width=6,
                ),
            ))
            fig.update_layout(
                height=300, xaxis_title=t("pred_wins_label"),
                xaxis_range=[0, max(lg["pred_W_high"] if "pred_W_high" in lg.columns else lg["pred_W"]) * 1.1],
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e0e0e0"),
                margin=dict(l=90),
                xaxis=dict(gridcolor="#222"), yaxis=dict(gridcolor="#222"),
            )
            st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True})
            st.caption(t("chart_annotation"))

        st.info(t("pred_range_brief"))
        with st.expander(t("pred_range_explain_title")):
            st.markdown(t("pred_range_explain"))

        missing_all = _get_missing_players(data)
        with st.expander(t("missing_expander_all")):
            st.markdown(t("missing_expander_content"))
            st.markdown("---")
            for league_code, label in [("CL", t("central_league")), ("PL", t("pacific_league"))]:
                league_teams = CENTRAL_TEAMS if league_code == "CL" else PACIFIC_TEAMS
                st.markdown(f"**{label}**")
                for team in league_teams:
                    missing = missing_all.get(team, [])
                    mc = len(missing)
                    unc = mc * 1.5
                    if not missing:
                        st.markdown(f"- **{team}**: {t('all_projected')}")
                    else:
                        sep = " / " if st.session_state.get("lang") == "English" else "、"
                        names_str = sep.join(
                            f"{m['name']}（{t('foreign_player') if m['kind'] == 'foreign' else t('rookie_no_data')}, {t('wraa_zero_inline')}）"
                            for m in missing
                        )
                        st.markdown(
                            t("missing_team_detail").format(n=mc, unc=unc, names=names_str)
                        )

        with st.expander(t("method_expander")):
            st.markdown(t("method_content"))

    st.markdown("---")
    st.markdown(f"### {t('historical_title')}")
    pyth = data["pythagorean"]
    if pyth.empty:
        st.error(t("no_data"))
        return

    years = sorted(pyth["year"].unique())
    year = st.selectbox(t("year_label"), [int(y) for y in years], index=len(years) - 1, key="pyth_year")
    df = pyth[pyth["year"] == year].copy()

    for league, label in [("CL", t("central_league")), ("PL", t("pacific_league"))]:
        lg = df[df["league"] == league].sort_values("pyth_WPCT_npb", ascending=False).reset_index(drop=True)
        if lg.empty:
            continue

        st.markdown(f"**{label}**")
        cards = ""
        for i, (_, row) in enumerate(lg.iterrows()):
            glow = NPB_TEAM_GLOW.get(row["team"], "#00e5ff")
            rank = i + 1
            medal = {1: "👑", 2: "🥈", 3: "🥉"}.get(rank, "")
            diff = row["diff_W_npb"]
            diff_color = "#4CAF50" if diff >= 0 else "#ff4466"
            cards += f"""
            <div style="overflow-x:auto;-webkit-overflow-scrolling:touch;margin:4px 0;">
              <div style="display:flex;align-items:center;gap:8px;padding:8px 12px;
                          background:#0d0d24;border-left:3px solid {glow};border-radius:6px;
                          font-family:'Segoe UI',sans-serif;min-width:max-content;">
                <span style="min-width:25px;font-size:14px;text-align:center;">{medal or rank}</span>
                <span style="min-width:90px;color:{glow};font-weight:bold;white-space:nowrap;">{row['team']}</span>
                <span style="color:#e0e0e0;white-space:nowrap;">{t("record_fmt").format(w=int(row['W']), l=int(row['L']))}</span>
                <span style="color:#888;font-size:12px;white-space:nowrap;">{row['actual_WPCT']:.3f}</span>
                <span style="color:#00e5ff;font-size:12px;white-space:nowrap;">{t("expected_prefix")}{row['pyth_W_npb']:.1f}{t("wins_suffix")}</span>
                <span style="color:{diff_color};font-size:12px;font-weight:bold;">{diff:+.1f}</span>
              </div>
            </div>"""

        components.html(f"<div>{cards}</div>", height=len(lg) * 50 + 10)

        fig = go.Figure()
        teams_rev = lg["team"].tolist()[::-1]
        fig.add_trace(go.Bar(
            name=t("actual_wins_bar"), y=teams_rev, x=lg["W"].tolist()[::-1],
            orientation="h",
            marker_color=[NPB_TEAM_COLORS.get(tn, "#333") for tn in teams_rev],
        ))
        fig.add_trace(go.Bar(
            name=t("expected_wins_bar"), y=teams_rev, x=lg["pyth_W_npb"].tolist()[::-1],
            orientation="h",
            marker_color="#555",
        ))
        fig.update_layout(
            barmode="group", height=350, xaxis_title=t("wins_y"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0e0"),
            margin=dict(l=90),
            xaxis=dict(gridcolor="#222"), yaxis=dict(gridcolor="#222"),
            legend=dict(orientation="h", y=1.12, font=dict(color="#e0e0e0")),
        )
        st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True})


# --- メイン ---


PAGE_KEYS = [
    "page_top", "page_standings", "page_hitter_rank", "page_pitcher_rank",
    "page_team_wpct", "page_hitter", "page_pitcher",
]

PAGE_FUNCS = {
    "page_top": page_top,
    "page_standings": page_pythagorean_standings,
    "page_hitter": page_hitter_prediction,
    "page_pitcher": page_pitcher_prediction,
    "page_hitter_rank": page_hitter_rankings,
    "page_pitcher_rank": page_pitcher_rankings,
    "page_team_wpct": page_team_wpct,
}


def main():
    st.set_page_config(page_title="NPB成績予測", page_icon="⚾")

    # グローバルCSS
    st.markdown("""
    <style>
    .stApp { font-family: 'Segoe UI', sans-serif; }
    .stRadio > div { gap: 2px; }
    .stRadio label { color: #aaa !important; font-size: 14px !important; }
    div[data-testid="stSidebar"] { background: #0d0d1f; }
    div[data-testid="stSidebar"] .stRadio label:hover { color: #00e5ff !important; }
    h1, h2, h3 { color: #e0e0e0 !important; }
    .stMarkdown a { color: #00e5ff !important; }
    /* Mobile responsive */
    button[kind="secondary"], button[kind="primary"] { min-height: 44px !important; }
    .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
    table, .stDataFrame { overflow-x: auto !important; }
    </style>
    """, unsafe_allow_html=True)

    # Language toggle FIRST — must precede any t() call
    st.sidebar.markdown(
        '<div style="text-align:center;padding:8px 0 4px;font-size:13px;color:#00e5ff;font-weight:bold;">🌐 Language / 言語</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.radio("Language / 言語", ["日本語", "English"], key="lang", horizontal=True, label_visibility="collapsed")

    st.sidebar.markdown(f"""
    <div style="text-align:center;padding:10px 0;">
      <div style="font-size:28px;">⚾</div>
      <div style="color:#00e5ff;font-size:16px;font-weight:bold;">{t("sidebar_title")}</div>
      <div style="color:#666;font-size:11px;">2026 Season</div>
    </div>
    """, unsafe_allow_html=True)

    data = load_all()

    page_key = st.sidebar.radio(
        t("nav_label"),
        PAGE_KEYS,
        format_func=t,
        key="page_key",
    )

    with st.sidebar.expander(t("glossary")):
        st.markdown(
            f"- {t('glossary_ops')}\n"
            f"- {t('glossary_era')}\n"
            f"- {t('glossary_whip')}\n"
            f"- {t('glossary_woba')}\n"
            f"- {t('glossary_wrcplus')}"
        )

    st.caption(t("data_source"))

    PAGE_FUNCS[page_key](data)


if __name__ == "__main__":
    main()
