"""
NPB成績予測 Streamlitダッシュボード — RPG風UI

Marcel法・LightGBM/XGBoost・ピタゴラス勝率・wOBA/wRC+の予測結果をブラウザで閲覧。

Data sources:
- プロ野球データFreak (https://baseball-data.com)
- 日本野球機構 NPB (https://npb.jp)
"""

import random

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


BASE_URL = "https://raw.githubusercontent.com/yasumorishima/npb-prediction/master/"

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

# --- RPG変換ロジック ---


def wrc_plus_to_level(wrc_plus: float) -> int:
    """wRC+ → RPG Lv変換。100(平均)=Lv.50, 200=Lv.99, 50=Lv.25"""
    return max(1, min(99, int(wrc_plus * 0.495)))


def ops_to_level(ops: float) -> int:
    """OPS → RPG Lv変換。.700(平均)=Lv.50, 1.100=Lv.99"""
    return max(1, min(99, int((ops - 0.3) * 123.75)))


def era_to_level(era: float) -> int:
    """ERA → RPG Lv変換。低いほど高レベル。1.0=Lv.99, 4.0=Lv.50"""
    return max(1, min(99, int(100 - era * 16.5)))


def lv_to_stars(lv: int) -> str:
    """Lv 1–99 → 星5段階表示 (例: ★★★★☆)。平均(Lv.50)=3星"""
    n = max(1, min(5, (lv - 1) // 20 + 1))
    return "★" * n + "☆" * (5 - n)


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
        "ml_hitters": load_csv("data/projections/ml_hitters_2026.csv"),
        "ml_pitchers": load_csv("data/projections/ml_pitchers_2026.csv"),
        "sabermetrics": load_csv("data/projections/npb_sabermetrics_2015_2025.csv"),
        "pythagorean": load_csv("data/projections/pythagorean_2015_2025.csv"),
    }
    # NPB公式2026ロースターに在籍する選手のみ残し、チーム名も公式に合わせる
    roster_names = get_all_roster_names()
    for key in ("marcel_hitters", "marcel_pitchers", "ml_hitters", "ml_pitchers"):
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
    return result


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


def _bar_html(label: str, value: float, max_val: float, display: str, color: str = "#00e5ff") -> str:
    pct = max(0, min(100, value / max_val * 100))
    return f"""
    <div style="display:flex;align-items:center;margin:4px 0;gap:8px;">
      <span style="width:60px;font-size:13px;color:#aaa;">{label}</span>
      <div style="flex:1;height:16px;background:#1a1a2e;border-radius:8px;overflow:hidden;">
        <div style="width:{pct:.0f}%;height:100%;background:linear-gradient(90deg,{color},{color}88);border-radius:8px;transition:width 0.5s;"></div>
      </div>
      <span style="width:50px;text-align:right;font-size:13px;font-weight:bold;color:#e0e0e0;">{display}</span>
    </div>"""


def render_hitter_card(row: pd.Series, ml_ops: float | None = None, glow: str = "#00e5ff") -> str:
    """RPGステータスカード（打者）をHTMLで生成"""
    lv = ops_to_level(row["OPS"])
    team = row.get("team", "")

    bars = ""
    bars += _bar_html(t("bar_hr"), row["HR"], 50, f"{row['HR']:.0f}", "#ff4466")
    bars += _bar_html(t("bar_avg"), row["AVG"], 0.350, f"{row['AVG']:.3f}", "#44ff88")
    bars += _bar_html(t("bar_obp"), row["OBP"], 0.450, f"{row['OBP']:.3f}", "#44aaff")
    bars += _bar_html(t("bar_slg"), row["SLG"], 0.650, f"{row['SLG']:.3f}", "#ffaa44")
    bars += _bar_html("OPS", row["OPS"], 1.100, f"{row['OPS']:.3f}", "#00e5ff")

    compare = ""
    if ml_ops is not None:
        compare = f"""
        <div style="margin-top:8px;padding:6px 10px;background:#1a1a2e;border-radius:6px;font-size:12px;color:#aaa;">
          {t("stat_pred")}: <span style="color:#4CAF50;font-weight:bold;">{row['OPS']:.3f}</span>
          &nbsp;|&nbsp; {t("ai_pred")}: <span style="color:#2196F3;font-weight:bold;">{ml_ops:.3f}</span>
        </div>"""

    return f"""
    <div style="background:linear-gradient(135deg,#0d0d24,#1a1a3a);border:1px solid {glow}44;
                border-radius:12px;padding:16px;margin:8px 0;box-shadow:0 0 15px {glow}22;
                font-family:'Segoe UI',sans-serif;max-width:400px;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
        <div>
          <span style="color:{glow};font-size:16px;font-weight:bold;">{lv_to_stars(lv)}</span>
          <span style="color:#e0e0e0;font-size:18px;font-weight:bold;margin-left:10px;">{row['player']}</span>
        </div>
        <span style="color:{glow};font-size:12px;border:1px solid {glow}66;padding:2px 8px;border-radius:4px;">{team}</span>
      </div>
      {bars}
      {compare}
    </div>"""


def render_pitcher_card(row: pd.Series, ml_era: float | None = None, glow: str = "#00e5ff") -> str:
    """RPGステータスカード（投手）をHTMLで生成"""
    lv = era_to_level(row["ERA"])
    team = row.get("team", "")

    bars = ""
    bars += _bar_html("WHIP", 1.0 / max(row["WHIP"], 0.5), 1.0 / 0.8, f"{row['WHIP']:.2f}", "#44aaff")
    bars += _bar_html(t("bar_so"), row["SO"], 250, f"{row['SO']:.0f}", "#ff4466")
    bars += _bar_html(t("bar_w"), row["W"], 20, f"{row['W']:.0f}", "#44ff88")
    bars += _bar_html(t("bar_ip"), row["IP"], 200, f"{row['IP']:.0f}", "#ffaa44")
    era_pct = max(0, min(100, (6.0 - row["ERA"]) / 5.0 * 100))
    bars += f"""
    <div style="display:flex;align-items:center;margin:4px 0;gap:8px;">
      <span style="width:60px;font-size:13px;color:#aaa;">{t("bar_era")}</span>
      <div style="flex:1;height:16px;background:#1a1a2e;border-radius:8px;overflow:hidden;">
        <div style="width:{era_pct:.0f}%;height:100%;background:linear-gradient(90deg,#00e5ff,#00e5ff88);border-radius:8px;"></div>
      </div>
      <span style="width:50px;text-align:right;font-size:13px;font-weight:bold;color:#e0e0e0;">{row['ERA']:.2f}</span>
    </div>"""

    compare = ""
    if ml_era is not None:
        compare = f"""
        <div style="margin-top:8px;padding:6px 10px;background:#1a1a2e;border-radius:6px;font-size:12px;color:#aaa;">
          {t("stat_pred")}: <span style="color:#4CAF50;font-weight:bold;">{row['ERA']:.2f}</span>
          &nbsp;|&nbsp; {t("ai_pred")}: <span style="color:#2196F3;font-weight:bold;">{ml_era:.2f}</span>
        </div>"""

    return f"""
    <div style="background:linear-gradient(135deg,#0d0d24,#1a1a3a);border:1px solid {glow}44;
                border-radius:12px;padding:16px;margin:8px 0;box-shadow:0 0 15px {glow}22;
                font-family:'Segoe UI',sans-serif;max-width:400px;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
        <div>
          <span style="color:{glow};font-size:16px;font-weight:bold;">{lv_to_stars(lv)}</span>
          <span style="color:#e0e0e0;font-size:18px;font-weight:bold;margin-left:10px;">{row['player']}</span>
        </div>
        <span style="color:{glow};font-size:12px;border:1px solid {glow}66;padding:2px 8px;border-radius:4px;">{team}</span>
      </div>
      {bars}
      {compare}
    </div>"""


def _safe_float(val, default: float = 0.0) -> float:
    try:
        v = float(val)
        return default if pd.isna(v) else v
    except (TypeError, ValueError):
        return default


def render_radar_chart(row: pd.Series, title: str = "", color: str = "#00e5ff") -> go.Figure:
    """打者レーダーチャート（5軸）"""
    categories = [t("radar_hr"), t("radar_avg"), t("radar_obp"), t("radar_slg"), "OPS"]
    values = [
        _norm_hr(_safe_float(row["HR"])),
        _norm_avg(_safe_float(row["AVG"])),
        _norm_obp(_safe_float(row["OBP"])),
        _norm_slg(_safe_float(row["SLG"])),
        _norm_ops(_safe_float(row["OPS"])),
    ]

    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)

    fig = go.Figure()
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
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
        height=300,
        margin=dict(l=50, r=50, t=30, b=30),
    )
    if title:
        layout_kwargs["title"] = dict(text=title, font=dict(size=14, color="#e0e0e0"))
    fig.update_layout(**layout_kwargs)
    return fig


def render_pitcher_radar_chart(row: pd.Series, title: str = "", color: str = "#00e5ff") -> go.Figure:
    """投手レーダーチャート（5軸: 防御率・WHIP・奪三振・投球回・勝利）"""
    categories = [t("radar_era"), "WHIP", t("radar_so"), t("radar_ip"), t("radar_w")]
    values = [
        _norm_era_r(_safe_float(row["ERA"])),
        _norm_whip_r(_safe_float(row["WHIP"])),
        _norm_so_p(_safe_float(row["SO"])),
        _norm_ip(_safe_float(row["IP"])),
        _norm_w(_safe_float(row["W"])),
    ]

    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)

    fig = go.Figure()
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
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
        height=300,
        margin=dict(l=50, r=50, t=30, b=30),
    )
    if title:
        layout_kwargs["title"] = dict(text=title, font=dict(size=14, color="#e0e0e0"))
    fig.update_layout(**layout_kwargs)
    return fig


def render_vs_radar(row1: pd.Series, row2: pd.Series, c1: str = "#ff4466", c2: str = "#44aaff") -> go.Figure:
    """2選手の重ねレーダーチャート"""
    categories = [t("radar_hr"), t("radar_avg"), t("radar_obp"), t("radar_slg"), "OPS"]
    v1 = [_norm_hr(_safe_float(row1["HR"])), _norm_avg(_safe_float(row1["AVG"])),
          _norm_obp(_safe_float(row1["OBP"])), _norm_slg(_safe_float(row1["SLG"])),
          _norm_ops(_safe_float(row1["OPS"]))]
    v2 = [_norm_hr(_safe_float(row2["HR"])), _norm_avg(_safe_float(row2["AVG"])),
          _norm_obp(_safe_float(row2["OBP"])), _norm_slg(_safe_float(row2["SLG"])),
          _norm_ops(_safe_float(row2["OPS"]))]
    cats = categories + [categories[0]]

    r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
    r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=v1 + [v1[0]], theta=cats, fill="toself",
        fillcolor=f"rgba({r1},{g1},{b1},0.15)",
        line=dict(color=c1, width=2), name=str(row1.get("player", "")),
    ))
    fig.add_trace(go.Scatterpolar(
        r=v2 + [v2[0]], theta=cats, fill="toself",
        fillcolor=f"rgba({r2},{g2},{b2},0.15)",
        line=dict(color=c2, width=2), name=str(row2.get("player", "")),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#0d0d24",
            radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, gridcolor="#333"),
            angularaxis=dict(gridcolor="#333", linecolor="#444"),
        ),
        showlegend=True,
        legend=dict(font=dict(color="#e0e0e0"), bgcolor="rgba(0,0,0,0)"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
        height=350,
        margin=dict(l=50, r=50, t=30, b=30),
    )
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
    ml_h = data["ml_hitters"]

    if mh.empty or mp.empty:
        st.error(t("no_data"))
        return

    # チーム選択ボタン
    if st.button(t("btn_all_top3"), key="top_reset", type="primary" if not st.session_state.get("selected_team") else "secondary"):
        st.session_state["selected_team"] = None

    st.markdown(f"<div style='color:#888;font-size:12px;margin-bottom:4px;'>{t('central_league')}</div>",
                unsafe_allow_html=True)
    cl_cols = st.columns(6)
    for i, team in enumerate(CENTRAL_TEAMS):
        glow = NPB_TEAM_GLOW.get(team, "#00e5ff")
        is_selected = st.session_state.get("selected_team") == team
        if cl_cols[i].button(team, key=f"team_{team}",
                             type="primary" if is_selected else "secondary"):
            st.session_state["selected_team"] = team
            st.rerun()

    st.markdown(f"<div style='color:#888;font-size:12px;margin-bottom:4px;'>{t('pacific_league')}</div>",
                unsafe_allow_html=True)
    pl_cols = st.columns(6)
    for i, team in enumerate(PACIFIC_TEAMS):
        glow = NPB_TEAM_GLOW.get(team, "#00e5ff")
        is_selected = st.session_state.get("selected_team") == team
        if pl_cols[i].button(team, key=f"team_{team}",
                             type="primary" if is_selected else "secondary"):
            st.session_state["selected_team"] = team
            st.rerun()

    selected_team = st.session_state.get("selected_team")

    if selected_team:
        # チーム選手一覧表示
        team_glow = NPB_TEAM_GLOW.get(selected_team, "#00e5ff")
        ml_p = data["ml_pitchers"]

        # 打者一覧
        team_hitters = mh[(mh["team"] == selected_team) & (mh["PA"] >= 100)].sort_values("OPS", ascending=False)
        st.markdown(f"### {t('team_batters_title').format(team=selected_team)}")
        st.caption(t("batter_pred_caption"))
        if team_hitters.empty:
            st.info(t("no_data_pa").format(team=selected_team))
        else:
            display_h = team_hitters[["player", "AVG", "HR", "RBI", "H", "BB", "SB", "OBP", "SLG", "OPS"]].copy()
            display_h.columns = [
                t("col_player"), t("col_avg"), t("col_hr"), t("col_rbi"), t("col_h"),
                t("col_bb"), t("col_sb"), t("col_obp"), t("col_slg"), "OPS",
            ]
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
            display_p.columns = [t("col_player"), t("col_era"), t("col_w"), t("col_so"), t("col_ip"), "WHIP"]
            display_p["防御率"] = display_p["防御率"].apply(lambda x: f"{x:.2f}")
            display_p["勝利"] = display_p["勝利"].apply(lambda x: f"{x:.0f}")
            display_p["奪三振"] = display_p["奪三振"].apply(lambda x: f"{x:.0f}")
            display_p["投球回"] = display_p["投球回"].apply(lambda x: f"{x:.0f}")
            display_p["WHIP"] = display_p["WHIP"].apply(lambda x: f"{x:.2f}")
            display_p = display_p.reset_index(drop=True)
            display_p.index = display_p.index + 1
            st.dataframe(display_p, use_container_width=True, height=min(400, len(display_p) * 40 + 60))
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

        cols = st.columns(3)
        medals = ["🥇", "🥈", "🥉"]
        for i, (_, row) in enumerate(top_hitters.iterrows()):
            with cols[i]:
                glow = NPB_TEAM_GLOW.get(row["team"], "#00e5ff")
                ml_match = ml_h[ml_h["player"] == row["player"]]
                ml_ops = ml_match.iloc[0]["pred_OPS"] if not ml_match.empty else None
                st.markdown(f"<div style='text-align:center;font-size:24px;'>{medals[i]}</div>",
                            unsafe_allow_html=True)
                components.html(render_hitter_card(row, ml_ops=ml_ops, glow=glow), height=260)
                st.plotly_chart(render_radar_chart(row, title=row["player"], color=glow), use_container_width=True)

        # TOP3 投手
        st.markdown(f"### {t('top3_pitchers')}")
        top_pitchers = mp[mp["IP"] >= 100].nsmallest(3, "ERA")

        cols = st.columns(3)
        ml_p = data["ml_pitchers"]
        for i, (_, row) in enumerate(top_pitchers.iterrows()):
            with cols[i]:
                glow = NPB_TEAM_GLOW.get(row["team"], "#00e5ff")
                ml_match = ml_p[ml_p["player"] == row["player"]]
                ml_era = ml_match.iloc[0]["pred_ERA"] if not ml_match.empty else None
                st.markdown(f"<div style='text-align:center;font-size:24px;'>{medals[i]}</div>",
                            unsafe_allow_html=True)
                components.html(render_pitcher_card(row, ml_era=ml_era, glow=glow), height=260)
                st.plotly_chart(render_pitcher_radar_chart(row, title=row["player"], color=glow), use_container_width=True)

        # 注目対決
        st.markdown(f"### {t('featured_matchup')}")
        top10 = mh[mh["PA"] >= 200].nlargest(10, "OPS")
        if len(top10) >= 2:
            pair = top10.sample(2, random_state=random.randint(0, 9999))
            p1, p2 = pair.iloc[0], pair.iloc[1]
            _render_vs_section(p1, p2)


def _render_vs_section(p1: pd.Series, p2: pd.Series):
    """VS演出（2選手比較）"""
    g1 = NPB_TEAM_GLOW.get(p1["team"], "#ff4466")
    g2 = NPB_TEAM_GLOW.get(p2["team"], "#44aaff")

    vs_html = f"""
    <div style="display:flex;align-items:center;justify-content:center;gap:20px;padding:20px 0;">
      <div style="text-align:center;">
        <div style="color:{g1};font-size:20px;font-weight:bold;">{p1['player']}</div>
        <div style="color:#888;font-size:12px;">{p1['team']} / {lv_to_stars(ops_to_level(p1['OPS']))}</div>
      </div>
      <div style="font-size:36px;font-weight:bold;color:#ff4466;
                  text-shadow:0 0 20px #ff446688;">VS</div>
      <div style="text-align:center;">
        <div style="color:{g2};font-size:20px;font-weight:bold;">{p2['player']}</div>
        <div style="color:#888;font-size:12px;">{p2['team']} / {lv_to_stars(ops_to_level(p2['OPS']))}</div>
      </div>
    </div>"""
    components.html(vs_html, height=100)

    col1, col2 = st.columns(2)
    stats = [
        (t("radar_hr"), "HR", ".0f"), (t("radar_avg"), "AVG", ".3f"), (t("radar_obp"), "OBP", ".3f"),
        (t("radar_slg"), "SLG", ".3f"), ("OPS", "OPS", ".3f"),
    ]

    rows_html = ""
    for label, key, fmt in stats:
        v1 = p1[key]
        v2 = p2[key]
        c1 = g1 if v1 >= v2 else "#666"
        c2 = g2 if v2 >= v1 else "#666"
        rows_html += f"""
        <div style="display:flex;align-items:center;justify-content:center;gap:10px;margin:4px 0;font-size:14px;">
          <span style="width:70px;text-align:right;color:{c1};font-weight:{'bold' if v1>=v2 else 'normal'};">{v1:{fmt}}</span>
          <span style="width:60px;text-align:center;color:#888;">{label}</span>
          <span style="width:70px;text-align:left;color:{c2};font-weight:{'bold' if v2>=v1 else 'normal'};">{v2:{fmt}}</span>
        </div>"""

    components.html(f"""
    <div style="background:#0d0d24;border-radius:10px;padding:12px;border:1px solid #333;">
      {rows_html}
    </div>""", height=180)

    st.plotly_chart(render_vs_radar(p1, p2, c1=g1, c2=g2), use_container_width=True)


QUICK_HITTERS = ["牧", "近藤", "村上", "宮崎", "佐藤輝", "岡本", "坂倉", "万波"]
QUICK_PITCHERS = ["才木", "モイネロ", "宮城", "戸郷", "東", "高橋宏", "伊藤大", "山下"]


def page_hitter_prediction(data: dict):
    st.markdown(f"### {t('hitter_pred_title')}")

    # クイックボタン
    st.markdown('<div style="margin-bottom:10px;">', unsafe_allow_html=True)
    btn_cols = st.columns(len(QUICK_HITTERS))
    for i, qname in enumerate(QUICK_HITTERS):
        if btn_cols[i].button(qname, key=f"qh_{qname}"):
            st.session_state["hitter_search"] = qname
    st.markdown('</div>', unsafe_allow_html=True)

    name = st.text_input(t("search_by_name"), key="hitter_search",
                         placeholder=t("search_hint_hitter"))
    if not name:
        st.info(t("search_prompt_btn"))
        return

    marcel = _search(data["marcel_hitters"], name)
    ml = _search(data["ml_hitters"], name)
    if marcel.empty and ml.empty:
        st.warning(t("no_player_found").format(name=name))
        return

    for _, row in marcel.iterrows():
        glow = NPB_TEAM_GLOW.get(row["team"], "#00e5ff")
        ml_match = ml[ml["player"] == row["player"]]
        ml_ops = ml_match.iloc[0]["pred_OPS"] if not ml_match.empty else None

        col1, col2 = st.columns([1, 1])
        with col1:
            components.html(render_hitter_card(row, ml_ops=ml_ops, glow=glow), height=280)
        with col2:
            st.plotly_chart(render_radar_chart(row, title=row["player"], color=glow),
                            use_container_width=True)

        if ml_ops is not None:
            fig = go.Figure(data=[
                go.Bar(name=t("stat_pred"), x=[t("ops_chart_label")], y=[row["OPS"]], marker_color="#4CAF50"),
                go.Bar(name=t("ai_pred"), x=[t("ops_chart_label")], y=[ml_ops], marker_color="#2196F3"),
            ])
            fig.update_layout(
                barmode="group", height=250, yaxis_title=t("ops_chart_label"),
                yaxis_range=[0, max(row["OPS"], ml_ops) * 1.2],
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e0e0e0"),
                xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333"),
            )
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")


def page_pitcher_prediction(data: dict):
    st.markdown(f"### {t('pitcher_pred_title')}")

    btn_cols = st.columns(len(QUICK_PITCHERS))
    for i, qname in enumerate(QUICK_PITCHERS):
        if btn_cols[i].button(qname, key=f"qp_{qname}"):
            st.session_state["pitcher_search"] = qname

    name = st.text_input(t("search_by_name"), key="pitcher_search",
                         placeholder=t("search_hint_pitcher"))
    if not name:
        st.info(t("search_prompt_btn"))
        return

    marcel = _search(data["marcel_pitchers"], name)
    ml = _search(data["ml_pitchers"], name)
    if marcel.empty and ml.empty:
        st.warning(t("no_player_found").format(name=name))
        return

    for _, row in marcel.iterrows():
        glow = NPB_TEAM_GLOW.get(row["team"], "#00e5ff")
        ml_match = ml[ml["player"] == row["player"]]
        ml_era = ml_match.iloc[0]["pred_ERA"] if not ml_match.empty else None

        components.html(render_pitcher_card(row, ml_era=ml_era, glow=glow), height=280)

        if ml_era is not None:
            fig = go.Figure(data=[
                go.Bar(name=t("stat_pred"), x=[t("era_chart_label")], y=[row["ERA"]], marker_color="#4CAF50"),
                go.Bar(name=t("ai_pred"), x=[t("era_chart_label")], y=[ml_era], marker_color="#2196F3"),
            ])
            fig.update_layout(
                barmode="group", height=250, yaxis_title=t("era_chart_label"),
                yaxis_range=[0, max(row["ERA"], ml_era) * 1.3],
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e0e0e0"),
                xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333"),
            )
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")


def page_vs_battle(data: dict):
    """VS対決画面"""
    st.markdown(f"### {t('vs_title')}")

    mh = data["marcel_hitters"]
    if mh.empty:
        st.error(t("no_data"))
        return

    eligible = mh[mh["PA"] >= 200].sort_values("OPS", ascending=False)
    players = eligible["player"].tolist()

    col1, col2 = st.columns(2)
    p1_name = col1.selectbox(t("player1_label"), players, index=0, key="vs_p1")
    p2_idx = min(1, len(players) - 1)
    p2_name = col2.selectbox(t("player2_label"), players, index=p2_idx, key="vs_p2")

    p1 = eligible[eligible["player"] == p1_name].iloc[0]
    p2 = eligible[eligible["player"] == p2_name].iloc[0]

    _render_vs_section(p1, p2)


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
    st.plotly_chart(fig, use_container_width=True)


def page_sabermetrics(data: dict):
    st.markdown(f"### {t('saber_title')}")
    saber = data["sabermetrics"]
    if saber.empty:
        st.error(t("no_data"))
        return

    col1, col2 = st.columns([2, 1])
    name = col1.text_input(t("search_label"), key="saber_search", placeholder=t("search_hint_saber"))
    years = sorted(saber["year"].unique())
    year_option = col2.selectbox(t("year_label"), [t("all_years")] + [str(int(y)) for y in years], key="saber_year")

    if not name:
        st.info(t("search_prompt"))
        return

    matched = _search(saber, name)
    if year_option != t("all_years"):
        matched = matched[matched["year"] == int(year_option)]

    if matched.empty:
        st.warning(t("no_match").format(name=name))
        return

    # wRC+をLv表示
    for _, row in matched.iterrows():
        lv = wrc_plus_to_level(row["wRC+"])
        glow = NPB_TEAM_GLOW.get(row["team"], "#00e5ff")
        card = f"""
        <div style="background:linear-gradient(135deg,#0d0d24,#1a1a3a);border:1px solid {glow}44;
                    border-radius:10px;padding:12px;margin:6px 0;box-shadow:0 0 10px {glow}22;
                    display:flex;align-items:center;gap:12px;font-family:'Segoe UI',sans-serif;">
          <div style="min-width:60px;text-align:center;">
            <div style="color:{glow};font-size:18px;font-weight:bold;">{lv_to_stars(lv)}</div>
          </div>
          <div style="flex:1;">
            <div style="color:#e0e0e0;font-weight:bold;">{row['player']}
              <span style="color:#888;font-size:12px;margin-left:8px;">{row['team']} / {int(row['year'])}</span>
            </div>
            <div style="color:#aaa;font-size:12px;margin-top:4px;">
              wOBA<span style="color:#666;font-size:10px;">({t("woba_desc")})</span>: <span style="color:#44ff88;">{row['wOBA']:.3f}</span> &nbsp;
              wRC+<span style="color:#666;font-size:10px;">({t("wrcplus_desc")})</span>: <span style="color:#00e5ff;">{row['wRC+']:.0f}</span> &nbsp;
              wRAA<span style="color:#666;font-size:10px;">({t("wraa_desc")})</span>: <span style="color:#ffaa44;">{row['wRAA']:.1f}</span> &nbsp;
              OPS: <span style="color:#ff4466;">{row.get('OPS', row['SLG']+row['OBP']):.3f}</span>
            </div>
          </div>
        </div>"""
        components.html(card, height=80)

    if len(matched) > 1:
        player_name = matched.iloc[0]["player"]
        player_data = matched[matched["player"] == player_name].sort_values("year")
        if len(player_data) > 1:
            st.markdown(f"**{t('wrc_trend_title').format(player=player_name)}**")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=player_data["year"], y=player_data["wRC+"],
                mode="lines+markers", line=dict(color="#00e5ff", width=2),
                marker=dict(size=8, color="#00e5ff"),
            ))
            fig.add_hline(y=100, line_dash="dash", line_color="#666",
                          annotation_text=t("league_average"), annotation_font_color="#888")
            fig.update_layout(
                height=350, xaxis_title=t("year_axis"), yaxis_title="wRC+",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e0e0e0"),
                xaxis=dict(gridcolor="#222"), yaxis=dict(gridcolor="#222"),
            )
            st.plotly_chart(fig, use_container_width=True)


def _leaderboard_card(rank: int, row: pd.Series, stat_key: str, fmt: str, glow: str) -> str:
    """ランキングカード1行"""
    medal = {1: "👑", 2: "🥈", 3: "🥉"}.get(rank, "")
    border_color = {1: "#ffd700", 2: "#c0c0c0", 3: "#cd7f32"}.get(rank, "#333")
    val = row[stat_key]
    lv = ops_to_level(val) if stat_key == "OPS" else max(1, min(99, int(val)))

    return f"""
    <div style="display:flex;align-items:center;gap:10px;padding:8px 12px;margin:4px 0;
                background:#0d0d24;border:1px solid {border_color}88;border-radius:8px;
                font-family:'Segoe UI',sans-serif;">
      <span style="min-width:30px;font-size:16px;text-align:center;">{medal or rank}</span>
      <span style="min-width:55px;color:{glow};font-size:13px;font-weight:bold;">{lv_to_stars(lv)}</span>
      <span style="flex:1;color:#e0e0e0;font-weight:bold;">{row['player']}</span>
      <span style="color:#888;font-size:12px;">{row['team']}</span>
      <span style="min-width:60px;text-align:right;color:#00e5ff;font-size:16px;font-weight:bold;">{val:{fmt}}</span>
    </div>"""


def page_hitter_rankings(data: dict):
    st.markdown(f"### {t('hitter_rank_title')}")
    mh = data["marcel_hitters"]
    if mh.empty:
        st.error(t("no_data"))
        return

    col1, col2 = st.columns(2)
    top_n = col1.slider(t("show_n"), 5, 50, 20, key="hitter_rank_n")
    sort_labels = {
        t("sort_ops"): "OPS", t("sort_avg"): "AVG",
        t("sort_hr"): "HR", t("sort_rbi"): "RBI",
    }
    sort_label = col2.selectbox(t("sort_by"), list(sort_labels.keys()), key="hitter_rank_sort")
    sort_by = sort_labels[sort_label]

    df = mh[mh["PA"] >= 200].sort_values(sort_by, ascending=False).head(top_n).reset_index(drop=True)

    fmt_map = {"OPS": ".3f", "AVG": ".3f", "HR": ".0f", "RBI": ".0f"}
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

    col1, col2 = st.columns(2)
    top_n = col1.slider(t("show_n"), 5, 50, 20, key="pitcher_rank_n")
    sort_labels = {
        t("sort_era"): "ERA", t("sort_whip"): "WHIP",
        t("sort_so"): "SO", t("sort_w"): "W",
    }
    sort_label = col2.selectbox(t("sort_by"), list(sort_labels.keys()), key="pitcher_rank_sort")
    sort_by = sort_labels[sort_label]

    ascending = sort_by in ("ERA", "WHIP")
    df = mp[mp["IP"] >= 50].sort_values(sort_by, ascending=ascending).head(top_n).reset_index(drop=True)

    fmt_map = {"ERA": ".2f", "WHIP": ".2f", "SO": ".0f", "W": ".0f"}
    fmt = fmt_map.get(sort_by, ".2f")

    cards = ""
    for i, (_, row) in enumerate(df.iterrows()):
        glow = NPB_TEAM_GLOW.get(row["team"], "#00e5ff")
        lv = era_to_level(row["ERA"]) if sort_by in ("ERA", "WHIP") else max(1, min(99, int(row[sort_by])))
        medal = {1: "👑", 2: "🥈", 3: "🥉"}.get(i + 1, "")
        border_color = {1: "#ffd700", 2: "#c0c0c0", 3: "#cd7f32"}.get(i + 1, "#333")
        val = row[sort_by]
        cards += f"""
        <div style="display:flex;align-items:center;gap:10px;padding:8px 12px;margin:4px 0;
                    background:#0d0d24;border:1px solid {border_color}88;border-radius:8px;
                    font-family:'Segoe UI',sans-serif;">
          <span style="min-width:30px;font-size:16px;text-align:center;">{medal or i+1}</span>
          <span style="min-width:55px;color:{glow};font-size:13px;font-weight:bold;">{lv_to_stars(lv)}</span>
          <span style="flex:1;color:#e0e0e0;font-weight:bold;">{row['player']}</span>
          <span style="color:#888;font-size:12px;">{row['team']}</span>
          <span style="min-width:60px;text-align:right;color:#00e5ff;font-size:16px;font-weight:bold;">{val:{fmt}}</span>
        </div>"""

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

    # --- wOBA回帰係数（2015-2025実績から算出）---
    df_fit = saber[saber["PA"] >= 100].dropna(subset=["wOBA", "OBP", "SLG"])
    X = np.column_stack([df_fit["OBP"].values, df_fit["SLG"].values, np.ones(len(df_fit))])
    coeffs, _, _, _ = np.linalg.lstsq(X, df_fit["wOBA"].values, rcond=None)
    a_obp, b_slg, intercept_w = coeffs

    # リーグ環境値（2022-2025）
    recent_s = saber[saber["year"] >= 2022]
    lg_woba = recent_s[recent_s["PA"] >= 50]["wOBA"].mean()
    woba_scale = 1.15  # NPB典型的wOBAスケール

    # 歴史的リーグ平均得点・失点（1チームあたり）
    recent_p = pyth[pyth["year"] >= 2022]
    lg_avg_rs = recent_p.groupby("year")["RS"].mean().mean()
    lg_avg_ra = recent_p.groupby("year")["RA"].mean().mean()

    # Marcel投手全体の加重平均ERA（リーグ基準ERA）
    lg_era = (mp["ERA"] * mp["IP"]).sum() / mp["IP"].sum() if mp["IP"].sum() > 0 else 3.5

    # --- 選手ごとのwOBA・wRAA推定 ---
    mh = mh.copy()
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
                <div style="display:flex;align-items:center;gap:8px;padding:10px 14px;margin:4px 0;
                            background:#0d0d24;border-left:4px solid {glow};border-radius:6px;
                            font-family:'Segoe UI',sans-serif;">
                  <span style="min-width:30px;font-size:16px;text-align:center;">{medal or rank}</span>
                  <span style="min-width:100px;color:{glow};font-weight:bold;font-size:16px;">{row['team']}</span>
                  {w_cell}
                  <span style="color:#888;font-size:14px;min-width:50px;">{row['pred_L']:.0f}{t("losses_suffix")}</span>
                  <span style="color:#aaa;font-size:12px;min-width:60px;">{t("wpct_prefix")}{row['pred_WPCT']:.3f}</span>
                  <span style="color:#666;font-size:11px;">{t("rs_label")}{row['pred_RS']:.0f} / {t("ra_label")}{row['pred_RA']:.0f}</span>{badge}
                </div>"""

            components.html(f"<div>{cards}</div>", height=len(lg) * 55 + 10)

            fig = go.Figure()
            err_plus  = (lg["pred_W_high"] - lg["pred_W"]).tolist() if "pred_W_high" in lg else None
            err_minus = (lg["pred_W"] - lg["pred_W_low"]).tolist()  if "pred_W_low"  in lg else None
            fig.add_trace(go.Bar(
                name=t("pred_wins_label"), x=lg["team"], y=lg["pred_W"],
                marker_color=[NPB_TEAM_COLORS.get(t, "#333") for t in lg["team"]],
                error_y=dict(
                    type="data", array=err_plus, arrayminus=err_minus,
                    visible=True, color="#ff9944", thickness=2, width=6,
                ),
            ))
            fig.update_layout(
                height=320, yaxis_title=t("pred_wins_label"),
                yaxis_range=[0, max(lg["pred_W_high"] if "pred_W_high" in lg.columns else lg["pred_W"]) * 1.1],
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e0e0e0"),
                xaxis=dict(gridcolor="#222"), yaxis=dict(gridcolor="#222"),
                annotations=[dict(
                    x=0.5, y=-0.18, xref="paper", yref="paper", showarrow=False,
                    text=t("chart_annotation"),
                    font=dict(size=10, color="#888"),
                )],
            )
            st.plotly_chart(fig, use_container_width=True)

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
            <div style="display:flex;align-items:center;gap:8px;padding:8px 12px;margin:4px 0;
                        background:#0d0d24;border-left:3px solid {glow};border-radius:6px;
                        font-family:'Segoe UI',sans-serif;">
              <span style="min-width:25px;font-size:14px;text-align:center;">{medal or rank}</span>
              <span style="min-width:90px;color:{glow};font-weight:bold;">{row['team']}</span>
              <span style="color:#e0e0e0;min-width:70px;">{t("record_fmt").format(w=int(row['W']), l=int(row['L']))}</span>
              <span style="color:#888;font-size:12px;min-width:50px;">{row['actual_WPCT']:.3f}</span>
              <span style="color:#00e5ff;font-size:12px;min-width:50px;">{t("expected_prefix")}{row['pyth_W_npb']:.1f}{t("wins_suffix")}</span>
              <span style="color:{diff_color};font-size:12px;font-weight:bold;">{diff:+.1f}</span>
            </div>"""

        components.html(f"<div>{cards}</div>", height=len(lg) * 50 + 10)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name=t("actual_wins_bar"), x=lg["team"], y=lg["W"],
            marker_color=[NPB_TEAM_COLORS.get(team_name, "#333") for team_name in lg["team"]],
        ))
        fig.add_trace(go.Bar(
            name=t("expected_wins_bar"), x=lg["team"], y=lg["pyth_W_npb"],
            marker_color="#555",
        ))
        fig.update_layout(
            barmode="group", height=300, yaxis_title=t("wins_y"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0e0"),
            xaxis=dict(gridcolor="#222"), yaxis=dict(gridcolor="#222"),
            legend=dict(orientation="h", y=1.12, font=dict(color="#e0e0e0")),
        )
        st.plotly_chart(fig, use_container_width=True)


# --- メイン ---


PAGE_KEYS = [
    "page_top", "page_standings", "page_hitter", "page_pitcher",
    "page_hitter_rank", "page_pitcher_rank", "page_vs", "page_team_wpct", "page_saber",
]

PAGE_FUNCS = {
    "page_top": page_top,
    "page_standings": page_pythagorean_standings,
    "page_hitter": page_hitter_prediction,
    "page_pitcher": page_pitcher_prediction,
    "page_hitter_rank": page_hitter_rankings,
    "page_pitcher_rank": page_pitcher_rankings,
    "page_vs": page_vs_battle,
    "page_team_wpct": page_team_wpct,
    "page_saber": page_sabermetrics,
}


def main():
    st.set_page_config(page_title="NPB成績予測", page_icon="⚾", layout="wide")

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
    </style>
    """, unsafe_allow_html=True)

    # Language toggle FIRST — must precede any t() call
    st.sidebar.radio("Language / 言語", ["日本語", "English"], key="lang", horizontal=True)

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
