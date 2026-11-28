"""roster_current.get_team_for_player: the dict index gives the same answers as
the original linear scan (first team in ROSTER_CURRENT order wins)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import roster_current as RC  # noqa: E402


def _linear(player_name):
    """The function as it was before 2026-11-28 (reference)."""
    fuzzy = player_name.replace("　", "").replace(" ", "").strip().translate(RC._VARIANT_MAP)
    for team, players in RC.ROSTER_CURRENT.items():
        for p in players:
            if p.replace("　", "").replace(" ", "").strip().translate(RC._VARIANT_MAP) == fuzzy:
                return team
    return None


def _queries():
    qs = []
    for players in RC.ROSTER_CURRENT.values():
        for p in players:
            qs += [p, p.replace("　", ""), p.replace("　", " "), " " + p + " ",
                   p.translate(str.maketrans("崎高浜沢辺斎", "﨑髙濵澤邊齋"))]
    return qs + ["存在しない　選手", "", "　"]


def test_same_answers_as_the_linear_scan():
    qs = _queries()
    assert len(qs) > 3000
    assert [RC.get_team_for_player(q) for q in qs] == [_linear(q) for q in qs]
    assert sum(RC.get_team_for_player(q) is not None for q in qs) > 3000


def test_first_team_wins_and_a_new_roster_is_picked_up(monkeypatch):
    monkeypatch.setattr(RC, "ROSTER_CURRENT", {"A": ["山田　太郎"], "B": ["山田太郎", "髙橋　一"]})
    assert RC.get_team_for_player("山田 太郎") == _linear("山田 太郎") == "A"
    assert RC.get_team_for_player("高橋一") == "B"
    assert RC.get_team_for_player("佐藤") is None
