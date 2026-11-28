"""web_fetch: sends the research User-Agent (baseball-data.com answers 403 to
urllib's default) and parses the page bytes like pd.read_html(url) did."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import web_fetch as W  # noqa: E402

HTML = """<html><body><table><tr><th>選手</th><th>打率</th></tr>
<tr><td>山田　太郎</td><td>.301</td></tr></table></body></html>""".encode("utf-8")


class _Resp:
    def __init__(self, data): self.data = data
    def read(self): return self.data
    def __enter__(self): return self
    def __exit__(self, *a): return False


def test_user_agent_is_sent_and_tables_parse(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["ua"] = req.get_header("User-agent")
        seen["url"] = req.full_url
        return _Resp(HTML)
    monkeypatch.setattr(W.urllib.request, "urlopen", fake_urlopen)
    tables = W.read_html_tables("https://baseball-data.com/25/stats/hitter-all/tpa-1.html")
    assert seen["ua"] == "Mozilla/5.0 (compatible; research-bot/1.0)"
    assert seen["url"].endswith("tpa-1.html")
    assert tables[0].iloc[0, 0] == "山田　太郎" and abs(tables[0].iloc[0, 1] - 0.301) < 1e-12


def test_both_baseball_data_fetchers_use_it():
    root = Path(__file__).resolve().parents[1]
    for f in ("fetch_npb_data.py", "pythagorean.py"):
        src = (root / f).read_text(encoding="utf-8")
        assert "read_html_tables(url" in src and "pd.read_html(url" not in src, f
