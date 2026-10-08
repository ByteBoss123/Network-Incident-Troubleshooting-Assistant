"""Inject data.json into template.html -> index.html (React dashboard artifact)."""
from pathlib import Path

here = Path(__file__).parent
data = (here / "data.json").read_text().replace("</", "<\\/")
(here / "index.html").write_text((here / "template.html").read_text().replace("/*DATA*/", data))
