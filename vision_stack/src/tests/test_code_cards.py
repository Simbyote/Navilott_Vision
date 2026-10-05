"""
test_code_cards.py  --  src/scripts/code_cards.py

A small fake project in a temp folder, written the way the real modules
are (Purpose / Flow docstrings, a class holding others in a list, a lazy
import, a constant whose string holds a '#'), with known answers for every
box on a card: sections, imports both ways, holds in source order, the
calls kept and the ones left out, constants and their comments. Then the
command line on it, and the facts the cards were built to get right on the
real code (Navigation's rule list; drive.py imported inside functions).

--software  Temp files and the repo's own source, read only. No hardware.
"""
import pytest

import src.scripts.code_cards as cc
from src.params import PIPELINE_ROOT

pytestmark = pytest.mark.software

RULES = '''"""Rules: the small parts.

Purpose:
    Each rule decides one thing. This line is
    wrapped the way docstrings are.
    Note:
        an indented heading stays in the section.

Flow (update() is the order):
    1. Track.
    2. Decide.
"""
COLOR = "#abcdef"   # the color
LIMIT = 3
_PRIVATE = 1


class Tracker:
    """Shared state."""
    def update(self):
        """Advance."""

    def reset(self):
        pass


class RuleA:
    """
    Inputs:
        tracker: the shared one.
    """
    level: int = 0

    def __init__(self, tracker):
        """
        Inputs:
            tracker: the shared one.
        """
        self.tracker = tracker

    def update(self, log):
        """Decide."""
        log.append("x")
        self._helper()
        return self.tracker.update()

    def _helper(self):
        pass

    @property
    def busy(self):
        return False


class RuleB:
    def update(self):
        return None
'''

BOSS = '''"""Boss: asks every rule.

Purpose:
    Holds the rules.
"""
from src.pkg.rules import RuleA, Tracker
from src.pkg import rules


def make():
    """Build one."""
    return Boss()


class Boss:
    """Asks every rule in order."""
    def __init__(self, tracker=None):
        self.tracker = tracker or Tracker()
        self.rules = [("a", RuleA(self.tracker)), ("b", rules.RuleB()), ("c", RuleA(self.tracker))]
        self.count = 0
        self.error = ValueError("bad")

    def update(self, ax, p):
        for _, rule in self.rules:
            rule.update()
        self.tracker.update()
        ax.text(0, 0, "drawn")
        p.add_argument("--x")
        return make()

    def lazy(self):
        from src.pkg import helper
        return helper.go()
'''

HELPER = '''"""Helper: loaded late."""


def go():
    return 1
'''


@pytest.fixture
def project(tmp_path):
    src = tmp_path / "src"
    (src / "pkg").mkdir(parents=True)
    (src / "tests").mkdir()
    for path, text in (("__init__.py", ""), ("pkg/__init__.py", ""), ("pkg/rules.py", RULES),
                       ("pkg/boss.py", BOSS), ("pkg/helper.py", HELPER),
                       ("tests/test_rules.py", "from src.pkg.rules import Tracker\n"),
                       ("tests/test_more.py", "import src.pkg.rules\nfrom src.pkg import helper\n"),
                       ("tests/conftest.py", "from src.pkg.boss import Boss\n")):
        (src / path).write_text(text)
    return tmp_path


# --- reading ------------------------------------------------------------------

def test_sections_are_titled_lines_and_flow_is_found_by_its_first_word():
    summary, sections = cc.doc_sections(RULES.split('"""')[1])
    assert summary == "Rules: the small parts."
    assert sections["Purpose"].splitlines()[0] == "Each rule decides one thing. This line is"
    assert "Note" not in sections and sections["Purpose"].splitlines()[2] == "Note:"
    assert cc.section(sections, "Flow") == "1. Track.\n2. Decide."
    assert cc.section(sections, "Main package") == ""
    assert cc.doc_sections("") == ("", {})


def test_only_the_project_files_are_read(project):
    assert [p.name for p in cc.source_files(project)] == ["boss.py", "helper.py", "rules.py"]
    assert set(cc.scan(project)) == {"src.pkg.boss", "src.pkg.helper", "src.pkg.rules"}


def test_imports_split_into_project_modules_lazy_ones_and_outside_packages(tmp_path):
    f = tmp_path / "src" / "m.py"
    f.parent.mkdir()
    f.write_text("import os\nimport numpy as np\nfrom pathlib import Path\nfrom src.a import X, Y\n"
                 "from src.a import X\nimport src.b\n\ndef f():\n    from src.c import Z\n    import src.d\n"
                 "    from src.a import X\n")
    info = cc.read_module(f, tmp_path)
    assert info.uses == {"src.a": ["X", "Y"], "src.b": [], "src.c": ["Z"], "src.d": []}
    assert info.lazy == {("src.c", "Z"), ("src.d", None)}      # src.a's X is also imported at the top
    assert [m for m in info.uses if cc.lazy_module(info, m)] == ["src.c", "src.d"]
    assert info.external == ["os", "numpy", "pathlib"]


def test_a_class_holds_what_its_methods_build_into_self_in_source_order(project):
    boss = cc.scan(project)["src.pkg.boss"].classes[0]
    assert boss.holds == {"tracker": ["Tracker"], "rules": ["RuleA"]}   # rules.RuleB() isn't a bare name
    assert "count" not in boss.holds and "error" not in boss.holds


def test_calls_keep_self_project_names_and_project_methods_and_drop_the_rest(project):
    mods = cc.scan(project)
    update = next(m for m in mods["src.pkg.boss"].classes[0].methods if m.name == "update")
    assert update.calls == ["rule.update", "tracker.update", "make"]      # walk order: the loop first
    rule_update = next(m for m in mods["src.pkg.rules"].classes[1].methods if m.name == "update")
    assert rule_update.calls == ["_helper", "tracker.update"]            # log.append is noise


def test_without_project_methods_every_method_call_on_a_local_is_kept(project):
    info = cc.read_module(project / "src/pkg/boss.py", project)
    update = next(m for m in info.classes[0].methods if m.name == "update")
    assert {"ax.text", "p.add_argument", "rule.update"} <= set(update.calls)
    rules = cc.read_module(project / "src/pkg/rules.py", project)
    assert "log.append" not in rules.classes[1].methods[1].calls       # container methods are always noise


def test_library_calls_are_dropped_unless_made_through_self():
    import ast
    fn = ast.parse("def f(self):\n    np.zeros(3)\n    self.np.go()\n    cv2.imread('x')\n").body[0]
    assert cc.calls_in(fn, set(), set()) == ["np.go"]


def test_methods_carry_their_kind_and_a_heading_only_docstring_is_no_description(project):
    rule_a = cc.scan(project)["src.pkg.rules"].classes[1]
    kinds = {m.name: m.kind for m in rule_a.methods}
    assert kinds["busy"] == "property" and kinds["update"] == "def"
    init = next(m for m in rule_a.methods if m.name == "__init__")
    assert init.doc == "" and rule_a.doc == ""
    assert rule_a.fields == ["level: int"]


def test_constants_are_public_upper_case_with_the_comment_after_the_statement(project):
    consts = cc.scan(project)["src.pkg.rules"].constants
    assert consts == [("COLOR", "'#abcdef'", "the color"), ("LIMIT", "3", "")]


def test_used_by_counts_package_imports_lazy_importers_and_test_files(project):
    mods = cc.scan(project)
    users = cc.used_by(mods, project)
    assert users["src.pkg.rules"] == (["src.pkg.boss"], 2)     # conftest.py isn't a test_ file
    assert users["src.pkg.helper"] == (["src.pkg.boss (inside a function)"], 1)
    assert users["src.pkg.boss"] == ([], 0)                     # only conftest.py imports it
    boss = mods["src.pkg.boss"]
    assert not cc.lazy_module(boss, "src.pkg")                 # rules at the top, helper inside a function


# --- drawing ------------------------------------------------------------------

def test_reflow_joins_prose_and_keeps_list_items():
    text = "One line\ncontinues here.\n\n1. Item\n2. Item\n   indented\nHeading:\nnext"
    assert cc.reflow(text) == "One line continues here.\n\n1. Item\n2. Item\n   indented\nHeading:\nnext"


def test_wrap_indents_only_list_continuations():
    assert cc.wrap("aaa bbb ccc", 7) == ["aaa bbb", "ccc"]
    assert cc.wrap("1. aaa bbb ccc", 10) == ["1. aaa bbb", "   ccc"]


def test_card_boxes_drop_empty_boxes_and_mark_cut_lists(project):
    mods = cc.scan(project)
    users = cc.used_by(mods, project)
    left, right = cc.card_boxes(mods["src.pkg.helper"], users["src.pkg.helper"])
    assert [b[0] for b in left] == ["Used by"] and [b[0] for b in right] == ["functions"]
    info = mods["src.pkg.rules"]
    info.constants = [(f"C{i}", str(i), "") for i in range(cc.MAX_CONSTANTS + 3)]
    left, _ = cc.card_boxes(info, users["src.pkg.rules"])
    rows = dict((b[0], b[2]) for b in left)["Constants"]
    assert rows[-1] == ("+ 3 more", "muted") and len(rows) == cc.MAX_CONSTANTS + 1
    assert ("+ 2 test files", "muted") in dict((b[0], b[2]) for b in left)["Used by"]
    info.constants = info.constants[:cc.MAX_CONSTANTS + 1]
    left, _ = cc.card_boxes(info, users["src.pkg.rules"])
    assert dict((b[0], b[2]) for b in left)["Constants"][-1] == ("+ 1 more", "muted")
    left, _ = cc.card_boxes(mods["src.pkg.helper"], users["src.pkg.helper"])
    assert dict((b[0], b[2]) for b in left)["Used by"][-1] == ("+ 1 test file", "muted")


def test_a_long_call_list_is_cut_with_an_ellipsis():
    rows = cc._calls_rows([f"function_number_{i}" for i in range(40)])
    assert len(rows) == cc.MAX_CALL_LINES
    assert rows[0][0].startswith("  calls → ") and rows[-1][0].endswith(" …")
    assert not cc._calls_rows(["a", "b"])[0][0].endswith("…")
    calls = []
    while len(cc.wrap(", ".join(calls), cc.RIGHT_CHARS - 14)) < cc.MAX_CALL_LINES:
        calls.append(f"call_{len(calls)}")
    exact = cc._calls_rows(calls)                                # exactly MAX_CALL_LINES: nothing cut
    assert len(exact) == cc.MAX_CALL_LINES and not exact[-1][0].endswith("…")


def test_card_names_follow_the_module_path():
    assert cc.card_name("src.navigation.stop_line") == "navigation__stop_line.png"
    assert cc.card_name("src.main") == "main.png"


# --- the command line ---------------------------------------------------------

def test_main_writes_one_png_per_module(project, tmp_path):
    out, said = tmp_path / "cards", []
    assert cc.main(["--root", str(project), "--out", str(out)], say=said.append) == 0
    pngs = sorted(p.name for p in out.iterdir())
    assert pngs == ["pkg__boss.png", "pkg__helper.png", "pkg__rules.png"]
    assert all((out / p).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" for p in pngs)
    assert said == [f"3 cards in {out}"]


def test_only_takes_dotted_names_and_paths(project, tmp_path):
    out, said = tmp_path / "cards", []
    assert cc.main(["--root", str(project), "--out", str(out), "--only", "src.pkg.rules",
                    "src/pkg/helper.py"], say=said.append) == 0
    assert sorted(p.name for p in out.iterdir()) == ["pkg__helper.png", "pkg__rules.png"]
    assert said == [f"2 cards in {out}"]


def test_an_unknown_module_is_exit_2_and_draws_nothing(project, tmp_path):
    out, said = tmp_path / "cards", []
    assert cc.main(["--root", str(project), "--out", str(out), "--only", "src.pkg.nope"], say=said.append) == 2
    assert said[0].startswith("ERROR: not a source module") and "src.pkg.nope" in said[0]
    assert not out.exists()


# --- the real code ------------------------------------------------------------

def test_navigation_card_shows_the_rules_in_order_and_the_call_through_the_loop():
    mods = cc.scan(PIPELINE_ROOT)
    nav = next(c for c in mods["src.navigation.navigation"].classes if c.name == "Navigation")
    assert nav.holds["rules"] == ["StopSignRule", "TrafficLightRule", "IntersectionRule", "EndOfCourseRule"]
    update = next(m for m in nav.methods if m.name == "update")
    assert "rule.update" in update.calls and "tracker.update" in update.calls


def test_drive_is_used_by_the_modules_that_import_it_inside_a_function():
    mods = cc.scan(PIPELINE_ROOT)
    importers, _ = cc.used_by(mods, PIPELINE_ROOT)["src.peripherals.drive"]
    assert "src.main (inside a function)" in importers
    assert "src.peripherals.sensing (inside a function)" in importers
    importers, _ = cc.used_by(mods, PIPELINE_ROOT)["src.navigation.navigation"]
    assert "src.navigation_linker" in importers                 # imported at the top: no marker
