#!/usr/bin/env python3
"""Code cards: one structure infographic per source file, from its docstring and its syntax tree.

Purpose:
    Generic tools draw structure without the why, and they lose calls made
    through a list or an attribute: code2flow drew Navigation.update() as
    calling itself, because the real call is `rule.update(...)` on each
    entry of self.rules and it matches calls by name. Every module here
    opens with a Purpose / Main package / Flow docstring, so a card puts the
    why next to the structure: what the file is for, which project modules
    it uses and which use it, each class with the objects it holds and what
    each method calls, its constants, and its flow. The code is only read,
    never imported or run, so any file can be carded on any machine.

Main package:
    doc_sections(): a docstring's first line and its titled sections.
    read_module(): one file's ModuleInfo.
    scan(): every non-test module under <root>/src.
    used_by(): who imports each module, and how many test files do.
    draw(): one card as a PNG.
    main(): the command line.

Flow:
    1. Parse every module under src/ (tests skipped): docstring sections,
       imports (top level or inside a function), classes, functions, constants.
    2. Turn the imports around into "used by".
    3. Draw one card per module (or per --only module) into --out.
"""
import argparse
import ast
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

from src.analysis.common import pyplot
from src.params import PIPELINE_ROOT, RUNS_DIR

_CLI_HELP = """\
One PNG per source file: purpose, imports both ways, classes, calls, constants, flow.

Run from vision_stack/:
    python3 -m src.scripts.code_cards                       every module, into runs/code_cards/
    python3 -m src.scripts.code_cards --only src.navigation.navigation src/peripherals/drive.py
    python3 -m src.scripts.code_cards --out docs/code_cards
"""

DEFAULT_OUT = RUNS_DIR / "code_cards"   # generated, so it lives with the other run output (not committed)

# Card layout, inches. 16 wide matches the docs/infographics slides; LINE fits
# one 10.5 pt line; the wrap widths are characters that fit each column at
# that size (left: proportional text, right: monospace)
CARD_W, LINE = 16.0, 0.215
LEFT_X, LEFT_W, RIGHT_X = 0.4, 5.3, 6.0
LEFT_CHARS, RIGHT_CHARS = 62, 108
MAX_PURPOSE_LINES = 14      # longer purposes end with a pointer to the file
MAX_CONSTANTS = 14
MAX_FLOW_LINES = 18
MAX_CALL_LINES = 3          # per method
MAX_COMMENT_LINES = 2       # per constant

# Calls dropped from "calls": container and string methods (log.append,
# d.get) are real calls but noise on a structure card; library namespaces
# are outside the project
NOISE_METHODS = frozenset({
    "append", "extend", "get", "items", "keys", "values", "pop", "setdefault", "join", "format", "copy",
    "strip", "split", "add", "sort", "insert", "remove", "clear", "lower", "upper", "startswith", "endswith",
})
LIBRARY_ROOTS = frozenset({"np", "cv2", "math", "time", "json", "os", "plt", "re", "csv", "Path", "sys",
                           "argparse", "textwrap", "ast", "subprocess", "warnings"})
# Capitalized callables that are exceptions or library types, not objects a class holds
NOT_HELD = frozenset({"Path", "ValueError", "RuntimeError", "TypeError", "KeyError", "FileNotFoundError"})

BG, INK, MUTED = "#f0efea", "#1f1f1f", "#6b6b6b"
MAG, CYAN, AMB, GRN = "#d4119a", "#2fb3c6", "#e39b17", "#1baf7a"
MONO = "DejaVu Sans Mono"


@dataclass
class FuncInfo:
    name: str
    doc: str                # docstring's first line
    calls: list
    kind: str = "def"       # def / property / static / class


@dataclass
class ClassInfo:
    name: str
    doc: str
    bases: list
    holds: dict             # self attribute -> classes constructed into it, in source order
    methods: list
    fields: list            # annotated class-level fields ("name: type")


@dataclass
class ModuleInfo:
    module: str             # dotted, e.g. src.navigation.navigation
    path: Path
    lines: int
    summary: str
    sections: dict          # docstring section title -> text
    uses: dict              # project module -> names imported from it
    lazy: set               # (module, name) imported only inside a function; name None for `import src.x`
    external: list          # outside packages, by top-level name
    classes: list = field(default_factory=list)
    functions: list = field(default_factory=list)
    constants: list = field(default_factory=list)   # (names, value, comment)


# =============================================================================
# Reading
# =============================================================================

def doc_sections(doc: str) -> tuple[str, dict]:
    """
    A module docstring's first line and its sections.

    A section starts at an unindented line ending in ':' that begins with a
    capital ("Purpose:", "Flow (process() is the order):"); its body is the
    following lines, one indent level removed.
    """
    if not doc:
        return "", {}
    lines = doc.strip("\n").splitlines()
    summary, sections, cur = lines[0].strip(), {}, None
    for line in lines[1:]:
        head = line.strip()
        if head.endswith(":") and not line.startswith(" ") and head[0].isupper():
            cur = head[:-1]
            sections[cur] = []
        elif cur:
            sections[cur].append(line[4:] if line.startswith("    ") else line.strip())
    return summary, {k: "\n".join(v).strip() for k, v in sections.items()}


def section(sections: dict, name: str) -> str:
    """The first section whose title starts with name ("Flow" finds "Flow, every frame")."""
    return next((v for k, v in sections.items() if k.startswith(name)), "")


def first_line(node) -> str:
    """A docstring's first line; empty when there's none or it's only a heading like 'Inputs:'."""
    d = ast.get_docstring(node)
    line = d.strip().splitlines()[0] if d else ""
    return "" if line.endswith(":") and len(line.split()) == 1 else line


def call_name(call: ast.Call) -> tuple[str, bool] | None:
    """
    (name, through self): 'self.tracker.update(...)' -> ('tracker.update', True),
    'rule.update(...)' -> ('rule.update', False), 'clamp(...)' -> ('clamp', False);
    None for calls on an expression (f()(), x[0].y()).
    """
    parts, n = [], call.func
    while isinstance(n, ast.Attribute):
        parts.append(n.attr)
        n = n.value
    if not isinstance(n, ast.Name):
        return None
    parts.append(n.id)
    parts.reverse()
    via_self = parts[0] == "self"
    if via_self:
        parts = parts[1:]
    return (".".join(parts), via_self) if parts else None


def calls_in(fn, known: set, project_methods: set | None = None) -> list:
    """
    The calls a function makes, in source order, once each.

    Kept: names in known (this module's own and what it imports from the
    project), anything through self (self.tracker.update, self._pi.write),
    and a method call on a local (rule.update) when the method is one the
    project defines; project_methods None keeps every local method call.
    Left out: library namespaces, NOISE_METHODS, and local helpers'
    methods (ax.text, p.add_argument).
    """
    out = []
    for n in sorted((n for n in ast.walk(fn) if isinstance(n, ast.Call)), key=lambda n: (n.lineno, n.col_offset)):
        got = call_name(n)
        if not got or got[0] in out:
            continue
        name, via_self = got
        root, _, rest = name.partition(".")
        method = name.rsplit(".", 1)[-1]
        if (root in LIBRARY_ROOTS and not via_self) or (rest and method in NOISE_METHODS):
            continue
        if via_self or (not rest and name in known) or (
                rest and (project_methods is None or method in project_methods)):
            out.append(name)
    return out


def constructed(expr) -> list:
    """Capitalized callables called anywhere in expr, in source order: what `self.x = ...` builds."""
    found = []
    for n in ast.walk(expr):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id[:1].isupper()
                and n.func.id not in NOT_HELD):
            found.append((n.lineno, n.col_offset, n.func.id))
    out = []
    for _, _, name in sorted(found):
        if name not in out:
            out.append(name)
    return out


def _imports(tree) -> tuple[dict, set, list]:
    """
    (project module -> names, lazy, outside packages). lazy holds the
    (module, name) pairs imported only inside a function; name is None for
    `import src.x`.
    """
    uses, external, top_pairs, inner_pairs = {}, [], set(), set()
    top = {id(n) for n in tree.body}
    for n in ast.walk(tree):
        pairs = []
        if isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            if n.module.split(".")[0] == "src":
                names = uses.setdefault(n.module, [])
                names += [a.name for a in n.names if a.name not in names]
                pairs = [(n.module, a.name) for a in n.names]
            elif n.module.split(".")[0] not in external:
                external.append(n.module.split(".")[0])
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name.split(".")[0] == "src":
                    uses.setdefault(a.name, [])
                    pairs.append((a.name, None))
                elif a.name.split(".")[0] not in external:
                    external.append(a.name.split(".")[0])
        (top_pairs if id(n) in top else inner_pairs).update(pairs)
    return uses, inner_pairs - top_pairs, external


def lazy_module(info, module: str) -> bool:
    """Every import from module in info happens inside a function."""
    return all((module, n) in info.lazy for n in (info.uses[module] or [None]))


def _class(node: ast.ClassDef, known: set, project_methods: set | None) -> ClassInfo:
    own = {m.name for m in node.body if isinstance(m, ast.FunctionDef)}
    methods, holds, fields_ = [], {}, []
    for m in node.body:
        if isinstance(m, ast.FunctionDef):
            kind = "def"
            for d in m.decorator_list:
                dn = d.id if isinstance(d, ast.Name) else getattr(d, "attr", "")
                kind = {"property": "property", "staticmethod": "static", "classmethod": "class"}.get(dn, kind)
            methods.append(FuncInfo(m.name, first_line(m), calls_in(m, known | own, project_methods), kind))
            for a in ast.walk(m):
                if not isinstance(a, ast.Assign):
                    continue
                for t in a.targets:
                    if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self":
                        made = constructed(a.value)
                        if made:
                            held = holds.setdefault(t.attr, [])
                            held += [c for c in made if c not in held]
        elif isinstance(m, ast.AnnAssign) and isinstance(m.target, ast.Name):
            fields_.append(f"{m.target.id}: {ast.unparse(m.annotation)}")
    return ClassInfo(node.name, first_line(node), [ast.unparse(b) for b in node.bases], holds, methods, fields_)


def _constant(node, source_lines: list) -> tuple | None:
    """(names, value, comment) for an UPPER_CASE module-level assignment; None otherwise."""
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    names = [t.id for t in targets if isinstance(t, ast.Name)]
    names += [e.id for t in targets if isinstance(t, ast.Tuple) for e in t.elts if isinstance(e, ast.Name)]
    names = [x for x in names if x.isupper() and not x.startswith("_")]
    if not names or node.value is None:
        return None
    line = source_lines[node.lineno - 1]
    comment = line[node.end_col_offset:].partition("#")[2].strip() if node.end_lineno == node.lineno else ""
    return ", ".join(names), ast.unparse(node.value), comment


def defined_names(path: Path) -> set:
    """Every function and method name a file defines."""
    return {n.name for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.FunctionDef)}


def read_module(path: Path, root: Path, project_methods: set | None = None) -> ModuleInfo:
    """
    One source file, read without importing it. root is the folder holding
    src/; project_methods (from defined_names over the project) narrows the
    method calls on locals that "calls" keeps.
    """
    source = path.read_text()
    tree = ast.parse(source)
    module = ".".join(path.relative_to(root).with_suffix("").parts)
    summary, sections = doc_sections(ast.get_docstring(tree) or "")
    uses, lazy, external = _imports(tree)
    info = ModuleInfo(module, path, len(source.splitlines()), summary, sections, uses, lazy, external)
    known = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    known |= {name for names in uses.values() for name in names}
    lines = source.splitlines()
    for n in tree.body:
        if isinstance(n, ast.ClassDef):
            info.classes.append(_class(n, known, project_methods))
        elif isinstance(n, ast.FunctionDef):
            info.functions.append(FuncInfo(n.name, first_line(n), calls_in(n, known, project_methods)))
        elif isinstance(n, (ast.Assign, ast.AnnAssign)):
            c = _constant(n, lines)
            if c:
                info.constants.append(c)
    return info


def source_files(root: Path) -> list:
    """Every module under root/src except tests, caches and package __init__ files."""
    return [p for p in sorted((root / "src").rglob("*.py"))
            if "tests" not in p.parts and "__pycache__" not in p.parts and p.name != "__init__.py"]


def scan(root: Path) -> dict:
    """{dotted module: ModuleInfo} for every source file under root/src."""
    files = source_files(root)
    methods = set().union(*(defined_names(p) for p in files)) if files else set()
    return {i.module: i for i in (read_module(p, root, methods) for p in files)}


def used_by(mods: dict, root: Path) -> dict:
    """
    {module: (importing modules, test files importing it)}.

    `from src.pkg import mod` counts as importing src.pkg.mod. An importer
    that only imports inside a function is marked "(inside a function)".
    """
    out = {m: ([], 0) for m in mods}

    def targets(mod, names):
        return [mod] if mod in out else [f"{mod}.{n}" for n in names if f"{mod}.{n}" in out]

    for m, info in mods.items():
        lazy_by_target = {}                     # target -> imported only inside functions
        for mod, names in info.uses.items():
            if mod in out:
                lazy_by_target[mod] = lazy_by_target.get(mod, True) and lazy_module(info, mod)
            for n in names:
                t = f"{mod}.{n}"
                if t in out:
                    lazy_by_target[t] = lazy_by_target.get(t, True) and (mod, n) in info.lazy
        for t, lazy in lazy_by_target.items():
            out[t][0].append(m + (" (inside a function)" if lazy else ""))
    for p in sorted((root / "src" / "tests").glob("test_*.py")):
        uses, _, _ = _imports(ast.parse(p.read_text()))
        hit = {t for mod, names in uses.items() for t in targets(mod, names)}
        for t in hit:
            out[t] = (out[t][0], out[t][1] + 1)
    return out


# =============================================================================
# Drawing
# =============================================================================

def reflow(text: str) -> str:
    """Join a docstring's hard-wrapped prose into paragraphs; list items and indented lines stay as they are."""
    out, joinable = [], False
    for line in text.splitlines():
        s = line.strip()
        item = s[:1] in "-•*" or s[:2].rstrip(".").isdigit() or line.startswith("  ")
        if joinable and s and not item:
            out[-1] = out[-1].rstrip() + " " + s
        else:
            out.append(line)
        joinable = bool(s) and not item and not out[-1].rstrip().endswith(":")
    return "\n".join(out)


def wrap(text: str, width: int) -> list:
    """Lines of at most width characters; a list item's or indented line's continuation is indented."""
    out = []
    for para in text.splitlines():
        lead = len(para) - len(para.lstrip())
        s = para.strip()
        item = lead > 0 or s[:1] in "-•*" or s[:2].rstrip(".").isdigit()
        out += textwrap.wrap(para, width, subsequent_indent=" " * (lead + (3 if item else 0))) or [""]
    return out


def _calls_rows(calls: list) -> list:
    """The "calls →" lines; a list cut at MAX_CALL_LINES ends with "…"."""
    lines = wrap(", ".join(calls), RIGHT_CHARS - 14)
    if len(lines) > MAX_CALL_LINES:
        lines = lines[:MAX_CALL_LINES]
        lines[-1] += " …"
    return [(("  calls → " if i == 0 else "          ") + t, "call") for i, t in enumerate(lines)]


def card_boxes(info: ModuleInfo, users: tuple) -> tuple[list, list]:
    """(left boxes, right boxes), each box (title, color, [(text, style)]); empty boxes left out."""
    left, right = [], []
    purpose = wrap(reflow(section(info.sections, "Purpose")), LEFT_CHARS)
    if len(purpose) > MAX_PURPOSE_LINES:
        purpose = purpose[:MAX_PURPOSE_LINES - 1] + ["… (more in the file's docstring)"]
    left.append(("Purpose", MAG, [(t, "text") for t in purpose]))
    uses = []
    for m, names in sorted(info.uses.items()):
        uses.append((m.removeprefix("src.") + ("   (imported inside a function)" if lazy_module(info, m) else ""), "name"))
        uses += [(t, "code") for t in wrap(", ".join(names), LEFT_CHARS - 6)]
    if info.external:
        uses += [(t, "muted") for t in wrap("outside the project: " + ", ".join(info.external), LEFT_CHARS)]
    left.append(("Uses", CYAN, uses))
    importers, n_tests = users
    used = [(m.removeprefix("src."), "code") for m in sorted(importers)]
    if n_tests:
        used.append((f"+ {n_tests} test file{'s' * (n_tests != 1)}", "muted"))
    left.append(("Used by", CYAN, used or [("no other module imports it: an entry point (python3 -m)", "muted")]))
    consts = []
    for names, value, comment in info.constants[:MAX_CONSTANTS]:
        line = f"{names} = {value}"
        consts.append((line if len(line) <= LEFT_CHARS else line[:LEFT_CHARS - 1] + "…", "code"))
        consts += [("  " + t, "muted") for t in wrap(comment, LEFT_CHARS - 4)[:MAX_COMMENT_LINES]] if comment else []
    if len(info.constants) > MAX_CONSTANTS:
        consts.append((f"+ {len(info.constants) - MAX_CONSTANTS} more", "muted"))
    left.append(("Constants", AMB, consts))

    right.append(("Flow", GRN, [(t, "text") for t in wrap(section(info.sections, "Flow"), RIGHT_CHARS)]
                  [:MAX_FLOW_LINES]))
    for c in info.classes:
        rows = [(t, "muted") for t in wrap(c.doc, RIGHT_CHARS + 12)] if c.doc else []
        if c.bases:
            rows.append(("is a " + ", ".join(c.bases), "muted"))
        if c.fields:
            rows += [("fields", "name")] + [("  " + t, "code") for t in wrap(" · ".join(c.fields), RIGHT_CHARS - 4)]
        if c.holds:
            rows += [("holds", "name")] + [(f"  {a}: {', '.join(cs)}", "code") for a, cs in c.holds.items()]
        for m in c.methods:
            if m.name.startswith("__") and m.name != "__init__":
                continue
            tag = {"property": " (property)", "static": " (static)", "class": " (classmethod)"}.get(m.kind, "")
            rows.append((f"{m.name}(){tag}", "name"))
            rows += [("  " + t, "muted") for t in wrap(m.doc, RIGHT_CHARS - 4)[:2]] if m.doc else []
            rows += _calls_rows(m.calls) if m.calls else []
        right.append((f"class {c.name}", MAG, rows))
    if info.functions:
        rows = []
        for f in info.functions:
            rows.append((f"{f.name}()", "name"))
            rows += [("  " + t, "muted") for t in wrap(f.doc, RIGHT_CHARS - 4)[:2]] if f.doc else []
            rows += _calls_rows(f.calls) if f.calls else []
        right.append(("functions", MAG, rows))
    return [b for b in left if b[2]], [b for b in right if b[2]]


def _column_height(boxes: list) -> float:
    return sum(0.62 + LINE * len(rows) + 0.3 for _, _, rows in boxes)


def _draw_column(ax, boxes, x, w, top) -> None:
    from matplotlib.patches import FancyBboxPatch
    styles = {"code": dict(family=MONO, fontsize=10, color=INK),
              "name": dict(family=MONO, fontsize=10.5, weight="bold", color=INK),
              "muted": dict(fontsize=10, color=MUTED),
              "text": dict(fontsize=10.5, color=INK),
              "call": dict(family=MONO, fontsize=9.5, color=MAG)}
    y = top
    for title, color, rows in boxes:
        h = 0.5 + LINE * len(rows) + 0.12
        ax.add_patch(FancyBboxPatch((x, y - h), w, h, boxstyle="round,pad=0.02,rounding_size=0.1",
                                    fc="white", ec=color, lw=2.2))
        ax.text(x + 0.15, y - 0.27, title, fontsize=12.5, weight="bold", family=MONO, va="center")
        yy = y - 0.6
        for text, style in rows:
            ax.text(x + 0.15, yy, text, va="center", **styles[style])
            yy -= LINE
        y -= h + 0.3


def draw(info: ModuleInfo, users: tuple, out_path: Path, plt) -> Path:
    """One card for info at out_path; users is used_by()'s entry for it."""
    left, right = card_boxes(info, users)
    height = 1.55 + max(_column_height(left), _column_height(right)) + 0.5
    fig = plt.figure(figsize=(CARD_W, height), dpi=110, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, CARD_W)
    ax.set_ylim(0, height)
    ax.axis("off")
    ax.text(LEFT_X, height - 0.45, info.module.replace(".", "/") + ".py", fontsize=24, family=MONO,
            weight="bold", va="center")
    ax.text(LEFT_X, height - 0.95, info.summary, fontsize=13, color=MUTED, va="center")
    ax.text(CARD_W - 0.4, height - 0.45, f"{info.lines} lines", fontsize=12, color=MUTED, ha="right",
            va="center", family=MONO)
    _draw_column(ax, left, LEFT_X, LEFT_W, height - 1.35)
    _draw_column(ax, right, RIGHT_X, CARD_W - RIGHT_X - 0.4, height - 1.35)
    ax.text(LEFT_X, 0.25, "calls →: what this function calls (self. dropped). A call on a loop variable "
            "(rule.update) reaches every object in the list named under holds.", fontsize=9, color=MUTED)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=BG)
    plt.close(fig)
    return out_path


def card_name(module: str) -> str:
    """src.navigation.stop_line -> navigation__stop_line.png"""
    return module.removeprefix("src.").replace(".", "__") + ".png"


def resolve(names: list, mods: dict, root: Path) -> list:
    """--only entries (dotted modules or .py paths) as module names; ValueError naming any unknown."""
    out, unknown = [], []
    for n in names:
        p = Path(n)
        if n.endswith(".py"):
            p = p if p.is_absolute() else (root / p if (root / p).exists() else Path.cwd() / p)
            try:
                n = ".".join(p.resolve().relative_to(root.resolve()).with_suffix("").parts)
            except ValueError:
                pass
        (out if n in mods else unknown).append(n)
    if unknown:
        raise ValueError(f"not a source module under {root / 'src'}: {', '.join(unknown)}")
    return out


def main(argv: list[str] | None = None, say=print) -> int:
    p = argparse.ArgumentParser(prog="code_cards", description=_CLI_HELP,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="+", metavar="MODULE", help="dotted module names or .py paths")
    p.add_argument("--out", default=str(DEFAULT_OUT), help=f"folder for the PNGs (default: {DEFAULT_OUT})")
    p.add_argument("--root", default=str(PIPELINE_ROOT), help="the folder holding src/ (default: this one)")
    args = p.parse_args(argv)
    plt = pyplot()
    if plt is None:
        say("ERROR: matplotlib is not installed")
        return 2
    root = Path(args.root)
    mods = scan(root)
    try:
        chosen = resolve(args.only, mods, root) if args.only else sorted(mods)
    except ValueError as exc:
        say(f"ERROR: {exc}")
        return 2
    users = used_by(mods, root)
    out = Path(args.out)
    for m in chosen:
        draw(mods[m], users[m], out / card_name(m), plt)
    say(f"{len(chosen)} card{'s' * (len(chosen) != 1)} in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
