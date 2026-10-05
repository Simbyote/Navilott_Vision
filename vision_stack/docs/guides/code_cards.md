# Code Cards

> One picture per source file: what it's for, what it uses and what uses it, each class with what it holds and what its methods call, its constants and its flow.

Generic tools draw structure but not the why, and they lose calls made through a list or an attribute. code2flow, for example, draws `Navigation.update()` as calling itself: the real call is `rule.update(...)` on each entry of `self.rules`, and code2flow matches calls by name. Every module here opens with a **Purpose / Main package / Flow** docstring, so a card puts that next to the structure read from the code. The code is only read, never imported or run, so cards can be made on any machine, with no camera or Pi.

**Code:** `src/scripts/code_cards.py` · **Tests:** `test_code_cards.py`

---

## 1. Making cards

From `vision_stack/`, venv active (needs matplotlib, already in the venv):

```
python3 -m src.scripts.code_cards                                      every source file, into runs/code_cards/
python3 -m src.scripts.code_cards --only src.navigation.navigation     one module (dotted name)
python3 -m src.scripts.code_cards --only src/peripherals/drive.py src.pipeline   paths work too
python3 -m src.scripts.code_cards --out docs/code_cards               somewhere else
```

- **One PNG per module.** The name follows the module path with `__` for each folder: `src/navigation/stop_line.py` becomes `navigation__stop_line.png`.
- **Tests are skipped.** They appear on each card as the number of test files that import it.
- **The default output is `runs/code_cards/`,** which git ignores: cards are generated, so regenerate them rather than committing them. Use `--out` to keep a set elsewhere.
- **After changing code, run it again.** The cards are rebuilt from the files as they are. For one file, `--only` that module plus the modules on its "Used by" box.
- An unknown module in `--only` is exit status 2, naming it; nothing is drawn.
- All 68 cards take about 40 s on a laptop.

---

## 2. Reading a card

| Box | What it tells you |
|---|---|
| **Title, summary, lines** | The file and its one-line job (its docstring's first line) |
| **Purpose** | Why the file exists, from its docstring. A long one ends with "… (more in the file's docstring)". |
| **Uses** | The project modules it imports and the names it takes from each. "Imported inside a function" means it's only loaded when needed: `main.py` loads the motor driver only when the motors are turned on. Outside packages (cv2, numpy) are listed last. |
| **Used by** | Every module that imports it ("inside a function" when that's the only place), and how many test files do. "No other module imports it" marks an entry point you run with `python3 -m`. |
| **Constants** | Its UPPER_CASE settings with their values and the comment on the same line |
| **Flow** | The order things happen, from the docstring |
| **class …** | **fields:** its declared data · **holds:** the objects it builds into `self`, and their classes, in source order · each method with its first docstring line, `(property)` where it is one · **calls →** what the method calls |
| **functions** | The module's functions, each with its first docstring line and its calls |

**Reading "calls →":**
- `self.` is dropped: `tracker.update` is `self.tracker.update()`; look up `tracker` under **holds** for its class.
- A call on a loop variable or a local, like `rule.update`, reaches whatever that variable holds. In `Navigation`, **holds** lists `rules: StopSignRule, TrafficLightRule, IntersectionRule, EndOfCourseRule`, and `update()` calls `rule.update`: update asks each rule in that list, in that order.
- Calls are in source order. Library calls (`np.`, `cv2.`), container methods (`.append`, `.get`) and methods of helpers that aren't project classes (`ax.text`, `p.add_argument`) are left out. Calls through `self` are always kept, so hardware calls such as `_pi.hardware_PWM` show.
- A list longer than three lines ends with "…".

---

## 3. Limits

- **Structure, not behavior.** A card says what calls what, not when or how often: thresholds, timing and the order across frames are in the docstrings and in `docs/infographics/`.
- **What a variable holds is read from what's built into it.** `self.tracker = tracker or StopLineTracker()` holds a `StopLineTracker`. An object passed in and stored without being built there (`self.tracker = tracker`) shows no class under **holds**.
- **The boxes are only as good as the docstrings.** A module without a `Purpose:` or `Flow:` section gets no Purpose or Flow box.

---

## Other ways to see the code

- **pydeps** (`pip install pydeps`, needs graphviz): which module imports which, as one graph. `pydeps src --cluster -o deps.svg`. An arrow from A to B means A imports B.
- **pyreverse** (ships with pylint): class diagrams; a line labeled with an attribute name means "holds". `pyreverse -o png src/navigation`.
- **py-spy** (on the Pi): where time goes while it runs, per thread. `sudo py-spy record -o flame.svg --threads --native --duration 30 -- python3 -m src.navigation_linker --camera --no-motors`.
- `guides/system_maps.md` and `guides/linker_code_maps.md`: hand-drawn diagrams of the whole system and of each linker's code.
