# Podcast prompts, one per report section

Each episode gives one section's main idea, so you can hear your own
project told back and work out what that section has to say before you
write it. They follow the report's order (`report.tex`).

Load the sources named for an episode into a podcast AI, then paste the
ground rules followed by the prompt into its customize box. Paths are from the
repository root. Always include the section's own `.tex` file: its TODOs
are the questions the hosts are steering toward.

**While you listen,** keep one note per episode: the sentence you would open
the section with, and the question you couldn't answer. The first is your
topic sentence. The second is what you still need to decide, measure or ask
Ana or Ignacio about.

## Ground rules (paste at the top of every prompt)

```text
Ground rules: this is Navilott, a lane-following robot built by a
three-person senior design team on a Raspberry Pi Zero 2 W, with classical
computer vision (thresholds, edges, shapes, color bands) and no machine
learning. Use only facts from the sources; if a number isn't in them, don't
give one. Where the code and a document disagree, the code is current. Most
hardware routines haven't been run on the robot yet: call a result pending
rather than guess it. Explain each idea before using it, for a listener who
knows embedded systems and digital design but not this robot. Name the file
or setting you're talking about. Spend the last two minutes on the one
question the section most needs its author to answer.
```

---

## Part One: framing

### 1. Overview: why this report exists

**Sources:** `vision_stack/docs/_latex/sections/overview.tex`,
`vision_stack/docs/README.md`, `vision_stack/docs/infographics/30_by_the_numbers.png`.

```text
[ground rules]
Main idea: this report is a self-evaluation of one part of a team project,
the vision stack, written by the person who built most of it. Discuss who
a report like this is for (the author, an interviewer, the team) and how
that changes what goes in it. Talk through who did what, using the commit
counts, and the honest question of how the code was written: by hand, in
AI-assisted sessions, and how that work was reviewed. End by asking what
the report is meant to prove, and to whom.
```

**Listen for:** how you'd answer "why should I read this?" in one sentence.

### 2. Problem: what the robot had to do

**Sources:** `vision_stack/docs/_latex/sections/problem.tex`, the PRD,
`vision_stack/docs/course.md`, `vision_stack/docs/requirements.md`.

```text
[ground rules]
Main idea: before any design, the robot had a job, a set of constraints and
a course, and they don't all agree with each other. Walk through the PRD's
rules that matter to vision: lane driving, traffic lights, stop signs, the
display and start button, the $100 budget, the 12 V limit and the PCB. Then
compare the PRD's course with the team's course, difference by difference,
and say whether each one makes vision easier or harder. Finally, explain
how a vague rule becomes a requirement you can check, using a few of the
P, D and R requirements as examples.
```

**Listen for:** which constraints actually shaped the design, and which were
only boxes to tick. That split is your Constraints subsection.

---

## Part Two: design

### 3. Architecture: one board, four phases

**Sources:** `vision_stack/docs/_latex/sections/architecture.tex`,
`vision_stack/docs/architecture.md`, `vision_stack/archive/run_pipeline.py`,
`vision_stack/src/pipeline.py`, `vision_stack/docs/infographics/08_four_phases.png`,
`vision_stack/docs/infographics/26_threads.png`.

```text
[ground rules]
Main idea: every robot splits into capture, perception, estimation,
navigation, control and actuation, and Navilott puts all of them on one
Raspberry Pi instead of splitting them between a Pi and a microcontroller.
Explain the three options (microcontroller only, Pi only, Pi plus
microcontroller) and why Pi only won, and what that cost: no dedicated motor
control loop, and Linux's scheduler in the timing. Then explain the software
shape: four phases, one frame at a time, a 50 ms frame budget, and which
work runs on its own thread. Compare the first integration script
(archive/run_pipeline.py) with today's pipeline.py, and name two or three
turning points between them. Briefly cover the language choice (Python over
OpenCV's C++) and the tooling (Make, pytest, the linkers).
```

**Listen for:** whether you can explain the turning points in order. That's
the backbone of Software Architecture, the subsection still to write.

### 4. Capture: light into a frame

**Sources:** `vision_stack/docs/_latex/sections/capture.tex`,
`vision_stack/docs/vision_pipeline/phase1_capture.md`,
`vision_stack/src/capture/camera.py`, `vision_stack/docs/infographics/09_capture.png`.

```text
[ground rules]
Main idea: before vision can do anything, light has to become a numbered,
timestamped array in Python, and every step of that path is a decision.
Follow a frame from the lens through the IMX290 sensor, the ISP, GStreamer
and into the program. Explain each decision and what would go wrong without
it: the pinned sensor mode, the 480x270 output size, keeping only the newest
frame, flipping in GStreamer, and who assigns frame ids and timestamps. Cover
why the IMX219 camera was replaced, and what dropped frames and a dead
camera do to the run. End with what is still unmeasured, such as the delay
from exposure to timestamp.
```

**Listen for:** which decisions you made on purpose and which you inherited.

### 5. Perception: what's in this frame?

**Sources:** `vision_stack/docs/_latex/sections/perception.tex`,
`vision_stack/docs/vision_pipeline/phase2_perception.md`,
`vision_stack/docs/concepts.md`, `vision_stack/docs/infographics/10_perception.png`,
`vision_stack/docs/infographics/05_reading_the_light.png`.

```text
[ground rules]
Main idea: perception answers one question per frame, with no memory of the
frame before: where are the lane lines, is there a stop line and how far, is
there a light and what color, is there a stop sign. Start with why classical
computer vision and not a learned model on this course. Then take each
detector in turn: the lane lines and the single offset steering reads, and
what its sign means; how a stop line is told apart from a lane line, and the
tilt gate; why the course's bare LEDs broke color detection and how glow
mode fixed it; and how far the stop-sign detector has been tuned. Finish
with why each gate counts its rejections, and how the debug views show why a
frame failed.
```

**Listen for:** the one detector you'd struggle to explain without notes.
Write that subsection first.

### 6. Estimation: from one frame to something to trust

**Sources:** `vision_stack/docs/_latex/sections/estimation.tex`,
`vision_stack/docs/vision_pipeline/phase3_estimation.md`,
`vision_stack/src/estimation/estimation.py`, `vision_stack/docs/infographics/11_estimation.png`,
`vision_stack/docs/infographics/04_vote.png`.

```text
[ground rules]
Main idea: perception says what one frame shows; estimation decides what to
believe, across frames and sensors. Explain where the line between the two
is drawn and why. Cover the lane filter: smoothing the offset, and the
vision, hold and stale states for when a frame isn't trusted. Cover the
heading tracker: what the gyro adds while vision is lost, and what
integrating a rate does over time (drift, bias). Explain voting across three
frames for lights, signs and stop lines, and what the window trades away.
End with the packet: the one record navigation receives each frame, and its
most important fields.
```

**Listen for:** whether you can say why three frames, not two or five. If not,
that's a number to justify or measure.

### 7. Navigation: what to do about it

**Sources:** `vision_stack/docs/_latex/sections/navigation.tex`,
`vision_stack/src/navigation/navigation.py`, `vision_stack/src/navigation/stop_line.py`,
`vision_stack/src/navigation/intersection.py`, `vision_stack/src/navigation/route.py`,
`vision_stack/docs/vision_pipeline/navigation_contract.md`,
`vision_stack/docs/infographics/12_navigation.png`.

```text
[ground rules]
Main idea: navigation turns the packet into wheel commands. The contract
between vision and navigation was fixed before navigation was written, so
both sides could be built and tested apart. Explain the contract: what a
navigator must always do and must never do. Then the rules in priority order:
stop sign, traffic light, intersection, lane keeping. Take the stop line
from first sight to the robot reaching it, taking STOP_DELAY_MS and
ADVANCE_MS from stop_line.py and intersection.py, since older docs give old
values. Then the intersection's stages (to the line, advance, turn, exit) and
how the route file picks left, right or straight. Ask whether the light rules
match the PRD, which allows entering only on green. Note that navigation was
built by more than one person.
```

**Listen for:** where the robot's behavior differs from the PRD's rules. Each
difference goes to Limitations.

---

## Part Three: evidence

### 8. Testing: proof without the robot

**Sources:** `vision_stack/docs/_latex/sections/testing.tex`,
`vision_stack/docs/guides/tests.md`, `vision_stack/docs/guides/pytest.md`,
`vision_stack/src/tests/scenes.py`, `vision_stack/docs/infographics/13_linker_ladder.png`.

```text
[ground rules]
Main idea: most of the pipeline can run, and be checked, with no robot
attached. Explain how: synthetic scenes drawn as frames, recorded frames
replayed, a simulated robot, and estimation that imports no hardware.
Describe the kinds of test (pinning tests that fail when a tuned value
changes, behavior tests with scripted packets, scene tests on synthetic
images) and what each catches. Be honest about the limit: a passing suite
says the code does what it was written to do, not that the robot drives the
course. Give an example where a tuned value changed on the mat and the
tests had to follow.
```

**Listen for:** the line between testing and verification, in your own words.

### 9. Diagnostics: seeing inside the Pi

**Sources:** `vision_stack/docs/_latex/sections/diagnostics.tex`,
`vision_stack/docs/guides/diagnostics.md`, `vision_stack/docs/infographics/27_reading_pi_load.png`,
`vision_stack/docs/infographics/28_reading_soak.png`.

```text
[ground rules]
Main idea: some failures don't show in the pipeline's own logs: CPU load,
thermal throttling, the scheduler waking a thread late, the I2C bus, frames
lost before Python sees them. Explain why that made diagnostic tools worth
building. Take each recorder and the question it answers. Discuss the
observer effect: how you know a recorder doesn't change the run it measures.
End with what the diagnostics showed about the frame rate, about 15 FPS
against a 20 FPS target, and which of their findings changed a decision.
```

**Listen for:** which recorder actually changed a decision. The ones that
didn't may only need a line.

### 10. Verification: does the robot meet its requirements?

**Sources:** `vision_stack/docs/_latex/sections/verification.tex`,
`vision_stack/docs/requirements.md`, `vision_stack/docs/guides/routines.md`,
`vision_stack/docs/routines/request_card.md`, `vision_stack/docs/infographics/31_routines.png`.

```text
[ground rules]
Main idea: verification asks the robot itself, on the floor, whether each
requirement holds, and the pass criteria are written down before the run,
not after. Explain what a routine is, what its request card fixes in
advance, and what each run records. Go through the requirements and the
check for each, and name the ones with no check yet. Discuss what counts as
verified: does a result from the old IMX219 camera still count? Then the
whole-robot routines (course run, figure eight, power profile) and what they
show that single-requirement checks can't.
```

**Listen for:** any requirement you can't name a check for.

### 11. Results: where things stand

**Sources:** `vision_stack/docs/_latex/sections/results.tex`,
`vision_stack/docs/requirements.md`, `vision_stack/docs/infographics/30_by_the_numbers.png`,
any `runs/routine_*` summaries you have.

```text
[ground rules]
Main idea: results grade each requirement on its evidence, with a date and a
commit, on a scale defined in advance (for example pass, fail, partial,
open). Discuss why a grading scale has to be fixed before grading. Go
through what is measured so far: the frame rate (about 15 FPS in
phase3_linker against about 19.8 in maneuver_linker, and why those differ),
glow mode on labelled traffic-light frames, and the course runs, where early
runs drifted right and failed at intersections. Be strict: anything not
measured on the robot is open, not passed.
```

**Listen for:** each "open" grade, and which routine run would close it
before Design Day.

---

## Part Four: assessment

### 12. Limitations: the facts

**Sources:** `vision_stack/docs/_latex/sections/limitations.tex`,
`vision_stack/docs/requirements.md`, `vision_stack/docs/concepts.md`,
`vision_stack/src/config.py`.

```text
[ground rules]
Main idea: limitations are facts, not regrets. Sort them into four kinds:
requirements that fail, requirements not yet shown, departures from the
PRD, and limits of the design that no tuning will fix. For the last kind,
discuss Linux not being real-time, the lighting dependence of the color and
brightness gates, a design that assumes a grid with no curves, Python's GIL,
and the 2-degree stop-line tilt gate, which trades missed crooked lines for
no false intersections. For each, say what it means for how the robot
drives.
```

**Listen for:** where a host slips from fact into judgment. That judgment
belongs in Discussion, not here.

### 13. Discussion: the judgment

**Sources:** `vision_stack/docs/_latex/sections/discussion.tex`,
`vision_stack/docs/architecture.md`, `vision_stack/archive/run_pipeline.py`,
`vision_stack/src/pipeline.py`, your Limitations draft.

```text
[ground rules]
Main idea: Limitations listed the facts; Discussion judges them. Which
decisions would the author make again, and what evidence says they were
right? Which ones cost the team, and when did that become clear? With
hindsight, what would change in the design, the process, the teamwork, and
how AI-assisted development was used? End with what must happen before
Design Day, in priority order, and argue for that order.
```

**Listen for:** which "worked" claims have evidence behind them, and which
are only feelings so far.

### 14. Conclusion and Abstract: the whole report in a minute

**Sources:** your drafts of every section so far, `vision_stack/docs/infographics/30_by_the_numbers.png`.

```text
[ground rules]
Main idea: if someone reads only this, what three things should they take
away? What the pipeline is, how it measured against its requirements, and
what matters most next. Summarize the report in three minutes, then in one
minute, then in three sentences. Point out which section's argument was
hardest to compress, since that one may not be clear yet.
```

**Listen for:** the three sentences. With small changes, they're your abstract.
Write it last.
