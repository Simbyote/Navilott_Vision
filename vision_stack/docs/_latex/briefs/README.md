# Section briefs: one podcast episode per section

Podcast tools build an episode from **every source loaded**, and a short
instruction can't override them. Loading the general docs (README,
architecture.md, concepts.md) gives an episode about the whole pipeline,
whatever the prompt says. So each section has its own brief here: one page
that covers that section and nothing else, with its main idea, its facts
and the questions the author still has to answer.

## How to run an episode

1. Start a **new notebook** per episode (or deselect every other source).
2. Load **only that section's brief.** Add the section's `.tex` from
   `../sections/` if you want your own draft discussed, nothing else.
3. Paste the prompt below into the customize box.

If an episode still drifts, the cause is a source, not the prompt: take
sources out until only the brief is left.

## Design: the five episodes

Every prompt has the same shape. Swap in the section's name.

### Architecture (03_architecture.md)

```text
Cover only the Architecture section of the Navilott report, using the
"Brief: Architecture" source as the outline: its main idea, then each part
in order. Don't summarize the whole robot or explain any detector. Use only
facts and numbers from the source. Close by putting the brief's questions to
the author, as open questions.
```

### Capture (04_capture.md)

```text
Cover only the Capture section of the Navilott report, using the "Brief:
Capture" source as the outline: its main idea, the path a frame takes, then
each decision and what goes wrong without it. Don't go past the moment a
frame reaches Python. Use only facts and numbers from the source. Close by
putting the brief's questions to the author, as open questions.
```

### Perception (05_perception.md)

```text
Cover only the Perception section of the Navilott report, using the "Brief:
Perception" source as the outline: why classical vision, then each detector
in turn. Stay inside a single frame; don't discuss smoothing or driving
decisions. Use only facts and numbers from the source. Close by putting the
brief's questions to the author, as open questions.
```

### Estimation (06_estimation.md)

```text
Cover only the Estimation section of the Navilott report, using the "Brief:
Estimation" source as the outline: the line between perception and
estimation, then each part, then the packet. Don't explain how detections
are made or what the robot does with them. Use only facts and numbers from
the source. Close by putting the brief's questions to the author, as open
questions.
```

### Navigation (07_navigation.md)

```text
Cover only the Navigation section of the Navilott report, using the "Brief:
Navigation" source as the outline: the contract, the rules in priority
order, the stop line's timing and why it changed, then the route. Don't
explain how anything is detected. Use only facts and numbers from the
source. Close by putting the brief's questions to the author, as open
questions.
```

## While you listen

Write down two things per episode: the sentence you'd open the section
with, and the question you couldn't answer. If a fact sounds wrong, fix the
brief rather than the prompt; the brief is what the episode is built from.
