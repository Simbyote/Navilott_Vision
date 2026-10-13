# Question episodes: one question about how the robot works, per episode

Each brief here answers one question, in one page, and covers nothing
else. Load one brief per notebook (no other sources), then paste the
prompt below with the question filled in. A single focused source is what
keeps the hosts on the question; see `../README.md`.

| Brief | The question | Good for |
|---|---|---|
| `q01_blind_stop_line.md` | How does the robot stop at a line it can't see? | a story with four values for one number |
| `q02_two_degrees.md` | Why can two degrees decide whether the robot gets lost? | trade-offs: missed vs false |
| `q03_lit_or_unlit.md` | How does a camera tell a lit LED from an unlit one? | a clever idea, explained from physics |
| `q04_never_one_frame.md` | Why does the robot never trust a single frame? | voting, smoothing, their cost |
| `q05_losing_the_lane.md` | What does the robot do when it loses the lane? | honesty about uncertainty |
| `q06_yellow_means_stop.md` | Why does yellow mean stop on this robot? | rules shaped by failures |
| `q07_turn_without_map.md` | How does the robot turn at the right corner without a map? | counting instead of mapping |
| `q08_tired_battery_turn.md` | How does a turn stay the same on a draining battery? | feedback control, where it was added |
| `q09_something_in_the_way.md` | How does the robot see something in its way? | defining an obstacle by what it isn't |
| `q10_when_things_go_wrong.md` | What happens when something breaks mid-run? | safety in layers |
| `q11_where_the_time_goes.md` | Why does the robot run at 15 frames a second instead of 20? | performance on a small computer |
| `q12_testing_without_the_robot.md` | How do you test a robot without the robot? | what tests prove and don't |
| `q13_where_am_i_in_the_lane.md` | How does the robot know where it is in the lane? | one frame to one number |

## The prompt

```text
Answer one question about the Navilott robot: "<the question>". Use the
source as the only material. Start with the short answer in plain words,
then explain how it works step by step, then tell the story of how it got
this way, with its dates and numbers. Explain every term before using it,
for a listener who knows electronics but not this robot. Don't describe the
rest of the robot. Use only facts from the source. End by putting the
source's open question to the listener, and argue both sides of it.
```

## Two variants

- **Interview:** swap the first sentence for "One host is an engineer who
  built this robot; the other asks them '<the question>' and keeps asking
  why." Good for practicing the answer yourself.
- **Explain it to a newcomer:** add "Use one everyday analogy, and say
  where the analogy breaks." Good for the questions with a physical idea
  (q03, q04, q08, q09).

## While you listen

Note the moment you'd explain it differently. That's your own version of
the answer, and the sentence to put in the report.
