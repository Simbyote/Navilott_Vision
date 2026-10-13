# Question: Why does the robot never trust a single frame?

**This brief answers one question about Navilott, a small lane-following
robot. It covers only how the robot turns noisy per-frame detections into
decisions it acts on. Nothing else about the robot is in scope.**

## The short answer

Each camera frame (about 20 a second) is judged on its own, and any one of
them can be wrong: a glint read as a lamp, a dash taken for a stop line, a
lane line lost for a frame. So the robot keeps a short memory and changes
its mind only when the evidence agrees.

## Three tools

- **Voting, for yes/no and color decisions.** Traffic light state, stop
  sign, stop line and obstacle each change only when 2 of the last 3 frames
  agree. With three frames there's always a majority, and one bad frame
  can't flip anything. Before voting, a frame's detection must pass a
  confidence gate: 0.40 for a light, 0.45 for a sign.
- **A jump gate, for the lane position.** A new offset more than 0.5 (half
  the lane view's width) away from the current estimate isn't physically
  possible between two frames at driving speed. It's treated as a dropout,
  not a measurement.
- **Smoothing.** Accepted offsets are averaged with an exponential moving
  average (α = 0.35): each new frame counts 35%, the history 65%.

## The cost

- **Delay.** A vote needs a second frame to agree, about 50 ms late. At the
  robot's speed (about 1,300 encoder counts per second at 0.40 duty) that's
  a few cm of travel before a light or stop line is believed.
- **A missed single-frame event.** Something true for only one frame is
  never acted on. On this course that's the point: real things last.
- **Smoothing lags.** It also lags a real, fast change of position, a
  trade between noise and response.

## Where this is checked

Two requirements are verified in software tests:
- **R2:** a lost lane is held for at most about 350 ms, then reported as
  unusable.
- **R3:** no single frame can flip the traffic light or stop sign state.

## The open question

Why a window of three, and not two or five? Two can't break a tie; five
would add another 100 ms of delay. Was three chosen, or measured?
