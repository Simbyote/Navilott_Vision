# Question: How does a camera tell a lit LED from an unlit one?

**This brief answers one question about Navilott, a small lane-following
robot. It covers only how the robot reads the traffic light's color.
Nothing else about the robot is in scope.**

## The short answer

By looking for what only a lit LED can do: overload the camera. A lit LED
is brighter than the sensor can record, so its center clips to pure white.
An unlit lens, however red or bright it looks, never does. The robot finds
the white spots in the traffic light's region, then names each spot by the
color of the thin ring of pixels around it. This is called glow mode.

## The story

- **First attempt: color bands.** The first approach looked for blobs of
  red, yellow or green, sized and shaped like a lamp. It was tuned on lamps
  on 2026-10-04.
- **The surprise (2026-10-06).** The course's lights are bare 5 mm LEDs on
  a board. Their lenses are colored plastic: an unlit red lens reads about
  as bright as a lit one's colored ring, so color alone kept "seeing" unlit
  lenses. On 2026-10-06's frames at the stop, the lit ring's 10th-percentile
  brightness was 133 against 178 for the same lens unlit.
- **Glow mode.**
  1. Find connected near-white pixels: bright and almost colorless.
  2. Ignore shapes that aren't spot-like. The board's edge glare was a long
     2 px strip; on 2026-10-07 it out-voted a real LED and picked up a red
     shirt's color.
  3. Count the ring's pixels against each color band. The winning color
     needs a minimum share of the ring (10%).
  4. Keep the spot with the most white pixels, since a traffic light shows
     one lamp at a time. That also beats the dimmer reflection of the LED
     in the clear panel.
- **Result:** on 196 labelled frames taken at the stop, every one was named
  right.

## What still worries

- **Exposure.** In a dim room the camera's auto exposure brightens the
  whole scene, and once (2026-10-04) it blew the lamps out to white.
  Locking exposure at the venue is the main lighting fix still open.
- **One board.** Another light, or this one seen from another stop line,
  has to land inside the small region the detector looks at.

## The open question

Glow mode depends on the LED clipping the sensor. What lighting on the day
could stop that from happening?
