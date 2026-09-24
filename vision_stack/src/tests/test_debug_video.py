"""
test_debug_video.py  --  src/debugger/debug_video.py

ViewWriter is checked by reading back what it wrote; CandidateView through a
minimal subclass that hands it scripted entries.

--software  Recording, stride, sidecar CSV, codec failure, label drawing, and
            the shared pass / low / reject grading. No camera.
"""
import csv

import cv2
import numpy as np
import pytest

import src.debugger.debug_video as dv


def frame(value=40, shape=(60, 80)):
    """Uniform BGR image."""
    return np.full(shape + (3,), value, np.uint8)


def read_video(path):
    """Every frame of a video file, as a list."""
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return out


def read_csv(path):
    """A CSV as a list of rows, header included."""
    with open(path, newline="") as f:
        return list(csv.reader(f))


@pytest.mark.software
def test_writer_records_every_pushed_frame_and_its_row(tmp_path):
    path = tmp_path / "v.avi"
    with dv.ViewWriter(str(path), ("a", "b")) as w:
        for i in range(3):
            w.push(frame(), [i, i * 2])
    assert len(read_video(path)) == 3 and w.frames_written == 3
    assert read_csv(tmp_path / "v.csv") == [["a", "b"], ["0", "0"], ["1", "2"], ["2", "4"]]


@pytest.mark.software
def test_stride_keeps_every_nth_frame_starting_with_the_first():
    w = dv.ViewWriter("unused.avi", stride=3)
    assert [w.take() for _ in range(7)] == [True, False, False, True, False, False, True]


@pytest.mark.software
def test_no_csv_fields_means_no_sidecar(tmp_path):
    with dv.ViewWriter(str(tmp_path / "v.avi")) as w:
        w.push(frame(), [1, 2])
    assert not (tmp_path / "v.csv").exists()


@pytest.mark.software
def test_csv_path_none_disables_the_sidecar_even_with_fields(tmp_path):
    with dv.ViewWriter(str(tmp_path / "v.avi"), ("a",), csv_path=None) as w:
        w.push(frame(), [1])
    assert not (tmp_path / "v.csv").exists()


@pytest.mark.software
def test_mismatched_frames_are_resized_to_the_first_size(tmp_path):
    # VideoWriter silently drops a frame of the wrong size, so push() must resize it
    path = tmp_path / "v.avi"
    with dv.ViewWriter(str(path)) as w:
        w.push(frame(shape=(60, 80)))
        out = w.push(frame(shape=(30, 50)))
    assert out.shape == (60, 80, 3)
    assert len(read_video(path)) == 2


@pytest.mark.software
def test_output_directory_is_created_on_first_frame(tmp_path):
    path = tmp_path / "nested" / "dir" / "v.avi"
    with dv.ViewWriter(str(path)) as w:
        w.push(frame())
    assert path.exists()


@pytest.mark.software
def test_an_unusable_codec_raises_instead_of_writing_nothing(tmp_path):
    w = dv.ViewWriter(str(tmp_path / "v.avi"), fourcc="ZZZZ")
    with pytest.raises(RuntimeError, match="ZZZZ"):
        w.push(frame())


@pytest.mark.software
def test_close_is_safe_before_any_frame_and_twice():
    w = dv.ViewWriter("unused.avi")
    w.close()
    w.close()


@pytest.mark.software
def test_draw_text_leaves_the_label_color_on_top_of_a_black_outline():
    img = np.full((60, 200, 3), 255, np.uint8)       # white, like tape
    dv.draw_text(img, "lane", (5, 45), dv.C_USABLE, 1.5, th=3)   # thick enough for solid, non-antialiased cores
    colors = {tuple(int(c) for c in p) for p in img.reshape(-1, 3)}
    assert tuple(dv.C_USABLE) in colors and (0, 0, 0) in colors


class ScriptedView(dv.CandidateView):
    """CandidateView whose entries come straight from data["entries"]."""
    def _entries(self, data):
        return data["entries"]


def entry(gate=None, conf=0.5):
    """One trace entry with only the fields CandidateView reads."""
    return {"gate": gate, "confidence": conf}


@pytest.mark.software
@pytest.mark.parametrize("thr, e, state", [
    (None, entry(conf=0.1), "pass"),              # no threshold: accepted always passes
    (0.5, entry(conf=0.5), "pass"),               # at the threshold passes
    (0.5, entry(conf=0.49), "low"),
    (0.5, entry(conf=None), "low"),               # a missing confidence counts as 0
    (0.5, entry(gate="area", conf=0.9), "reject"),  # a gate always wins over confidence
])
def test_candidates_are_graded_against_the_threshold(thr, e, state):
    assert ScriptedView(thr)._state(e) == state


@pytest.mark.software
def test_summary_counts_states_and_picks_the_best_accepted_entry():
    v = ScriptedView(0.5)
    sm = v._summary({"entries": [entry(conf=0.7), entry(conf=0.3), entry("area", 0.99)]})
    assert (sm["passed"], sm["low"], sm["rejected"]) == (1, 1, 1)
    assert sm["best"]["confidence"] == 0.7        # the rejected 0.99 is never the best


@pytest.mark.software
def test_header_color_follows_the_best_state_and_hides_low_without_a_threshold():
    passed = {"passed": 1, "low": 0, "rejected": 2}
    low_only = {"passed": 0, "low": 1, "rejected": 0}
    assert ScriptedView(0.5)._header(passed) == (dv.C_USABLE, "pass 1  low 0  rejected 2")
    assert ScriptedView(0.5)._header(low_only)[0] == dv.C_AMBER
    assert ScriptedView(None)._header(passed)[1] == "pass 1  rejected 2"


@pytest.mark.software
def test_threshold_report_shares_are_of_all_frames():
    # 4 frames, 2 with a candidate (0.35, 0.65): 0.30 -> 50%, 0.40 -> 25%, 0.70 -> 0%
    lines = ScriptedView()._threshold_report([0.35, 0.65], n=4)
    assert "0.30 -> 50%" in lines[-1] and "0.40 -> 25%" in lines[-1] and "0.70 -> 0%" in lines[-1]
    assert ScriptedView()._threshold_report([], n=4) == []


@pytest.mark.software
def test_zoom_multiplies_the_run_scale():
    assert ScriptedView(zoom=2)._metrics(3)[0] == 6
    assert ScriptedView(zoom=0)._metrics(1)[0] == 1   # zoom floors at 1