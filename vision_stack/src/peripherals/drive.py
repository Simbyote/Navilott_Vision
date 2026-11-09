"""TB6612 motor driver and N20 quadrature encoders.

Purpose:
    The drive hardware and nothing above it: duty per wheel to the TB6612,
    and each wheel's encoder edges counted on pigpio callbacks, so counting
    never blocks the loop. Two watchdogs brake the motors if commands stop:
    a thread here, and a copy inside pigpiod for when Python is stalled.

Main package:
    MotorController: drive(left, right), brake(), stop().
    EncoderReader: counts() -> (left, right), + = forward.
"""

import logging
import threading
import time

import pigpio

from src.os_threads import name_os_thread, name_pigpio_threads
from src.params import MOTOR_DAEMON_WATCHDOG, MOTOR_PWM_HZ, MOTOR_WATCHDOG_S

log = logging.getLogger(__name__)

# Looks per timeout, so a brake is at most a fifth late
WATCHDOG_CHECKS = 5


class EncoderReader:
    """Left on GPIO 19/16, right on 20/21."""
    # Each encoder goes with the wheel its same-named command turns: the
    # 2026-10-01 motor pin fix moved 21/20 to the right wheel (confirmed
    # 2026-10-03). + = forward was measured per encoder by hand (2026-09-29);
    # forward, 19 leads 16 and 21 leads 20
    LEFT_C1  = 19
    LEFT_C2  = 16
    RIGHT_C1 = 20
    RIGHT_C2 = 21

    def __init__(self, pi: pigpio.pi) -> None:
        self._pi = pi
        if not self._pi.connected:
            raise RuntimeError("pigpio daemon not reachable. Run: sudo pigpiod")

        self._left_pos = 0
        self._right_pos = 0

        self._left_c1_state = 0
        self._left_c2_state = 0
        self._right_c1_state = 0
        self._right_c2_state = 0

        for pin in [self.LEFT_C1, self.LEFT_C2, self.RIGHT_C1, self.RIGHT_C2]:
            self._pi.set_mode(pin, pigpio.INPUT)
            self._pi.set_pull_up_down(pin, pigpio.PUD_UP)

        self._cb_l1 = self._pi.callback(self.LEFT_C1, pigpio.EITHER_EDGE, self._left_cb)
        self._cb_l2 = self._pi.callback(self.LEFT_C2, pigpio.EITHER_EDGE, self._left_cb)
        self._cb_r1 = self._pi.callback(self.RIGHT_C1, pigpio.EITHER_EDGE, self._right_cb)
        self._cb_r2 = self._pi.callback(self.RIGHT_C2, pigpio.EITHER_EDGE, self._right_cb)
        name_pigpio_threads()           # the counting runs in pigpio's callback thread

    def _left_cb(self, gpio: int, level: int, tick: int) -> None:
        if gpio == self.LEFT_C1:
            self._left_c1_state = level
        elif gpio == self.LEFT_C2:
            self._left_c2_state = level

        if gpio == self.LEFT_C1 and level == 1:
            if self._left_c2_state == 0:
                self._left_pos += 1
            else:
                self._left_pos -= 1

    def _right_cb(self, gpio: int, level: int, tick: int) -> None:
        if gpio == self.RIGHT_C1:
            self._right_c1_state = level
        elif gpio == self.RIGHT_C2:
            self._right_c2_state = level

        if gpio == self.RIGHT_C1 and level == 1:
            if self._right_c2_state == 0:
                self._right_pos -= 1
            else:
                self._right_pos += 1

    def counts(self) -> tuple[int, int]:
        """Since reset(). Changes no state, so any thread may call it."""
        return self._left_pos, self._right_pos

    def reset(self) -> None:
        self._left_pos = 0
        self._right_pos = 0

    def cancel(self) -> None:
        self._cb_l1.cancel()
        self._cb_l2.cancel()
        self._cb_r1.cancel()
        self._cb_r2.cancel()


DAEMON_CHECK_MS = 100
DAEMON_INIT_S = 1.0
PARAM_MAX = 0x7FFFFFFF          # script params are 32-bit

# pigpio script language. p0 heartbeat, p1 armed; v0 heartbeat last seen, v1
# looks since it moved. Armed and unchanged for `looks` looks: short brake,
# then wait for p0 to move. CMP sets F = A - x; the jumps test F
_DAEMON_SCRIPT = """
ld v0 p0 ld v1 0
tag 1 mils {check_ms}
lda p1 cmp 0 jz 3
lda p0 cmp v0 jnz 3
inr v1 lda v1 cmp {looks} jm 1
w {stby} 1 w {ain1} 1 w {ain2} 1 w {bin1} 1 w {bin2} 1
hp {pwma} {freq} 1000000 hp {pwmb} {freq} 1000000
tag 2 mils {check_ms}
lda p0 cmp v0 jz 2
tag 3 ld v0 p0 ld v1 0 jmp 1
"""


def daemon_script(pins: dict, freq: int, timeout_s: float, check_ms: int = DAEMON_CHECK_MS) -> str:
    looks = max(1, round(timeout_s * 1000.0 / check_ms))
    return " ".join(_DAEMON_SCRIPT.format(check_ms=check_ms, looks=looks, freq=freq, **pins).split())


class DaemonWatchdog:
    """
    The motor watchdog as a script inside pigpiod. A stalled or dead Python process takes
    its watchdog thread with it while the PWM keeps the last command (a 1,558 ms frame on
    2026-10-20); pigpiod keeps watching. Never raises: if pigpiod refuses the script, on
    stays False and the Python watchdog carries on, with a warning.
    """
    def __init__(self, pi, pins: dict, freq: int, timeout_s: float, sleep=time.sleep) -> None:
        self._pi, self._sleep = pi, sleep
        self._sid = None
        self._beat = 0
        self.status = "off"
        try:
            self._sid = pi.store_script(daemon_script(pins, freq, timeout_s).encode())
            why = self._ready()
            if why:
                raise RuntimeError(why)
            pi.run_script(self._sid, [0, 0])
        except Exception as exc:                # pigpio.error, an old pigpiod, a refused script
            self._delete()
            self.status = f"off ({exc})"
            log.warning("motor watchdog in pigpiod not running: %s; the Python watchdog only", exc)
            return
        self.status = "running"

    @property
    def on(self) -> bool:
        return self._sid is not None

    def _ready(self) -> str:
        """"" once the stored script can run, else why not."""
        end = time.monotonic() + DAEMON_INIT_S
        while True:
            status = self._pi.script_status(self._sid)[0]
            if status == pigpio.PI_SCRIPT_FAILED:
                return "pigpiod failed the script"
            if status != pigpio.PI_SCRIPT_INITING:
                return ""
            if time.monotonic() > end:
                return "the script never finished initialising"
            self._sleep(0.01)

    def beat(self) -> None:
        """Move the heartbeat and arm."""
        if self._sid is not None:
            self._beat = self._beat % PARAM_MAX + 1
            self._pi.update_script(self._sid, [self._beat, 1])

    def disarm(self) -> None:
        """After a brake or stop, the motors are safe however long the loop is gone."""
        if self._sid is not None:
            self._pi.update_script(self._sid, [self._beat, 0])

    def close(self) -> None:
        if self._sid is not None:
            try:
                self._pi.stop_script(self._sid)
            except Exception:
                pass
        self._delete()
        self.status = "off"

    def _delete(self) -> None:
        if self._sid is not None:
            try:
                self._pi.delete_script(self._sid)
            except Exception:
                pass
        self._sid = None


class MotorController:
    """
    drive() arms the watchdog; brake() and stop() disarm it. Armed with no drive() for
    watchdog_s, it short-brakes and counts a trip; the next drive() drives again, so a
    slow loop carries on and a dead one stays braked. One lock covers every pin sequence.
    """

    def __init__(
        self,
        pi: pigpio.pi,
        pwma: int = 12,   # left motor: PWM
        ain1: int = 22,   # left motor: high = reverse
        ain2: int = 27,   # left motor: high = forward
        pwmb: int = 13,   # right motor: PWM
        bin1: int = 25,   # right motor: high = forward
        bin2: int = 24,   # right motor: high = reverse
        stby: int = 23,
        pwm_freq: int = MOTOR_PWM_HZ,
        watchdog_s: float | None = MOTOR_WATCHDOG_S,
        clock=time.monotonic,
        daemon_watchdog: bool = MOTOR_DAEMON_WATCHDOG,
    ) -> None:
        """watchdog_s None or 0 turns the watchdog off; daemon_watchdog needs it on."""
        self._pi = pi
        if not self._pi.connected:
            raise RuntimeError("pigpio daemon not reachable. Run: sudo pigpiod")

        self.pwma = pwma
        self.ain1 = ain1
        self.ain2 = ain2
        self.pwmb = pwmb
        self.bin1 = bin1
        self.bin2 = bin2
        self.stby = stby
        self.pwm_freq = pwm_freq

        self.watchdog_s = watchdog_s or None
        self.watchdog_trips = 0
        self._clock = clock
        self._lock = threading.Lock()
        self._driven_at: float | None = None      # None when disarmed
        self._wd_stop = threading.Event()
        self._wd_thread: threading.Thread | None = None

        self._init_gpio()
        name_pigpio_threads()
        self._use_daemon = bool(daemon_watchdog and self.watchdog_s)
        self._daemon_why = ""
        self.daemon: DaemonWatchdog | None = None
        self._start_daemon()

    def _init_gpio(self) -> None:
        for pin in [self.ain1, self.ain2, self.bin1, self.bin2, self.stby]:
            self._pi.set_mode(pin, pigpio.OUTPUT)

    def drive(self, left_speed: float, right_speed: float) -> None:
        """Duty per wheel in [-1, 1]; arms the watchdog."""
        with self._lock:
            self._pi.write(self.stby, 1)

            # The direction pins are mirrored: AIN1 high reverses the left wheel, BIN1 high drives the right forward
            spd_l = int(max(0.0, min(1.0, abs(left_speed))) * 1000000)
            self._pi.hardware_PWM(self.pwma, self.pwm_freq, spd_l)
            self._pi.write(self.ain1, 1 if left_speed < 0 else 0)
            self._pi.write(self.ain2, 1 if left_speed > 0 else 0)

            spd_r = int(max(0.0, min(1.0, abs(right_speed))) * 1000000)
            self._pi.hardware_PWM(self.pwmb, self.pwm_freq, spd_r)
            self._pi.write(self.bin1, 1 if right_speed > 0 else 0)
            self._pi.write(self.bin2, 1 if right_speed < 0 else 0)
            self._driven_at = self._clock()
            if self.daemon is None:
                self._start_daemon()
            if self.daemon is not None:
                self.daemon.beat()
        self._start_watchdog()

    def stop(self) -> None:
        """Coast, standby, and end the watchdog."""
        with self._lock:
            self._pi.hardware_PWM(self.pwma, self.pwm_freq, 0)
            self._pi.hardware_PWM(self.pwmb, self.pwm_freq, 0)
            self._pi.write(self.ain1, 0)
            self._pi.write(self.ain2, 0)
            self._pi.write(self.bin1, 0)
            self._pi.write(self.bin2, 0)
            self._pi.write(self.stby, 0)
            self._driven_at = None
            if self.daemon is not None:
                self.daemon.close()
                self.daemon = None
        self._stop_watchdog()

    def brake(self) -> None:
        """
        Short brake: IN1 = IN2 = HIGH ties each motor's terminals together, so its back-EMF
        brakes it, where stop() lets it freewheel. STBY stays high. Disarms the watchdog.
        """
        with self._lock:
            self._short_brake()
            self._driven_at = None
            if self.daemon is not None:
                self.daemon.disarm()

    @property
    def watchdog_status(self) -> str:
        if self.watchdog_s is None:
            return "off"
        if self.daemon is not None:
            return "python + pigpiod"
        return f"python (pigpiod: {self._daemon_why})" if self._daemon_why else "python"

    def _start_daemon(self) -> None:
        """A refusal is remembered, not retried."""
        if not self._use_daemon or self.daemon is not None:
            return
        d = DaemonWatchdog(self._pi, {"stby": self.stby, "ain1": self.ain1, "ain2": self.ain2, "bin1": self.bin1,
                                      "bin2": self.bin2, "pwma": self.pwma, "pwmb": self.pwmb},
                           self.pwm_freq, self.watchdog_s)
        if d.on:
            self.daemon = d
        else:
            self._use_daemon, self._daemon_why = False, d.status

    def check(self) -> bool:
        """One watchdog look; True if it braked just now. Tests call it directly."""
        with self._lock:
            if self.watchdog_s is None or self._driven_at is None:
                return False
            quiet = self._clock() - self._driven_at
            if quiet < self.watchdog_s:
                return False
            self._short_brake()
            self._driven_at = None
            self.watchdog_trips += 1
            if self.daemon is not None:
                self.daemon.disarm()
        log.warning("motor watchdog: no drive() for %.2f s (limit %.2f s), motors braked: "
                    "the run loop is stuck (a camera that stopped delivering frames?)",
                    quiet, self.watchdog_s)
        return True

    def _short_brake(self) -> None:
        """The caller holds the lock."""
        self._pi.write(self.stby, 1)
        for pin in (self.ain1, self.ain2, self.bin1, self.bin2):
            self._pi.write(pin, 1)
        self._pi.hardware_PWM(self.pwma, self.pwm_freq, 1000000)
        self._pi.hardware_PWM(self.pwmb, self.pwm_freq, 1000000)

    def _start_watchdog(self) -> None:
        if self.watchdog_s is None or (self._wd_thread is not None and self._wd_thread.is_alive()):
            return
        self._wd_stop = threading.Event()        # its own, so a slow old thread stays stopped
        self._wd_thread = threading.Thread(target=self._watch, args=(self._wd_stop,),
                                           name="motor-watchdog", daemon=True)
        self._wd_thread.start()

    def _stop_watchdog(self) -> None:
        self._wd_stop.set()
        t = self._wd_thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=1.0)
        self._wd_thread = None

    def _watch(self, stop_evt: threading.Event) -> None:
        name_os_thread("motor-watchdog")
        period = self.watchdog_s / WATCHDOG_CHECKS
        while not stop_evt.wait(period):
            try:
                self.check()
            except Exception:                   # a failed brake must not end the watching
                log.exception("motor watchdog: check failed")
