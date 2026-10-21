"""SYS-A bench harness: battery monitor chain with a bench supply as the battery.

Purpose:
    Logs everything the power chain does while an operator steps a bench
    supply through the test voltages: every raw ADC sample the monitor takes,
    the smoothed voltage and state at 10 Hz, both encoder counts, each
    warning / critical / recovered event with its timestamp, and the operator's
    multimeter readings. analyze_sysa.py turns the log into the report figures.

    Raw samples come from wrapping Power._read_battery_voltage, so the log
    holds exactly the samples the state machine saw, with no extra I2C
    traffic on the bus the IMU shares.

Run (on the Pi, from vision_stack, WHEELS OFF THE GROUND):
    sudo pigpiod                       # if not already running
    python3 -m src.diagnostics.sysa_power_chain

Commands (type + Enter):
    s <volts>   you are changing the supply to this setpoint NOW
    m <volts>   multimeter reading on the supply output, once it is stable
    n <text>    free-text note (display reading, "sag-begin", "sag-end", ...)
    g / x       wheels spin at low duty / stop (spin loop honours should_stop())
    q           finish and write the log
"""

import argparse
import csv
import threading
import time
from datetime import datetime

import pigpio

from src.diagnostics.battery import Power
from src.peripherals.drive import EncoderReader, MotorController
from src.peripherals.system import System

SPIN_DUTY = 0.5            # x MOTOR_DUTY_SCALE in the driver
SPIN_PERIOD_S = 0.1        # keeps the motor watchdog fed
TRACE_PERIOD_S = 0.1


class Logger:
    COLUMNS = ["t", "kind", "value", "state", "left", "right", "note"]

    def __init__(self, path):
        self.t0 = time.monotonic()
        self._f = open(path, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(self.COLUMNS)
        self._lock = threading.Lock()

    def now(self):
        return time.monotonic() - self.t0

    def log(self, kind, value="", state="", left="", right="", note=""):
        row = [f"{self.now():.3f}", kind, value, state, left, right, note]
        with self._lock:
            self._w.writerow(row)
            self._f.flush()

    def close(self):
        with self._lock:
            self._f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="CSV path (default sysa_<timestamp>.csv)")
    args = ap.parse_args()
    path = args.out or datetime.now().strftime("sysa_%Y%m%d_%H%M%S.csv")

    pi = pigpio.pi()
    system = System()
    motors = MotorController(pi)
    enc = EncoderReader(pi)
    pwr = Power()
    lg = Logger(path)

    # Log every raw sample the monitor takes (including preflight and seed reads).
    orig_read = pwr._read_battery_voltage

    def logged_read():
        v = orig_read()
        lg.log("raw", f"{v:.4f}")
        return v

    pwr._read_battery_voltage = logged_read

    spinning = threading.Event()
    done = threading.Event()

    def on_warning():
        lg.log("event", state="WARNING", note="warning")
        system.show_text("Lo-b")

    def on_critical():
        motors.stop()
        lg.log("event", state="CRITICAL", note="critical")
        lg.log("event", note="motors_stop")
        system.show_text("Stop")

    def on_recovered():
        lg.log("event", state="OK", note="recovered")
        system.show_text("rdy ")

    def on_fault():
        lg.log("event", note="fault")
        system.show_text("Err ")

    pwr.on_warning(on_warning)
    pwr.on_critical(on_critical)
    pwr.on_recovered(on_recovered)
    pwr.on_fault(on_fault)

    ok, v = pwr.preflight()
    lg.log("event", f"{v:.3f}", note=f"preflight ok={ok}")
    print(f"preflight: {v:.2f} V -> {'OK' if ok else 'LOW'}")
    system.show_text("rdy " if ok else "Lo-b")
    pwr.start_monitoring()          # production default: 1 Hz sampling
    lg.log("event", note="monitoring_started")

    def trace_loop():
        while not done.is_set():
            left, right = enc.counts()
            lg.log("trace", f"{pwr.voltage():.4f}", pwr.state().name, left, right,
                   "spin" if spinning.is_set() else "")
            time.sleep(TRACE_PERIOD_S)

    def spin_loop():
        # Stands in for the navigation loop: drives while allowed, stops on should_stop().
        while not done.is_set():
            if spinning.is_set() and not pwr.should_stop():
                motors.drive(SPIN_DUTY, SPIN_DUTY)
            time.sleep(SPIN_PERIOD_S)

    threading.Thread(target=trace_loop, daemon=True).start()
    threading.Thread(target=spin_loop, daemon=True).start()

    print(__doc__.split("Commands")[1])
    try:
        while True:
            line = input("> ").strip()
            if not line:
                continue
            cmd, _, arg = line.partition(" ")
            if cmd == "s":
                lg.log("setpoint", arg)
            elif cmd == "m":
                lg.log("meter", arg)
            elif cmd == "n":
                lg.log("note", note=arg)
            elif cmd == "g":
                spinning.set()
                lg.log("note", note="wheels_on")
            elif cmd == "x":
                spinning.clear()
                motors.stop()
                lg.log("note", note="wheels_off")
            elif cmd == "q":
                break
            print(f"  V={pwr.voltage():.2f}  state={pwr.state().name}  sensor_ok={pwr.sensor_ok()}")
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        done.set()
        motors.stop()
        pwr.cleanup()
        enc.cancel()
        system.cleanup()
        lg.close()
        pi.stop()
        print(f"log written: {path}")


if __name__ == "__main__":
    main()
