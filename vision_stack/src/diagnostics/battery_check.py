"""Pre-run battery check on the TM1637 display.

Purpose:
    Shows whether the pack is good before a run, and keeps watching it.
    Run on the Pi with the motors idle (needs sudo pigpiod, like the rest
    of the stack):

        python3 -m src.diagnostics.battery_check

    Takes a resting-voltage reading with Power.preflight() and shows "Lo-b"
    below the warning threshold, else "rdy ". It then keeps monitoring:
    WARNING shows "Lo-b", CRITICAL shows "Stop", an ADC fault shows "Err".
    This is a standalone check; it does not run alongside src.main, which
    owns the display during a run.
"""

import time

from src.diagnostics.battery import Power
from src.peripherals.system import System


def main() -> None:
    system = System()
    pwr = Power()

    ok, v = pwr.preflight()
    print(f"Resting voltage: {v:.2f}V -> {'OK' if ok else 'LOW'}")
    system.show_text("rdy " if ok else "Lo-b")

    pwr.on_warning(lambda: system.show_text("Lo-b"))
    pwr.on_critical(lambda: system.show_text("Stop"))
    pwr.on_recovered(lambda: system.show_text("rdy "))
    pwr.on_fault(lambda: system.show_text("Err "))
    pwr.start_monitoring()

    try:
        while True:
            print(f"{pwr.voltage():.2f}V  {pwr.percentage():.0f}%  "
                  f"{pwr.state().name}  sensor_ok={pwr.sensor_ok()}")
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        pwr.cleanup()
        system.cleanup()


if __name__ == "__main__":
    main()
