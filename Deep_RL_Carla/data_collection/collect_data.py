#!/usr/bin/env python3
"""CLI entrypoint for the modular CARLA data collector."""

import signal

from carla_collector.config import parse_args
from carla_collector.runner import SessionRunner


def main():
    # SessionRunner bao mot hoac nhieu CarlaCollector. Voi --total-samples = 0
    # (mac dinh) no chay dung mot session roi thoat, y het hanh vi cu.
    runner = SessionRunner(parse_args())

    def stop_handler(_signum, _frame):
        runner.request_stop("user_interrupt")

    signal.signal(signal.SIGINT, stop_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, stop_handler)
    try:
        runner.run()
    except KeyboardInterrupt:
        runner.request_stop("user_interrupt")
    except Exception as exc:
        print("LOI: %s" % exc)
        raise


if __name__ == "__main__":
    main()
