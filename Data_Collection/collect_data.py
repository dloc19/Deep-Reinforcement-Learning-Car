#!/usr/bin/env python3
"""CLI entrypoint for the modular CARLA data collector."""

import signal

from carla_collector.collector import CarlaCollector
from carla_collector.config import parse_args


def main():
    collector = CarlaCollector(parse_args())

    def stop_handler(_signum, _frame):
        collector.stop_event.set()

    signal.signal(signal.SIGINT, stop_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, stop_handler)
    try:
        collector.run()
    except KeyboardInterrupt:
        collector.stop_event.set()
    except Exception as exc:
        print("LOI: %s" % exc)
        raise
    finally:
        collector.cleanup()


if __name__ == "__main__":
    main()
