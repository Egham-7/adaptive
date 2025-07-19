import litserve as ls  # type:ignore
import time
from api_server import LitGPTOpenAIAPI


class AppLifecycleLogger(ls.Logger):
    def process(self, key, value):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {key}: {value}", flush=True)


class PerformanceLogger(ls.Logger):
    def process(self, key, value):
        if isinstance(value, (int, float)):
            print(f"METRIC {key}: {value:.4f}", flush=True)
        else:
            print(f"INFO {key}: {value}", flush=True)


def main() -> None:

    api = LitGPTOpenAIAPI()
    loggers = [AppLifecycleLogger(), PerformanceLogger()]

    server = ls.LitServer(
        api,
        devices="auto",
        workers_per_device=2,
        loggers=loggers,
    )

    server.run(num_api_servers=4)


if __name__ == "__main__":
    main()
