import litserve as ls  # type:ignore
import time
from dotenv import load_dotenv
from minion_service.api_server import VLLMOpenAIAPI

# Load environment variables from .env file
load_dotenv()


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
    api = VLLMOpenAIAPI(spec=ls.OpenAISpec())
    loggers = [AppLifecycleLogger(), PerformanceLogger()]

    server = ls.LitServer(api, devices="auto", workers_per_device=1, loggers=loggers)

    server.run()


if __name__ == "__main__":
    main()
