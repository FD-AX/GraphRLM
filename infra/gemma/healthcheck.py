import os
import sys
import urllib.request
import urllib.error


def main() -> int:
    port = os.getenv("GEMMA_PORT", "8000")
    url = f"http://127.0.0.1:{port}/v1/models"

    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                return 0
            print(f"[GEMMA_HEALTHCHECK] Bad status: {resp.status}", flush=True)
            return 1
    except urllib.error.HTTPError as exc:
        print(f"[GEMMA_HEALTHCHECK] HTTP error: {exc.code}", flush=True)
        return 1
    except Exception as exc:
        print(f"[GEMMA_HEALTHCHECK] Not ready: {type(exc).__name__}: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())