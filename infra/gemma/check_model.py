import os
import sys
from huggingface_hub import model_info, snapshot_download
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError, HfHubHTTPError


def main() -> int:
    model_id = os.getenv("GEMMA_MODEL", "google/gemma-2-9b-it").strip()
    hf_token = os.getenv("HF_TOKEN", "").strip()

    if not model_id:
        print("[GEMMA_CHECK] GEMMA_MODEL is empty", flush=True)
        return 2

    if not hf_token:
        print(
            "[GEMMA_CHECK] HF_TOKEN is missing. Gemma on Hugging Face usually requires accepting the license and using a token.",
            flush=True,
        )
        return 2

    print(f"[GEMMA_CHECK] Checking access to model: {model_id}", flush=True)

    try:
        info = model_info(model_id, token=hf_token)
        print(f"[GEMMA_CHECK] Model found: {info.modelId}", flush=True)
    except GatedRepoError as exc:
        print(
            f"[GEMMA_CHECK] Access denied/gated repo for {model_id}. Accept the Gemma license on Hugging Face first.",
            flush=True,
        )
        print(str(exc), flush=True)
        return 3
    except RepositoryNotFoundError as exc:
        print(f"[GEMMA_CHECK] Model repository not found: {model_id}", flush=True)
        print(str(exc), flush=True)
        return 4
    except HfHubHTTPError as exc:
        print(f"[GEMMA_CHECK] Hugging Face HTTP error while checking {model_id}", flush=True)
        print(str(exc), flush=True)
        return 5
    except Exception as exc:
        print(f"[GEMMA_CHECK] Unexpected error while checking model: {type(exc).__name__}: {exc}", flush=True)
        return 6

    should_download = os.getenv("PRELOAD_MODEL", "0").lower() in {"1", "true", "yes", "on"}
    if should_download:
        print(f"[GEMMA_CHECK] Preloading model snapshot: {model_id}", flush=True)
        try:
            path = snapshot_download(
                repo_id=model_id,
                token=hf_token,
                local_files_only=False,
            )
            print(f"[GEMMA_CHECK] Model snapshot available at: {path}", flush=True)
        except Exception as exc:
            print(f"[GEMMA_CHECK] Failed to preload model: {type(exc).__name__}: {exc}", flush=True)
            return 7

    print("[GEMMA_CHECK] OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())