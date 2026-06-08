# Gemma vLLM Runtime

The model weights are already stored locally under:

```text
hf-cache/google/gemma-3-4b-it
hf-cache/google/gemma-2-9b-it
```

Do not download them again inside Docker. Run vLLM by mounting the local cache:

```powershell
docker build -t graphmemory-gemma:dev .\infra\gemma
.\infra\gemma\run_gemma.ps1
```

The helper script starts or creates:

```text
jmlc-gemma-host
```

It mounts:

```text
.\hf-cache -> /models/huggingface
```

The OpenAI-compatible endpoint is:

```text
http://127.0.0.1:8000/v1
```

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/v1/models
```

Default served model:

```text
gemma-3-4b-it
```
