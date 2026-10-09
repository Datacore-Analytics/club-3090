# Datacore Gemma 4 inference service

Gemma 4 26B-A4B on a single RTX 3090 (24 GB, TP=1), served through the existing
`vllm/gemma-26ba4b-single` slug
([`models/gemma-4-26b-a4b/vllm/compose/single/awq/int8.yml`](../models/gemma-4-26b-a4b/vllm/compose/single/awq/int8.yml)):
cyankiwi AWQ weights, vLLM v0.22.0 with the vendored INT8 per-token-head KV patch,
Google's canonical chat template, the external MTP assistant
`google/gemma-4-26B-A4B-it-assistant`, `--reasoning-parser gemma4` and prefix caching.
[`scripts/launch-datacore-gemma.sh`](../scripts/launch-datacore-gemma.sh) only pins the
launch settings below; it does not change the engine configuration.

| Setting | Value |
|---|---|
| `SPEC` / `SPEC_N` | `on` / `4` (MTP drafter depth) |
| `MAX_MODEL_LEN` | `12500` (prompt + output) |
| `GPU_MEMORY_UTILIZATION` | `0.94` |
| `MAX_NUM_SEQS` | `8` |
| `MAX_NUM_BATCHED_TOKENS` | `4096` |
| Tensor parallel | `1` |
| Port | `8040` (slug default) |

## Setup

Requires one RTX 3090 and Docker with the NVIDIA container runtime. Download the model
and the MTP assistant (the assistant is only fetched with the flag):

```bash
WITH_ASSISTANT_DRAFT=1 bash scripts/setup.sh gemma-4-26b-a4b
```

## Start and stop

```bash
bash scripts/launch-datacore-gemma.sh   # waits until /v1/models is ready
bash scripts/switch.sh --down           # stop
```

Any `switch.sh` flag passes through, e.g. `--no-wait`.

## API

- Base URL: `http://<host>:8040/v1` (OpenAI-compatible)
- Model: `gemma-4-26b-a4b-awq`

Thinking, its budget and structured output are set **per request**:

```bash
curl -s http://localhost:8040/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemma-4-26b-a4b-awq",
    "messages": [{"role": "user", "content": "Classify: \"Warung Kopi Sederhana\". Return JSON."}],
    "max_tokens": 1536,
    "chat_template_kwargs": {"enable_thinking": true},
    "thinking_token_budget": 512,
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "poi_category",
        "strict": true,
        "schema": {
          "type": "object",
          "properties": {"category": {"type": "string"}},
          "required": ["category"],
          "additionalProperties": false
        }
      }
    }
  }' | jq '.choices[0].message | {reasoning_content, content}'
```

The reasoning trace is returned in `reasoning_content` and the JSON answer in `content`.

`thinking_token_budget` caps only the reasoning section of that request: once it is
reached, vLLM ends the thinking block and the model moves on to the answer. It is not a
server-level output-token limit. Set `max_tokens` per request so it covers the
thinking budget plus the complete JSON answer; `MAX_MODEL_LEN` bounds prompt plus output.

## LoRA (Indonesia adapter)

LoRA is off by default; with it off the service is unchanged. When on, the same server
serves the base model `gemma-4-26b-a4b-awq` and the adapter `poi-ai-gemma-silver-v1`
side by side (both listed in `/v1/models`); each request picks one with `model`. All
settings above (AWQ, INT8-PTH KV, MTP n=4, context, batching, parser, template, prefix
caching) apply in both modes.

Place the PEFT adapter directory (`adapter_config.json`, `adapter_model.safetensors`)
inside the model cache, i.e. on the host at `<MODEL_DIR>/poi-ai-gemma-silver-v1`
(default `MODEL_DIR` is the repo's sibling `models-cache/`). No extra mount is needed.

| Server setting | Default (launcher) |
|---|---|
| `VLLM_ENABLE_LORA` | `false` |
| `VLLM_LORA_ADAPTER_NAME` | `poi-ai-gemma-silver-v1` |
| `VLLM_LORA_ADAPTER_PATH` | `/root/.cache/huggingface/poi-ai-gemma-silver-v1` (container path) |
| `VLLM_MAX_LORA_RANK` | `32` (must be >= the adapter's rank) |

```bash
VLLM_ENABLE_LORA=true bash scripts/launch-datacore-gemma.sh
```

When enabled, the entrypoint adds `--enable-lora --max-loras 1 --max-lora-rank <rank>
--lora-modules <name>=<path>` and refuses to boot if the adapter is missing.

## Datacore worker (external service)

Run `poi-ai-worker` against this service instead of its own vLLM.

Common:

```env
START_VLLM=false
VLLM_BASE_URL=http://<inference-host>:8040/v1
VLLM_MODEL=gemma-4-26b-a4b-awq
WORKER_PROCESSES=4
ENABLE_THINKING=true
THINKING_TOKEN_BUDGET=512
USE_STRUCTURED_OUTPUTS=true
LLM_TEMPERATURE=1.0
LLM_TOP_P=0.95
LLM_TOP_K=64
```

Global mode (base model; server may run with LoRA on or off):

```env
AI_WORKER_PROFILE=global_en
VLLM_ENABLE_LORA=false
MAX_OUTPUT_TOKENS=2500
```

Indonesia LoRA mode (server must run with `VLLM_ENABLE_LORA=true`):

```env
AI_WORKER_PROFILE=id_local
VLLM_ENABLE_LORA=true
VLLM_REQUEST_MODEL_ID_LOCAL=poi-ai-gemma-silver-v1
MAX_OUTPUT_TOKENS=2675
```

On the worker, `VLLM_ENABLE_LORA` only selects the request model:
`VLLM_REQUEST_MODEL_ID_LOCAL` for `id_local` batches, `VLLM_MODEL` otherwise. It must
equal the server's `VLLM_LORA_ADAPTER_NAME`. The worker sends
`thinking_token_budget` per request; the server has no thinking setting.
