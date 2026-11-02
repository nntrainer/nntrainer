# Native CausalLM serving

Build and run through the existing C++ application:

```sh
meson setup --reconfigure build -Denable-transformer=true -Denable-causallm-server=true
ninja -C build Applications/CausalLM/nntr_causallm
build/Applications/CausalLM/nntr_causallm /path/to/model --serve \
  --host 127.0.0.1 --port 8080 --model-id local-model \
  --max-sessions 32 --queue-capacity 32
```

The HTTP server, request queue, model execution, sessions and KV management all
run inside this process. There is no Python runtime or subprocess protocol.
Server support is optional at build time. The server defaults to CPU, paged KV
and prefix reuse when those environment variables are unset. Explicitly disabling
prefix reuse is supported; selecting a non-CPU or legacy KV backend is rejected.
The model must use batch size 1. Model loading completes before the socket binds.

## Architecture

`HttpServer` parses HTTP/JSON with cpp-httplib, using a bounded pool of HTTP
workers. `Service` owns a bounded FIFO of tasks and a single inference worker.
All session changes, template rendering and model/cache operations execute on
that worker; an HTTP thread never calls the model. Queue saturation returns 503.

Logical sessions own native `KVCacheManager` session IDs. The native default
session stays empty so any conversation can be deselected and destroyed.
Sessions share the model weights and graph; native paged storage owns their KV
pages. The first request settles the prefill plane before allocating storage.

Every completion contains the **full prompt or complete message history**.
Chat requests use the model's existing `ChatTemplate`; no generic role delimiters
are substituted if the model has no template. Each run starts at logical
position zero. The existing token-prefix cache compares actual token IDs and
reuses matching committed pages, including across sessions. An edited history
therefore cannot blindly append to an unrelated cache. Usage reports the native
cache's reused token count. Without prefix caching, the full prompt is recomputed.

This separation follows llama.cpp's HTTP routes, request queue, inference context
and sequence cache design:
https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README-dev.md
The current CausalLM execution API runs a complete generation per task; requests
execute in FIFO order rather than continuous batching.

## OpenAI-compatible endpoints

- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `GET /health`

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local-model","messages":[{"role":"user","content":"Hello"}],"max_tokens":64,"temperature":0,"stream":false}'
```

Chat Completions clients can point their `base_url` at
`http://127.0.0.1:8080/v1` and use the advertised model ID. Responses include
`id`, `object`, `created`, `model`, `choices`, `finish_reason`, and token `usage`.
Errors use the OpenAI `error` object shape. The optional extra `session_id`
response field identifies the cache session.

Supported inputs: `model`, text `messages` (system/developer/user/assistant),
string `prompt` for Completions, `max_tokens` or `max_completion_tokens`,
`temperature`, `top_p`, `seed`, `n=1`, boolean `stream`, and `user`.
The local `chat_template_kwargs` extension accepts boolean `enable_thinking`
for model templates such as Qwen3; clients can pass it in their extra body fields.
Unsupported parameters are rejected rather than silently ignored. This is text
Chat Completions/Completions compatibility; Responses, tools, multimodal content,
stop sequences, penalties and logprobs are not implemented.
The request body limit is 1 MiB. Input exceeding the context or prefill window
is rejected instead of truncated. An inference failure invalidates and releases
its session; validation errors preserve an existing session.

## Sessions and reclamation

Standard OpenAI clients need no session extension. Their requests allocate
reclaimable cache sessions, and the native prefix cache may reuse a matching
history. The least recently used reclaimable session is released when logical
session capacity is reached.

For explicit retention:

```sh
curl -X POST http://127.0.0.1:8080/sessions
# {"id":"session-1","object":"session"}
```

Send that ID in the optional `session_id` completion field. Continue sending the
**complete history**, including assistant messages. Explicit sessions are pinned
until deleted or process exit. If every session is pinned, creation returns 503.
`GET /sessions` lists sessions; `DELETE /sessions/{id}` releases one. This is a
local extension, not an OpenAI session endpoint. Native shared prefix pages can
remain resident according to the prefix cache's own bounded eviction policy.
Queue shutdown drains accepted work, then releases sessions on the model worker.

## Verification

```sh
ninja -C build Applications/CausalLM/unittest_serving
build/Applications/CausalLM/unittest_serving
```

The C++ tests cover request/response schemas, full chat histories, invalid fields,
queue saturation, session capacity/LRU, failure cleanup, native HTTP requests,
and actual tiny-Qwen2 paged cache execution across interleaved and edited
histories. The HTTP test requires permission to bind a localhost socket.

The opt-in full-size integration test `Serving.Qwen3_06B_NativeHttp` loads
`<builddir>/models/qwen3-0.6b-fp32` when available. Set
`NNTR_SERVING_QWEN3_DIR` to select another converted Qwen3 0.6B directory. It
tests real HTTP chat completions, two interleaved conversations, history replay,
cache reuse, context rejection, streaming, cancellation and session deletion. It writes
its responses to `/tmp/nntrainer-qwen3-serving-results.json`.

## Streaming

Both completion endpoints accept `stream:true`. Chat responses use
`chat.completion.chunk` with an initial assistant role delta, content deltas,
a final `finish_reason`, and `data: [DONE]`. Text completions use `text` chunks.
`stream_options:{"include_usage":true}` adds a final usage chunk with empty
`choices` before `[DONE]`; other chunks then carry `usage:null`.

```sh
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local-model","messages":[{"role":"user","content":"Hello"}],"stream":true,"stream_options":{"include_usage":true},"max_tokens":64}'
```

Every request owns a bounded token mailbox (128 events, 256 KiB). Model callbacks
never block on client reads. Overflow cancels that generation. Disconnects are
checked while queued and during SSE writes/heartbeats; cancellation is observed
by the model at its next decoded-output boundary (prefill is not interrupted).
An active cancelled generation's session, including a pinned session, is deleted
on the inference worker. A cancellation before execution leaves an existing
session intact. Other sessions and requests are unaffected; recreate a deleted
pinned session before retrying. Completed streams retain their session normally.

Validation errors before generation starts retain HTTP status/error JSON. Once
SSE starts, inference failures are delivered as an SSE OpenAI-shaped error then
`[DONE]`, without a successful finish chunk. EOS marker text is held across token
callbacks and removed before emission, while preserving UTF-8 boundaries.
Socket writes have a 30-second timeout. Shutdown still drains accepted tasks;
stream queues bound memory even if a reader disappears.

Streaming response format reference:
https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events
