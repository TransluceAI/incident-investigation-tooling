# Request and result format

The runner is the only GPU code. Applications such as the judges never fork it:
they write request rows, run this tool, and read result rows. The runner knows
nothing about an application's prompts, labels or citations.

A manifest contains either text rows or request rows, never both.

## Text rows

`unit_id`, `chunk_idx`, `text` and `text_sha256`. The config supplies
`system_prompt` and `schema`; the text is wrapped in `<document>` tags. This is
the original format and is unchanged.

## Request rows

Each line is one self-contained chat request in the batch shape used by vLLM
`run-batch` and hosted batch APIs:

```json
{"custom_id": "item7:w0:hacking",
 "body": {"messages": [{"role": "system", "content": "..."},
                       {"role": "user", "content": "..."}],
          "response_format": {"type": "json_schema",
                              "json_schema": {"name": "hacking", "schema": {"type": "object"}}},
          "logprobs": true,
          "top_logprobs": 10}}
```

- `custom_id`: unique, nonempty string. Results are keyed by it; never rely on line order.
- `body.messages`: nonempty list of `{role, content}` with role `system`, `user` or `assistant`.
- `body.response_format`: optional JSON Schema for this request. Without it the
  config's `schema` applies. Each distinct schema is checked and compiled once.
- `body.logprobs` / `body.top_logprobs`: optional; `top_logprobs` (0–20) requires
  `logprobs: true`.
- Optional `method: "POST"` and `url: "/v1/chat/completions"` are accepted so
  OpenAI-style batch lines need no rewriting. Other keys, including sampling
  settings, are rejected: sampling, thinking and decoding controls come from the
  config profile so that one run uses one setting.

The runner hashes each body: `request_sha256` is the SHA-256 of the UTF-8 JSON
with sorted keys, `(',', ':')` separators and non-ASCII characters unescaped
(Python `json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(',', ':'))`).
Resume compares that hash, so a changed request cannot reuse an earlier answer,
and applications can recompute it to bind each result to the request they wrote. Keep requests that
share a long prefix adjacent in the file so they share a batch and vLLM's prefix
cache. `citation_constraints` applies to text rows only; an application that
wants narrowed citations writes the narrowed schema into each request.

## Result rows

Workers write `outputs/worker-<id>/part-*.jsonl`. Request results carry
`custom_id`, `request_sha256`, `status` (`success`/`failed`), `result` (parsed JSON
or null), `error`, `raw_output`, `reasoning`, `final_output`, token counts and
`finish_reason`, plus `attempt`/`previous_attempts` when retries are configured.
`unit_id` equals `custom_id`.

When a request asked for logprobs, a successful row also has
`logprobs: {"content": [{"token", "logprob", "top_logprobs": [{"token", "logprob"}]}]}`
covering the final answer only (reasoning tokens are excluded). Candidates are
those vLLM returned: the top `top_logprobs` plus the sampled token. Missing or
non-finite values are `null`; a failed row has `logprobs: null`. Turning these
into label scores is the application's job.

`examples/requests.jsonl` is a synthetic request for preflight checks.
