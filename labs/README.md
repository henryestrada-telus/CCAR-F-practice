# Labs

Hands-on labs: a way to practise without an API key.

**You do not need an API key.** Each lab ships with `shopassist_lab.py`, a small
standard-library-only simulator of the `anthropic` SDK. The code you write is exactly the
code you would write against the real API; the replies are canned fixtures chosen by rules,
not a real model. Set `FUELIX_API_KEY` in your environment (or in a `.env` file) and the
very same notebook calls Claude for real.

## How to run a lab

```bash
cd labs/lab_first_request
jupyter lab            # or open notebook.ipynb in VS Code
```

Open **`notebook.ipynb`** and work through it top to bottom. Every cell already holds the
code, with blanks marked `...` and a comment saying what goes in each one. Each task ends
with a `check(...)` call that prints PASS with an explanation or FAIL with a specific hint.
`START_HERE.md` in the lab folder has the details; `solution.ipynb` is the completed
notebook if you get stuck.

The only requirement is Python 3.10+ and Jupyter (`pip install jupyter`). The `anthropic`,
`python-dotenv`, `fastapi` and `pydantic` packages are optional: if they are missing, the simulator stands in for
them. An installed `anthropic` is used only when `FUELIX_API_KEY` is set (in the
environment or in `.env`); without a key the simulator answers, so no lab fails with an
authentication error.

## The labs

In course order. Each lab mirrors a script in [`shopassist/`](../shopassist/).

| Lab | Practises | Script |
| --- | --- | --- |
| `lab_first_request` | Your first `client.messages.create(...)`: `content`, `stop_reason`, `usage`, `max_tokens` | `first_request.py` |
| `lab_conversation` | The conversation is a list in your code, sent in full every turn; input tokens grow | `conversation.py` |
| `lab_chat_server` | A server between the chat page and Claude: the `/chat` endpoint keeps the history and the rules | `app.py` |
| `lab_system_prompt` | Role and scope in the system prompt; a rule that must hold, checked in code | `rules.py` |
| `lab_model_settings` | A model and an effort level for each step; check `stop_reason` before the customer sees a reply | - |
| `lab_evaluation` | Two system prompts on the same test cases, scored by code | `evaluate.py` |
| `lab_tool_call` | One tool call end to end: `tool_use`, your function, `tool_result`, the answer | `tool_call.py` |
| `lab_structured_output` | Schema-valid JSON with `output_config.format`; valid is not the same as correct | `extract.py` |
| `lab_validation` | A nullable field, validation in code, a retry with the exact error | `validate.py` |
| `lab_human_review` | Field-level confidence calibrated on labeled data, accuracy per group, stratified sampling | - |
| `lab_extract_returns` | BUILD: five emails in, `return_requests.json` out, a status per record | `build_extract.py` |
| `lab_agentic_loop` | The loop: run every `tool_use` block, return every result, repeat until `end_turn` | `agent_loop.py` |
| `lab_tool_errors` | Not found is data; a real failure is a structured error with `is_error` | `tool_errors.py` |
| `lab_support_agent` | BUILD: case facts in code, a hook before every tool call, a handoff with `escalate` | `agent.py` |
| `lab_coordinator` | A coordinator and two subagents with their own tools; context passed explicitly | `coordinator.py` |

## The simulator follows the current API

The labs use `claude-sonnet-5-5` and `output_config={"effort": "low"}`, as the lectures
do. Like the real API, the simulator rejects `temperature`, `top_p`, `top_k`, an assistant
prefill and a forced `tool_choice` on current Opus, Sonnet and Fable models. It also
enforces the tool protocol: every `tool_use` needs its `tool_result` in the next message.
A mistake you make offline gives the same error the API would.
