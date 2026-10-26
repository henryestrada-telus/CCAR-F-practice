"""
shopassist_lab - an offline Claude simulator for the ShopAssist AI course labs.

WHAT THIS IS
------------
A single-file, standard-library-only stand-in for the `anthropic` SDK, so that the
course's lab exercises run inside a Udemy workspace that has no API key, no network
access, and no `pip install`.

    import shopassist_lab            # installs the shims, nothing else needed

    from dotenv import load_dotenv   # works even without python-dotenv installed
    from anthropic import Anthropic  # works even without anthropic installed
    load_dotenv()
    client = Anthropic()

Every line after that first import is IDENTICAL to the real course code. Take your
finished lab notebook home, add a real FUELIX_API_KEY, and the same cells call the
real Claude API - this module steps out of the way automatically.

WHAT THIS IS NOT
----------------
This is NOT Claude. Responses are canned fixtures chosen by a rules table. The
simulator has no language model in it and never touches the network.

It is honest about that in four ways you cannot miss:
  * a banner printed once, when you construct the client
  * every simulated message id starts with "msg_sim_"
  * every simulated message has `.simulated == True` (real ones do not)
  * `explain(message)` names the exact fixture it chose and why

WHAT IS FAITHFULLY SIMULATED
----------------------------
The parameters matter. That is the whole point - a mock that ignores your prompt
teaches you nothing.

    max_tokens      truncates the reply and sets stop_reason="max_tokens"
    output_config   effort is recorded; format (json_schema) returns JSON that fits
    system          rules detected in your system prompt change the reply
    tools           tool_use blocks with schema-valid input; the next tool is chosen
                    from the tool_result content you send back, errors included
    tool protocol   a tool_result must answer a tool_use from the turn before, as
                    the API requires
    retired params  temperature, top_p, top_k, a prefill, and tool_choice "any" or
                    a forced tool are rejected by current Opus, Sonnet and Fable
                    models, as the API does; Haiku still accepts them
    stop_sequences  cuts at the marker, sets stop_reason="stop_sequence"

Call `lab_info()` for the full disclosure, including the known gaps.

Licensed for use with the "Claude Certified Architect Foundations" course.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import math
import os
import random
import re
import sys
import types
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Anthropic",
    "load_dotenv",
    "install",
    "check",
    "check_all",
    "explain",
    "lab_info",
    "reset",
    "verbose",
    "register_scenario",
    "LabConfig",
    "Message",
    "TextBlock",
    "ToolUseBlock",
    "Usage",
    "APIError",
    "BadRequestError",
    "AuthenticationError",
    "browser_post",
]

__version__ = "1.0.0"


# ---------------------------------------------------------------------------
# 1. Configuration and mode
# ---------------------------------------------------------------------------


class LabConfig:
    """Tunables. Instructors may change these; students should not need to."""

    chars_per_token = 4          # how max_tokens is converted into a character budget
    max_tool_turns = 12          # hard stop so a broken agentic loop cannot hang the kernel
    tool_preambles = True        # emit a TextBlock alongside tool_use when tool_choice="auto"
    include_optional_tool_inputs = False   # synthesize only `required` fields by default
    verbose = False              # print a one-line trace per simulated call


_BANNER_SHOWN = False
_WARNED: set = set()


def verbose(on: bool = True) -> None:
    """Print a one-line trace for every simulated call. Useful when recording a lesson."""
    LabConfig.verbose = bool(on)


def _warn_once(msg: str) -> None:
    if msg not in _WARNED:
        _WARNED.add(msg)
        print("[shopassist_lab] note: " + msg)


def _module_importable(name: str) -> bool:
    if name in sys.modules:
        return True
    try:
        __import__(name)
        return True
    except Exception:
        return False


FUELIX_BASE_URL = "https://api.fuelix.ai"


def _fuelix_key() -> str:
    """The Fuel iX API key from the environment, or "" when none is usable."""
    key = os.environ.get("FUELIX_API_KEY", "").strip()
    if key.lower().startswith(("your", "paste", "changeme")):
        return ""
    return key


def _export_fuelix_env() -> None:
    """Let a bare, real ``anthropic.Anthropic()`` reach Fuel iX too.

    With a Fuel iX key present the shim is not installed and the notebook gets the
    real SDK class directly, so give the SDK its own variables for the same thing.
    """
    key = _fuelix_key()
    if key:
        os.environ.setdefault("ANTHROPIC_BASE_URL", os.environ.get("FUELIX_BASE_URL", FUELIX_BASE_URL))
        os.environ.setdefault("ANTHROPIC_AUTH_TOKEN", key)


def _has_real_key() -> bool:
    # An explicit override, so an instructor who has a key on their machine can still
    # record or verify the offline experience the students will actually get.
    if os.environ.get("SHOPASSIST_LAB_FORCE_SIM", "").strip().lower() in ("1", "true", "yes"):
        return False
    return bool(_fuelix_key())


def _fuelix_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Point the real SDK at the Fuel iX gateway unless the caller chose otherwise.

    Fuel iX serves the native Anthropic Messages API, so the official SDK is used
    as-is; only the address and the credential differ. The key travels as a bearer
    token (auth_token), not as the x-api-key header an Anthropic key would use.
    """
    out = dict(kwargs)
    out.setdefault("base_url", os.environ.get("FUELIX_BASE_URL", FUELIX_BASE_URL))
    if "api_key" not in out:
        out.setdefault("auth_token", _fuelix_key())
    return out


# ---------------------------------------------------------------------------
# 2. Response model - duck-types anthropic.types without pydantic
# ---------------------------------------------------------------------------


class _LabModel:
    """A minimal stand-in for a pydantic BaseModel. No third-party dependencies."""

    _fields: Tuple[str, ...] = ()

    def model_dump(self, *, exclude_none: bool = False, **_: Any) -> dict:
        out: Dict[str, Any] = {}
        for f in self._fields:
            v = getattr(self, f, None)
            if v is None and exclude_none:
                continue
            if isinstance(v, list):
                out[f] = [
                    x.model_dump(exclude_none=exclude_none) if isinstance(x, _LabModel) else x
                    for x in v
                ]
            elif isinstance(v, _LabModel):
                out[f] = v.model_dump(exclude_none=exclude_none)
            else:
                out[f] = v
        return out

    def model_dump_json(self, *, indent: Optional[int] = None,
                        exclude_none: bool = False, **_: Any) -> str:
        return json.dumps(self.model_dump(exclude_none=exclude_none),
                          indent=indent, default=str)

    # pydantic v1 aliases students may reach for
    def dict(self, **kw: Any) -> dict:          # noqa: A003
        return self.model_dump(**kw)

    def json(self, **kw: Any) -> str:           # noqa: A003
        return self.model_dump_json(**kw)

    # forgiving dict-ish access (a deliberate superset of the real SDK)
    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def get(self, k: str, default: Any = None) -> Any:
        return getattr(self, k, default)

    def __contains__(self, k: str) -> bool:
        return k in self._fields

    def __repr__(self) -> str:
        inner = ", ".join("{}={!r}".format(f, getattr(self, f, None)) for f in self._fields)
        return "{}({})".format(type(self).__name__, inner)


class TextBlock(_LabModel):
    _fields = ("citations", "text", "type")

    def __init__(self, text: str) -> None:
        self.citations = None
        self.text = text
        self.type = "text"


class ToolUseBlock(_LabModel):
    _fields = ("id", "input", "name", "type")

    def __init__(self, id: str, name: str, input: dict) -> None:  # noqa: A002
        self.id = id
        self.input = input
        self.name = name
        self.type = "tool_use"


class Usage(_LabModel):
    _fields = ("cache_creation_input_tokens", "cache_read_input_tokens",
               "input_tokens", "output_tokens", "service_tier")

    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.cache_creation_input_tokens = 0
        self.cache_read_input_tokens = 0
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.service_tier = "standard"


class Message(_LabModel):
    _fields = ("id", "content", "model", "role", "stop_reason",
               "stop_sequence", "type", "usage")

    #: Present only on simulated messages. Real anthropic Messages have no such attribute.
    simulated = True

    def __init__(self, id: str, content: list, model: str, stop_reason: str,  # noqa: A002
                 stop_sequence: Optional[str], usage: Usage) -> None:
        self.id = id
        self.content = content
        self.model = model
        self.role = "assistant"
        self.stop_reason = stop_reason
        self.stop_sequence = stop_sequence
        self.type = "message"
        self.usage = usage
        self._lab: Dict[str, Any] = {}

    @property
    def text(self) -> str:
        """Convenience shortcut, not part of the real SDK."""
        return "".join(b.text for b in self.content if b.type == "text")

    def __repr__(self) -> str:
        kinds = ", ".join(b.type for b in self.content) or "empty"
        return ("Message(id={!r}, simulated=True, stop_reason={!r}, "
                "content=[{}], model={!r})".format(
                    self.id, self.stop_reason, kinds, self.model))


# ---------------------------------------------------------------------------
# 3. Errors
# ---------------------------------------------------------------------------


class APIError(Exception):
    """Base class, mirroring anthropic.APIError."""


class BadRequestError(APIError):
    """400-equivalent, mirroring anthropic.BadRequestError."""


class AuthenticationError(APIError):
    """401-equivalent, mirroring anthropic.AuthenticationError."""


# ---------------------------------------------------------------------------
# 4. Determinism helpers
# ---------------------------------------------------------------------------


def est_tokens(text: str) -> int:
    """Rough token estimate. Real tokenizers differ; this is close enough to teach with."""
    return max(1, int(len(text) / LabConfig.chars_per_token)) if text else 0


def _fingerprint(*parts: Any) -> str:
    blob = "␟".join(json.dumps(p, sort_keys=True, default=str) for p in parts)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _rng(fingerprint: str, salt: int) -> random.Random:
    seed = int(fingerprint[:16], 16) ^ (salt * 0x9E3779B1)
    return random.Random(seed)


_LAST_VARIANT: Dict[str, int] = {}


def pick_variant(variants: Sequence[str], temperature: float,
                 rng: random.Random, fingerprint: str) -> Tuple[str, int]:
    """Choose a response variant.

    temperature == 0 always returns variant 0, byte for byte.
    temperature > 0 samples with a softmax over variant rank, then refuses to repeat
    the immediately previous choice - real Claude at temperature 1 essentially never
    returns the identical string twice in a row either.
    """
    if len(variants) == 1 or temperature <= 0:
        _LAST_VARIANT[fingerprint] = 0
        return variants[0], 0

    weights = [math.exp(-i / (temperature * 2.0)) for i in range(len(variants))]
    previous = _LAST_VARIANT.get(fingerprint)

    def sample() -> int:
        target, acc = rng.random() * sum(weights), 0.0
        for i, w in enumerate(weights):
            acc += w
            if target <= acc:
                return i
        return len(variants) - 1

    idx = sample()
    if previous is not None and idx == previous:
        alternatives = [i for i in range(len(variants)) if i != previous]
        idx = alternatives[rng.randrange(len(alternatives))]

    _LAST_VARIANT[fingerprint] = idx
    return variants[idx], idx


def reset(seed: Optional[int] = None) -> None:
    """Forget the call history, so a temperature>0 sequence replays identically."""
    _LAST_VARIANT.clear()
    for client in _CLIENTS:
        client._calls = 0
    if seed is not None:
        random.seed(seed)


_CLIENTS: List["MockAnthropic"] = []


# ---------------------------------------------------------------------------
# 5. Prompt analysis
# ---------------------------------------------------------------------------


FLAG_PATTERNS: Dict[str, str] = {
    "no_refund_promise": r"do ?n(?:o|')?t promise a refund|not promise a refund"
                         r"|do ?n(?:o|')?t approve a refund|until the order is checked",
    "ask_one_question": r"one (?:clear )?question at a time",
    "ask_for_order": r"ask for the order (?:number|id)|order number if it is missing",
    "concise": r"\bconcise\b|keep (?:it|the response) (?:short|under)|under \d+ words",
    "polite": r"\bpolite\b|\bcalm\b",
    "escalate_on_risk": r"escalate if|fraud or legal|legal action",
    "json_only": r"return only (?:a )?valid json|only valid json|do not include markdown",
    "no_preamble": r"return only the message|no explanations|do not include explanations",
}

INTENT_PATTERNS: List[Tuple[str, str]] = [
    ("billing_issue", r"charged twice|double charge|duplicate charge|billed twice|billing"),
    ("damaged_item", r"damaged|arrived broken|\bbroken\b|scratched|defective|cracked"
                     r"|do(?:es)? ?n(?:o|')?t work"),
    ("refund_request", r"\brefund\b|money back|want to return|return my order"
                       r"|return an order|wants to return|about a return"),
    ("order_status", r"where is my order|has ?n(?:o|')?t arrived|tracking"
                     r"|shipping status|supposed to arrive"),
    ("order_number", r"\border\s*(?:number|id|#)\b|^\s*my order number is"),
    ("product_question", r"how do i use|compatible with|does it work with"),
    ("greeting", r"^\s*(?:hi|hello|hey)\b"),
]

_XML = re.compile(r"<(\w+)>(.*?)</\1>", re.S)
_WORD_LIMIT = re.compile(r"under (\d+) words", re.I)

SLOT_PATTERNS: Dict[str, str] = {
    "order_id": r"\b(ORD-[A-Z0-9]{3,})\b|\border\s*(?:number|id|#)?\s*(?:is)?\s*[:# ]?\s*(\d{4,})\b",
    "email": r"([\w.+-]+@[\w-]+(?:\.[\w-]+)+)",
    "customer_id": r"\b(CUS-\d+)\b",
}


@dataclass(frozen=True)
class PromptFacts:
    has_system: bool
    system_text: str
    flags: frozenset
    intents: Tuple[str, ...]
    xml_tags: Dict[str, str]
    has_xml: bool
    word_limit: Optional[int]
    effective_user_text: str
    turns: int
    #: (role, text) for every message in order - what multi-turn replies read.
    history: Tuple[Tuple[str, str], ...] = ()


def _flatten(content: Any) -> str:
    """Turn a system= or content= value into plain text, whatever shape it arrived in."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return str(content.get("text", ""))
    if isinstance(content, (list, tuple)):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif getattr(item, "type", None) == "text":
                parts.append(getattr(item, "text", ""))
        return "\n".join(parts)
    return str(content)


def _last_user_text(messages: Sequence[Any]) -> str:
    for msg in reversed(list(messages)):
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        if role == "user":
            body = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
            text = _flatten(body)
            if text.strip():
                return text
    return ""


def detect_flags(*texts: str) -> frozenset:
    blob = "\n".join(t for t in texts if t)
    found = {name for name, pat in FLAG_PATTERNS.items()
             if re.search(pat, blob, re.I)}
    return frozenset(found)


def detect_intents(text: str) -> Tuple[str, ...]:
    hits = []
    for name, pat in INTENT_PATTERNS:
        m = re.search(pat, text, re.I)
        if m:
            hits.append((m.start(), name))
    hits.sort()
    return tuple(name for _, name in hits)


def extract_slots(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for slot, pat in SLOT_PATTERNS.items():
        m = re.search(pat, text, re.I)
        if m:
            value = next((g for g in m.groups() if g), None)
            if value:
                out[slot] = value
    return out


def analyze_prompt(system_text: str, messages: Sequence[Any]) -> PromptFacts:
    user_text = _last_user_text(messages)

    tags = {k.lower(): v.strip() for k, v in _XML.findall(user_text)}
    effective = tags.get("customer_message") or tags.get("user_message") or user_text

    limit_match = _WORD_LIMIT.search(user_text) or _WORD_LIMIT.search(system_text)

    return PromptFacts(
        has_system=bool(system_text.strip()),
        system_text=system_text,
        flags=detect_flags(system_text, user_text),
        intents=detect_intents(effective),
        xml_tags=tags,
        has_xml=bool(tags),
        word_limit=int(limit_match.group(1)) if limit_match else None,
        effective_user_text=effective,
        turns=len(list(messages)),
        history=tuple(
            (str((m.get("role") if isinstance(m, dict) else getattr(m, "role", "")) or ""),
             _flatten(m.get("content") if isinstance(m, dict) else getattr(m, "content", None)))
            for m in messages),
    )


# ---------------------------------------------------------------------------
# 6. Blackboard - what the transcript already established
# ---------------------------------------------------------------------------


def _block_type(block: Any) -> str:
    if isinstance(block, dict):
        return str(block.get("type", ""))
    return str(getattr(block, "type", ""))


def _block_get(block: Any, key: str, default: Any = None) -> Any:
    if isinstance(block, dict):
        return block.get(key, default)
    return getattr(block, key, default)


@dataclass
class Blackboard:
    turns: int = 0
    tools_called: List[str] = field(default_factory=list)
    facts: Dict[str, Any] = field(default_factory=dict)
    assistant_turns: int = 0


def build_blackboard(messages: Sequence[Any]) -> Blackboard:
    bb = Blackboard()
    for msg in messages:
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        body = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
        bb.turns += 1
        if role == "assistant":
            bb.assistant_turns += 1
        if isinstance(body, str):
            bb.facts.update(extract_slots(body))
            continue
        for block in (body or []):
            btype = _block_type(block)
            if btype == "text":
                bb.facts.update(extract_slots(_block_get(block, "text", "") or ""))
            elif btype == "tool_use":
                bb.tools_called.append(str(_block_get(block, "name", "")))
                for k, v in (_block_get(block, "input") or {}).items():
                    if v not in (None, ""):
                        bb.facts[k] = v
    return bb


# ---------------------------------------------------------------------------
# 7. Scenario table  (DATA - this is what grows when a lesson is added)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Match:
    intent: Tuple[str, ...] = ()
    requires_flags: Tuple[str, ...] = ()
    forbids_flags: Tuple[str, ...] = ()
    has_system: Optional[bool] = None
    has_tools: Optional[bool] = None
    user_regex: Optional[str] = None
    max_turns: Optional[int] = None
    min_turns: Optional[int] = None
    knows: Tuple[str, ...] = ()
    lacks: Tuple[str, ...] = ()

    def specificity(self) -> int:
        score = 0
        for value in dataclasses.astuple(self):
            if value is None:
                continue
            if isinstance(value, tuple):
                score += len(value)
            else:
                score += 1
        return score


@dataclass(frozen=True)
class Scenario:
    id: str
    lesson: str
    when: Match
    variants: Tuple[str, ...]
    note: str = ""
    priority: int = 0


def _matches(m: Match, facts: PromptFacts, bb: Blackboard, has_tools: bool) -> bool:
    if m.intent and not any(i in facts.intents for i in m.intent):
        return False
    if any(f not in facts.flags for f in m.requires_flags):
        return False
    if any(f in facts.flags for f in m.forbids_flags):
        return False
    if m.has_system is not None and m.has_system != facts.has_system:
        return False
    if m.has_tools is not None and m.has_tools != has_tools:
        return False
    if m.user_regex and not re.search(m.user_regex, facts.effective_user_text, re.I):
        return False
    if m.max_turns is not None and facts.turns > m.max_turns:
        return False
    if m.min_turns is not None and facts.turns < m.min_turns:
        return False
    if any(k not in bb.facts for k in m.knows):
        return False
    if any(k in bb.facts for k in m.lacks):
        return False
    return True


# Long on purpose: with max_tokens=300 the whole thing fits (end_turn), with
# max_tokens=20 it is cut mid-sentence (max_tokens). Lesson 12.
_RETURN_LONGFORM = (
    "Thanks for reaching out about your return. Here is how the process works.\n\n"
    "First, I'll need the order number so I can pull up the purchase and confirm it is "
    "inside our 30-day return window. Once I have that, I'll check the item's condition "
    "notes and whether the order has been marked delivered. If everything lines up, I can "
    "start the return and email you a prepaid shipping label the same day. Refunds are "
    "issued to the original payment method and usually land within three to five business "
    "days after the carrier scans the parcel.\n\n"
    "If the item arrived damaged, please don't send it back yet. A photo is usually enough "
    "and it saves you a trip to the post office.\n\n"
    "What is the order number?"
)

_RETURN_LONGFORM_B = (
    "Happy to help you get this returned. Here is what happens next.\n\n"
    "Send me the order number and I will look up the purchase, confirm it is still inside "
    "the 30-day window, and check that it has been marked delivered on our side. Assuming "
    "it all checks out, I will open the return straight away and email you a prepaid label. "
    "Once the carrier scans the parcel, the refund goes back to your original payment "
    "method, normally within three to five business days.\n\n"
    "One exception worth knowing: if the item turned up damaged, hold on to it for now. "
    "A quick photo is usually all we need and it saves you the trip.\n\n"
    "Could you share the order number?"
)

_RETURN_LONGFORM_C = (
    "Sorry the order didn't work out, and thanks for getting in touch about returning it.\n\n"
    "The process is short. I need the order number first so I can open the purchase, verify "
    "we are still inside the 30-day return window, and confirm the delivery status. When "
    "those line up I will create the return and send a prepaid shipping label to your email "
    "the same day. The refund is issued to whatever payment method you originally used and "
    "typically clears three to five business days after the parcel is scanned.\n\n"
    "If the reason for the return is damage in transit, don't ship it back. Send a photo "
    "instead and we will handle it from there.\n\n"
    "What is the order number on the purchase?"
)


SCENARIOS: List[Scenario] = [
    # -- Lesson 12: the very first request ---------------------------------
    Scenario(
        id="return.longform",
        lesson="12",
        # Outranks the customer-message scenarios below: this prompt is an instruction
        # TO the model ("write a response"), not a customer speaking.
        priority=40,
        note="Deliberately long. max_tokens=300 returns it whole (end_turn); "
             "max_tokens=20 cuts it mid-sentence (max_tokens).",
        when=Match(
            user_regex=r"wants? to return an order|write a short helpful response",
            has_tools=False,
        ),
        variants=(_RETURN_LONGFORM, _RETURN_LONGFORM_B, _RETURN_LONGFORM_C),
    ),

    # -- Lesson 16: stop sequences -----------------------------------------
    Scenario(
        id="return.reply_short",
        lesson="16",
        priority=42,
        note="A short support reply. When the prompt asks for an end marker, the reply "
             "emits it and the API cuts there.",
        when=Match(
            user_regex=r"short (?:customer )?support reply",
            has_tools=False,
        ),
        variants=(
            "Thanks for getting in touch about your return. I can help with that. "
            "Could you send me the order number so I can look up the purchase and check "
            "it against our 30-day return window?",
            "Happy to help with the return. Send over the order number and I will confirm "
            "the purchase is still inside our 30-day window and take it from there.",
            "I can get that return started for you. What is the order number? Once I have "
            "it I will check the return window and confirm the next step.",
        ),
    ),

    # -- Lesson 13: the API is stateless -----------------------------------
    Scenario(
        id="orderid.no_context",
        lesson="13",
        priority=30,
        note="An order number arrived with no conversation history, so there is nothing "
             "to attach it to. This is what a stateless API looks like from the outside.",
        when=Match(
            intent=("order_number",),
            max_turns=1,
            has_tools=False,
        ),
        variants=(
            "Thanks for that number. I don't have any earlier context in this conversation, "
            "so I'm not sure yet what you'd like me to do with it.\n\n"
            "Are you checking on delivery, starting a return, or something else?",
            "Got the order number, but this is the first message I can see - I don't know "
            "what it relates to.\n\n"
            "What would you like me to do with this order?",
            "I have the number, though there's no prior conversation on my side to connect "
            "it to.\n\n"
            "Could you tell me what you need help with for this order?",
        ),
    ),
    Scenario(
        id="orderid.with_context",
        lesson="13",
        priority=32,
        note="The same order number, but the history establishes that this is a return. "
             "Now the reply can actually move the case forward.",
        when=Match(
            intent=("order_number",),
            min_turns=2,
            has_tools=False,
        ),
        variants=(
            "Thank you. I've got the order number for the return you mentioned.\n\n"
            "Let me pull up that order. Was the item damaged when it arrived, or is this "
            "a change of mind?",
            "Perfect, that's what I needed for the return.\n\n"
            "I'm opening the order now. Can you tell me the reason for the return?",
            "Great, thanks. I can look up the return with that number.\n\n"
            "What was the problem with the item?",
        ),
    ),

    # -- Lesson 15: the system prompt changes behaviour --------------------
    Scenario(
        id="return.opening",
        lesson="13",
        priority=30,
        note="A neutral return request with no complaint and no demand for money. The "
             "right first move is to ask which order, so the multi-turn lesson has a "
             "question for its second turn to answer.",
        when=Match(
            intent=("refund_request",),
            user_regex=r"(?i)^\s*i want to return my order\.?\s*$",
            has_system=False,
            has_tools=False,
            lacks=("order_id",),
            max_turns=1,
        ),
        variants=(
            "I can help you with that return. Could you share your order number so I "
            "can pull up the purchase?",
            "Happy to help with the return. What is the order number?",
            "Sure, I can start a return for you. Could you give me the order number "
            "first so I can check the details?",
        ),
    ),
    Scenario(
        id="refund.unguarded",
        lesson="15",
        priority=20,
        note="No system prompt and no rules in the user text, so nothing stops the reply "
             "from promising a refund it has no authority to promise. This is the "
             "'before' half of the system-prompt lesson.",
        when=Match(
            intent=("refund_request", "damaged_item", "billing_issue"),
            has_system=False,
            forbids_flags=("no_refund_promise", "ask_for_order", "json_only"),
            has_tools=False,
        ),
        variants=(
            "I'm so sorry about that! No problem at all - I've gone ahead and approved a "
            "full refund for you. You should see the money back on your original payment "
            "method within 3-5 business days. Thanks for your patience!",
            "Absolutely, I can refund that right away. Your refund is approved and the "
            "funds will be returned to your card shortly. Sorry for the trouble!",
            "Of course - consider it refunded. I've processed the return on my end, so "
            "there's no need to send anything back.",
        ),
    ),
    Scenario(
        id="refund.guarded",
        lesson="15",
        priority=25,
        note="The system prompt forbids promising a refund before the order is checked, "
             "so the reply withholds the promise and asks for exactly one thing.",
        when=Match(
            intent=("refund_request", "damaged_item", "billing_issue"),
            requires_flags=("no_refund_promise",),
            has_tools=False,
            lacks=("order_id",),
        ),
        variants=(
            "I'm sorry the item arrived in that condition. Before I can look at a refund "
            "I need to check the order on our side.\n\n"
            "Could you share your order number?",
            "Sorry about that - I understand the frustration. I'll need to review the "
            "order first before I can discuss a refund.\n\n"
            "What is the order number?",
            "Apologies for the trouble. I'm not able to confirm a refund until I've "
            "checked the order details.\n\n"
            "Can you give me the order number, please?",
        ),
    ),

    # -- Generic support fallbacks -----------------------------------------
    Scenario(
        id="support.generic_guarded",
        lesson="15",
        priority=5,
        note="A polite, rule-following answer for a support question with no more "
             "specific fixture.",
        when=Match(requires_flags=("no_refund_promise",), has_tools=False),
        variants=(
            "Thanks for reaching out. I can help with that.\n\n"
            "Could you share your order number so I can look up the details?",
            "Happy to help. To get started, what is the order number?",
        ),
    ),
]


FALLBACK = Scenario(
    id="fallback.unscripted",
    lesson="-",
    when=Match(),
    note="No scenario matched this prompt. The simulator says so rather than "
         "inventing an answer.",
    variants=(
        "[simulated] I don't have a scripted answer for that one. The simulator matched "
        "no scenario, so this is a placeholder rather than a real reply.\n\n"
        "Call explain(message) to see what it looked for, or set a real "
        "FUELIX_API_KEY to get a genuine Claude response.",
    ),
)


def register_scenario(scenario: Scenario, replace: bool = False) -> None:
    """Add a scenario at runtime. Instructors can extend the table from a notebook cell."""
    global SCENARIOS
    if replace:
        SCENARIOS = [s for s in SCENARIOS if s.id != scenario.id]
    SCENARIOS.append(scenario)


def select_scenario(facts: PromptFacts, bb: Blackboard, has_tools: bool) -> Scenario:
    hits = [s for s in SCENARIOS if _matches(s.when, facts, bb, has_tools)]
    if not hits:
        return FALLBACK
    return max(hits, key=lambda s: (s.priority, s.when.specificity()))


# ---------------------------------------------------------------------------
# 7b. Computed replies
#
# The scenario table returns fixed text, which is right for a support answer.
# Some lessons need a reply that DEPENDS on the customer's own words -
# classification, extraction - so those get a small function instead of a canned
# string. Checked after tools and before the scenario table.
# ---------------------------------------------------------------------------


#: Reply ids produced here, with the note explain() should print for each.
COMPUTED_NOTES: Dict[str, str] = {
    "classify.intent_json": (
        "The prompt asked for an intent label as JSON, so the reply is computed from "
        "the customer's own words rather than picked from a fixture."
    ),
    "extract.freeform_json": (
        "The prompt asked for JSON without giving a schema, so the reply is valid "
        "JSON with field names the model chose for itself - which is the point."
    ),
    "policy.yes_no": "A YES/NO verdict computed from the 30-day rule and the "
                     "categories the system prompt says cannot be returned.",
    "policy.out_of_scope": "The system prompt limits the scope and the request is "
                           "outside it.",
    "policy.ask_details": "The request does not say which item or when it was "
                          "delivered - often because the history was not sent.",
    "policy.return_ok": "Item and days found in the conversation, inside the window.",
    "policy.noted": "Item and days found in the conversation, inside the window.",
    "policy.too_late": "Item and days found in the conversation, past the window.",
    "policy.refund": "A refund asked for inside the window.",
    "policy.receipt": "A follow-up about a lost receipt.",
    "orders.shipped": "A shipping question about a known order (STORE_ORDERS).",
    "orders.one": "A question about one known order (STORE_ORDERS).",
    "orders.several": "Several orders in one question: one paragraph each, so a small "
                      "max_tokens cuts it mid-way.",
    "summary.log": "A two-line note for the log, written from the conversation.",
    "format.json_schema": "output_config.format asked for JSON matching a schema; the "
                          "values are read from the document after the instruction.",
    "agent.final": "The agent loop's last turn: the answer is written from the tool "
                   "results in the conversation.",
    "grade.support_reply": (
        "The prompt asked this call to grade another assistant's reply, so the score "
        "is computed by checking that reply against the four stated rules."
    ),
}

CLASSIFY_LABELS: Tuple[str, ...] = ("refund_request", "order_status", "billing_issue",
                                    "product_question", "other")

_CUSTOMER_BLOCK = re.compile(
    r"customer(?:'s)?\s+message\s*:\s*(.+?)(?:\n\s*\n|\Z)", re.I | re.S)


def embedded_customer_message(facts: PromptFacts) -> str:
    """Pull the customer's own words out of a prompt that wraps them in instructions.

    Without this, intent detection runs over the whole prompt - including the list
    of allowed labels - and every message looks like a billing issue.
    """
    tagged = facts.xml_tags.get("customer_message") or facts.xml_tags.get("user_message")
    if tagged:
        return tagged.strip()
    match = _CUSTOMER_BLOCK.search(facts.effective_user_text)
    return match.group(1).strip() if match else ""


def _wants_intent_json(text: str) -> bool:
    return bool(re.search(r"classif", text, re.I)
                and re.search(r"\bintent\b", text, re.I)
                and re.search(r"\bjson\b", text, re.I))


def classify_intent_label(message: str) -> str:
    """Map a customer message onto one of the five course intent labels."""
    for name in detect_intents(message):
        if name in ("billing_issue", "order_status", "product_question"):
            return name
        if name in ("refund_request", "damaged_item"):
            return "refund_request"
    return "other"


def _reply_classify(facts: PromptFacts) -> Optional[Tuple[str, str]]:
    if not _wants_intent_json(facts.effective_user_text):
        return None
    message = embedded_customer_message(facts)
    if not message:
        return None
    return "classify.intent_json", json.dumps(
        {"intent": classify_intent_label(message)})


_LABELLED_BLOCK = r"{}\s*:\s*(.+?)(?:\n\s*\n|\Z)"


def _labelled(text: str, label: str) -> str:
    match = re.search(_LABELLED_BLOCK.format(label), text, re.I | re.S)
    return match.group(1).strip() if match else ""


def _is_polite(text: str) -> bool:
    """Deliberately generous. A grader that invents a tone complaint is worse than
    one that misses a curt sentence, because the learner then argues with the
    grader instead of reading the real violation."""
    return bool(re.search(
        r"\bsorry\b|\bthank|\bplease\b|happy to|glad to|apolog|i can help|let me"
        r"|no problem|of course|\bsure\b|absolutely|certainly|my pleasure"
        r"|i understand|\bhope\b|\bcould you\b|\bwould you\b", text, re.I))


def _reply_grade(facts: PromptFacts) -> Optional[Tuple[str, str]]:
    """Act as a model-based grader over another assistant's reply.

    Scores against the four rules the course's grading prompt states, using the
    same behavioural helpers the lab checks use - so the grade a learner sees and
    the grade the checker expects can never drift apart.
    """
    criteria = facts.system_text + "\n" + facts.effective_user_text
    if not re.search(r"evaluat\w*\s+(?:a\s+)?customer support assistant", criteria, re.I):
        return None
    if not (re.search(r"\bscore\b", criteria, re.I) and re.search(r"\bjson\b", criteria, re.I)):
        return None

    reply = _labelled(facts.effective_user_text, "assistant response")
    if not reply:
        return None
    customer = _labelled(facts.effective_user_text, "customer message")

    risky = bool(re.search(r"\bfraud\b|\blawyer\b|\blegal\b|\bsue\b|\bchargeback\b",
                           customer, re.I))
    escalates = bool(re.search(r"escalat|human|specialist|supervisor|our team will",
                               reply, re.I))

    score, faults = 10, []
    if _promises_refund(reply):
        score -= 6
        faults.append("it promised a refund before the order was checked")
    if not _asks_for_order_number(reply) and not re.search(r"ORD-", reply, re.I):
        score -= 2
        faults.append("it never asked for the order number")
    if not _is_polite(reply):
        score -= 1
        faults.append("the tone is curt")
    if risky and not escalates:
        score -= 4
        faults.append("the customer mentioned fraud or legal action and it did not escalate")

    score = max(1, min(10, score))
    if not faults:
        reason = "The response follows all four rules."
    else:
        joined = faults[0] if len(faults) == 1 else (
            ", ".join(faults[:-1]) + " and " + faults[-1])
        reason = "Rule violations: " + joined + "."
    return "grade.support_reply", json.dumps(
        {"score": score, "passed": score >= 7, "reason": reason}, indent=2)


def _reply_freeform_json(facts: PromptFacts) -> Optional[Tuple[str, str]]:
    """The "just ask for JSON" path, before the lesson introduces a schema.

    It returns clean, parseable JSON on purpose. Staging a syntax error would be
    dishonest - a good model usually does return valid JSON here. What it cannot
    do is agree with a schema nobody sent it, so the field NAMES and the enum
    values are its own invention. That is the real failure mode, and it is the one
    that survives contact with a live API key.
    """
    text = facts.effective_user_text
    if not re.search(r"\bextract\b", text, re.I):
        return None
    if not re.search(r"only valid json|valid json object|return only json", text, re.I):
        return None
    if re.search(r"\bintent\b", text, re.I):
        return None
    message = embedded_customer_message(facts)
    if not message:
        return None

    if re.search(ENUM_HINTS["damaged_item"], message, re.I):
        reason = "damaged"
    elif re.search(ENUM_HINTS["billing_dispute"], message, re.I):
        reason = "billing problem"
    else:
        reason = "return request"

    if re.search(ENUM_HINTS["replacement"], message, re.I):
        action = "send a replacement"
    elif re.search(ENUM_HINTS["refund"], message, re.I):
        action = "refund"
    else:
        action = "unknown"

    payload = {
        "order_id": extract_slots(message).get("order_id"),
        "item": _extract_item(message),
        "reason": reason,
        "requested_action": action,
        "photo_available": bool(re.search(r"photo|picture|image", message, re.I)),
    }
    return "extract.freeform_json", json.dumps(payload, indent=2)


# -- The re-recorded course: answers that follow the store's return policy ----
#
# The practice lectures run one policy - returns within 30 days of delivery - and
# ask the same few kinds of question about it. These replies are computed from the
# days and the item the customer actually mentioned, anywhere in the conversation,
# so that keeping or dropping the history visibly changes the answer.

RETURN_WINDOW_DAYS = 30

_DAYS = re.compile(r"(\d+)\s+days?\s+ago|after\s+(\d+)\s+days|(\d+)\s+days?\s+(?:back|since)",
                   re.I)
_RELATIVE_DAYS = ((r"\byesterday\b", 1), (r"\btoday\b", 0), (r"\blast week\b", 7),
                  (r"\ba week ago\b", 7), (r"\btwo weeks ago\b", 14),
                  (r"\blast month\b", 31), (r"\ba month ago\b", 31))


_UNITS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
          "fourteen fifteen sixteen seventeen eighteen nineteen").split()
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}


def _spelled_to_digits(text: str) -> str:
    """'twenty days' -> '20 days', 'forty-five' -> '45': customers write numbers out."""
    def number(match: "re.Match[str]") -> str:
        tens, unit, alone = match.group(1), match.group(2), match.group(3)
        if tens:
            return str(_TENS[tens.lower()] + (_UNITS.index(unit.lower()) if unit else 0))
        return str(_UNITS.index(alone.lower()))
    pattern = r"\b(?:({})(?:[- ]({}))?|({}))\b".format(
        "|".join(_TENS), "|".join(_UNITS[1:10]), "|".join(_UNITS[1:]))
    return re.sub(pattern, number, text, flags=re.I)


def _days_in(text: str) -> Optional[int]:
    text = _spelled_to_digits(text)
    match = _DAYS.search(text)
    if match:
        return int(next(g for g in match.groups() if g))
    for pattern, days in _RELATIVE_DAYS:
        if re.search(pattern, text, re.I):
            return days
    return None


def _user_texts(facts: PromptFacts) -> List[str]:
    return [text for role, text in facts.history if role == "user"]


def _policy_known(facts: PromptFacts) -> bool:
    blob = facts.system_text + "\n" + "\n".join(_user_texts(facts))
    return bool(re.search(r"(?:within|inside)\s+30\s+days|30[- ]day", blob, re.I))


#: What the store does not take back, as the course's v2 evaluation prompt names it.
_NON_RETURNABLE = (("gift card", r"gift\s*cards?"), ("underwear", r"underwear"))


def _reply_yes_no(facts: PromptFacts) -> Optional[Tuple[str, str]]:
    """The evaluation lecture's verdict: YES or NO first, then one sentence.

    Underwear counts as non-returnable only if the system prompt names it. That is
    what makes the first prompt fail the underwear case and the second pass it -
    the gap the test set is built to find, as in the video.
    """
    if not re.search(r"\bYES\b\s+or\s+\bNO\b", facts.system_text):
        return None
    text = facts.effective_user_text
    # Gift cards are non-returnable almost everywhere, and a real model knows it
    # without being told. Underwear is a store policy: unknown unless the prompt says.
    if re.search(r"gift\s*cards?", text, re.I):
        return "policy.yes_no", "NO. Gift cards cannot be returned."
    for label, pattern in _NON_RETURNABLE:
        if re.search(pattern, text, re.I) and re.search(pattern, facts.system_text, re.I):
            return "policy.yes_no", ("NO. Underwear cannot be returned." if label == "underwear"
                                     else "NO. {}s cannot be returned.".format(
                                         label.capitalize()))
    days = _days_in(text)
    if days is not None and days > RETURN_WINDOW_DAYS:
        return "policy.yes_no", ("NO. It was delivered {} days ago, which is past our "
                                 "{}-day return window.".format(days, RETURN_WINDOW_DAYS))
    return "policy.yes_no", ("YES. It is within our {}-day return window, so you can "
                             "send it back.".format(RETURN_WINDOW_DAYS))


_ON_TOPIC = (r"\border|\breturn|\brefund|\bdeliver|\bship|\bpackage|\bparcel|\bcharge"
             r"|\breceipt|\bexchange|\bitem\b|\bbought\b|\bpurchase")


def _reply_scope(facts: PromptFacts) -> Optional[Tuple[str, str]]:
    """A system prompt that limits the scope declines what falls outside it."""
    if not re.search(r"help only with|only help with|only with orders", facts.system_text,
                     re.I):
        return None
    if re.search(_ON_TOPIC, facts.effective_user_text, re.I):
        return None
    return "policy.out_of_scope", ("I'm sorry, I can only help with orders, returns "
                                   "and refunds. Is there anything about an order I "
                                   "can help you with?")


def _reply_returns_policy(facts: PromptFacts) -> Optional[Tuple[str, str]]:
    """Answers about returning an item under the 30-day rule.

    The item and the days come from every user turn the request carries. Send only
    the last question and they are simply not there - so the reply asks for them,
    which is the statelessness lesson made visible.
    """
    if not _policy_known(facts):
        return None
    users = _user_texts(facts)
    if not users:
        return None
    last = users[-1]
    if not re.search(r"\breturn|\brefund|\breceipt|\bbought\b|\bmoney back", last, re.I):
        return None
    item = next((i for i in (_extract_item(t) for t in reversed(users)) if i), None)
    days = next((d for d in (_days_in(t) for t in reversed(users)) if d is not None), None)
    left = None if days is None else RETURN_WINDOW_DAYS - days
    two_sentences = bool(re.search(r"two sentences", "\n".join(users), re.I))

    if re.search(r"\breceipt\b", last, re.I):
        if item and days is not None and left >= 0:
            return "policy.receipt", ("No problem - we can find the purchase from your "
                                      "order, so you can still return the {} within the "
                                      "{} days you have left.".format(item, left))
        return "policy.receipt", ("That's fine - we can usually find the purchase from "
                                  "your order number. Which item is it, and when was it "
                                  "delivered?")
    if item is None and days is None:
        return "policy.ask_details", ("Which item would you like to return, and when was "
                                      "it delivered? Then I can check it against our "
                                      "{}-day return window.".format(RETURN_WINDOW_DAYS))
    thing = "the {}".format(item) if item else "it"
    if days is None:
        return "policy.ask_details", ("When was {} delivered? Returns are accepted within "
                                      "{} days of delivery.".format(thing,
                                                                    RETURN_WINDOW_DAYS))
    if left < 0:
        return "policy.too_late", ("I'm sorry, {} was bought {} days ago, which is past "
                                   "our {}-day return window, so it can no longer be "
                                   "returned.".format(thing, days, RETURN_WINDOW_DAYS))
    if re.search(r"\brefund|\bmoney back", last, re.I):
        return "policy.refund", ("I can help with that refund: {} was bought {} days ago, "
                                 "inside our {}-day window. Please share your order number "
                                 "so I can start it.".format(thing, days, RETURN_WINDOW_DAYS))
    if not re.search(r"\?", last):
        return "policy.noted", ("Thanks for letting me know. Since you bought {} {} days "
                                "ago, you still have about {} days left in our {}-day "
                                "return window.".format(thing, days, left,
                                                        RETURN_WINDOW_DAYS))
    reply = ("Yes, you can still return {}: it was bought {} days ago, so about {} days "
             "are left in our {}-day return window.".format(thing, days, left,
                                                           RETURN_WINDOW_DAYS))
    if two_sentences:
        reply += (" Please send it back unused and in its original packaging, and keep "
                  "the order number handy.")
    return "policy.return_ok", reply


#: The orders the course's practice scripts know (CCAR-F-practice README).
STORE_ORDERS: Dict[str, Dict[str, Any]] = {
    "ORD-12345678": {"item": "jacket", "status": "delivered", "delivered_days_ago": 20,
                     "price": 80},
    "ORD-87654321": {"item": "lamp", "status": "delivered", "delivered_days_ago": 45,
                     "price": 60},
    "ORD-11112222": {"item": "headphones", "status": "shipped", "delivered_days_ago": None,
                     "price": 120},
}


def _order_line(order_id: str) -> str:
    order = STORE_ORDERS.get(order_id)
    if order is None:
        return "I could not find an order {} - please check the number.".format(order_id)
    if order["status"] != "delivered":
        return ("Order {} (the {}) has shipped and is on its way; once it arrives you have "
                "{} days to return it.".format(order_id, order["item"], RETURN_WINDOW_DAYS))
    days = order["delivered_days_ago"]
    if days <= RETURN_WINDOW_DAYS:
        return ("Order {} (the {}) was delivered {} days ago, so it is still inside our "
                "{}-day window: you can return it, and you have {} days left.".format(
                    order_id, order["item"], days, RETURN_WINDOW_DAYS,
                    RETURN_WINDOW_DAYS - days))
    return ("Order {} (the {}) was delivered {} days ago, which is past our {}-day window, "
            "so it can no longer be returned.".format(order_id, order["item"], days,
                                                      RETURN_WINDOW_DAYS))


_RETURN_STEPS = (
    "To return it, open Orders in your account, choose the order and select Return item. "
    "Pick a reason, print the prepaid label we email you, and pack the item in its "
    "original packaging with all tags and accessories. Drop the parcel at any carrier "
    "point within 14 days of starting the return, and keep the receipt the carrier gives "
    "you. Once the parcel reaches our warehouse we inspect it, usually within three "
    "business days, and the refund goes back to the original payment method within five "
    "to seven business days after that. If you would rather have a different size or "
    "colour, choose Exchange instead of Return in the same screen: we reserve the new "
    "item as soon as you start, ship it the day your parcel is scanned, and charge or "
    "refund any difference in price automatically, so there is nothing else to do. ")


def _reply_orders(facts: PromptFacts) -> Optional[Tuple[str, str]]:
    """Questions about the store's known orders, answered from STORE_ORDERS.

    Several orders in one question get one detailed paragraph each, so the reply is
    long enough that a 300-token ceiling cuts it in the middle of the second order -
    the lecture's "loud" problem.
    """
    if not _policy_known(facts):
        return None
    ids = list(dict.fromkeys(re.findall(r"\bORD-\d{8}\b", facts.effective_user_text)))
    if not ids:
        return None
    text = facts.effective_user_text
    if len(ids) == 1 and re.search(r"\bship", text, re.I):
        order = STORE_ORDERS.get(ids[0])
        if order and order["status"] == "delivered":
            return "orders.shipped", ("Yes - order {} (the {}) shipped and was delivered "
                                      "{} days ago.".format(ids[0], order["item"],
                                                            order["delivered_days_ago"]))
        return "orders.shipped", _order_line(ids[0])
    if len(ids) == 1:
        return "orders.one", _order_line(ids[0])
    parts = []
    for order_id in ids:
        line = _order_line(order_id)
        order = STORE_ORDERS.get(order_id)
        if order and order["status"] == "delivered" and \
                order["delivered_days_ago"] <= RETURN_WINDOW_DAYS:
            line += " " + _RETURN_STEPS
        elif order and order["status"] == "delivered":
            line += (" If the {} is faulty, it may still be covered by the manufacturer's "
                     "warranty: reply with a photo of the fault and a short description, "
                     "and our team will check it with the manufacturer and come back to "
                     "you within two business days with the options available. ".format(
                         order["item"]))
        parts.append(line)
    if re.search(r"partial(?:ly)? refund", text, re.I):
        parts.append("For the order that was partly refunded, only the part that was not "
                     "refunded can still be returned, and the refund will be reduced by "
                     "the amount already paid back.")
    parts.append("If you would like, I can start the eligible return for you right now: "
                 "just confirm which items you are sending back and whether you prefer "
                 "a refund to your original payment method or store credit, which is "
                 "issued as soon as the parcel is scanned. Store credit never expires and "
                 "can be used on anything in the shop, including sale items. And if "
                 "anything about these orders is not what you expected, tell me what "
                 "happened and I will look into it.")
    return "orders.several", ("Thanks for the details - here is where each order stands. "
                              + "\n\n".join(p.strip() for p in parts))


def _reply_log_summary(facts: PromptFacts) -> Optional[Tuple[str, str]]:
    """A two-line note for the log, written from the conversation it is given."""
    text = facts.effective_user_text
    if not (re.search(r"summar", text, re.I) and re.search(r"\blog\b|two lines|two-line",
                                                         text, re.I)):
        return None
    convo = _document_of(text) if "\n\n" in text else text
    item = _extract_item(convo) or "an order"
    days = _days_in(convo)
    first = ("Customer asked to return the {} bought {} days ago.".format(item, days)
             if days is not None else "Customer asked about {}.".format(item))
    second = ("ShopAssist confirmed it is inside the 30-day window."
              if days is not None and days <= RETURN_WINDOW_DAYS
              else "ShopAssist answered and closed the conversation.")
    return "summary.log", first + "\n" + second


# -- Structured outputs: output_config={"format": {"type": "json_schema", ...}} ----

_INSTRUCTION_THEN_DOCUMENT = re.compile(
    r"\s*(?:extract|classify|summari[sz]e|read|pull)\b.*?\n\s*\n(.*)\Z", re.I | re.S)


def _document_of(text: str) -> str:
    """The document an instruction wraps: everything after its first blank line.

    The course's extraction prompts say "Order IDs look like ORD-12345678" before
    the email. Searching the whole prompt would find that example in every email.
    """
    match = _INSTRUCTION_THEN_DOCUMENT.match(text)
    return match.group(1).strip() if match else text


def _nullable(spec: dict) -> bool:
    types_ = spec.get("type")
    return isinstance(types_, list) and "null" in types_


def fill_json_schema(schema: dict, messages: Sequence[Any]) -> Dict[str, Any]:
    """What a schema-constrained reply contains, read from the source document.

    A required string the document does not contain comes back as "unknown" - the
    schema demanded a string and got one, and the value is useless. A nullable field
    comes back null instead. On a retry, the field the feedback names is set to null
    when the feedback allows it.
    """
    users = [_flatten(m.get("content") if isinstance(m, dict) else getattr(m, "content", None))
             for m in messages
             if (m.get("role") if isinstance(m, dict) else getattr(m, "role", None)) == "user"]
    source = _document_of(users[0] if users else "")
    one = [{"role": "user", "content": source}]
    out = synthesize_input({"input_schema": schema}, analyze_prompt("", one),
                           build_blackboard(one))
    props = schema.get("properties") or {}
    for key, spec in props.items():
        if out.get(key) == "" and not _nullable(spec) and "string" in str(spec.get("type")):
            out[key] = "unknown"
    assistant_turns = sum(
        1 for m in messages
        if (m.get("role") if isinstance(m, dict) else getattr(m, "role", None)) == "assistant")
    if assistant_turns and len(users) > 1:
        feedback = users[-1]
        for key, spec in props.items():
            if (re.search(r"\b{}\b".format(re.escape(key)), feedback)
                    and re.search(r"use null|not a valid|invalid|not allowed", feedback, re.I)
                    and _nullable(spec)):
                out[key] = None
    if schema.get("additionalProperties") is False:
        out = {k: v for k, v in out.items() if k in props}
    return out


def _check_format(fmt: Any) -> dict:
    if not isinstance(fmt, dict) or fmt.get("type") != "json_schema":
        raise BadRequestError(
            'output_config.format must be {"type": "json_schema", "schema": {...}}')
    schema = fmt.get("schema")
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise BadRequestError("output_config.format.schema must be a JSON schema object")
    return schema


#: Order matters only in that the first responder to return wins.
COMPUTED_REPLIES: Tuple[Callable[[PromptFacts], Optional[Tuple[str, str]]], ...] = (
    _reply_classify,
    _reply_grade,
    _reply_freeform_json,
    _reply_yes_no,
    _reply_scope,
    _reply_log_summary,
    _reply_orders,
    _reply_returns_policy,
)


def compute_reply(facts: PromptFacts) -> Optional[Tuple[str, str]]:
    for responder in COMPUTED_REPLIES:
        result = responder(facts)
        if result is not None:
            return result
    return None


# ---------------------------------------------------------------------------
# 8. Tool support
# ---------------------------------------------------------------------------


ENUM_HINTS: Dict[str, str] = {
    "damaged_item": r"damaged|arrived broken|\bbroken\b|scratched|defective|cracked",
    "damaged": r"damaged|arrived broken|\bbroken\b|scratched|defective|cracked"
               r"|stopped working|do(?:es)? ?n(?:o|')?t work",
    "too_small": r"too small|too tight",
    "too_big": r"too big|too large|too loose|bigger than|larger than",
    "wrong_item": r"wrong item|not what i ordered",
    "billing_dispute": r"charged twice|duplicate charge|billing",
    "changed_mind": r"changed my mind",
    "refund": r"\brefund\b|money back|want to return|send it back|return it",
    "replacement": r"replacement|replace it",
    "exchange": r"\bexchange\b",
    "store_credit": r"store credit|gift card",
    "high": r"\btoday\b|urgent|asap|immediately|right now",
    "low": r"no rush|whenever|not urgent",
    "policy_exception": r"\d+\s*months? ago|a year ago|years? ago"
                        r"|outside (?:the )?(?:normal )?policy|past the (?:return )?window"
                        r"|never used it|bought (?:this|it) (?:six|6|five|5|four|4)",
    "normal_return": r"not what i expected|changed my mind|wrong size"
                     r"|does ?n(?:o|')?t fit|no longer need|do ?n(?:o|')?t need it",
    "refund_request": r"\brefund\b|money back|want to return",
    "order_status": r"where is my order|supposed to arrive|tracking",
    "billing_issue": r"charged twice|duplicate charge|billing",
    "product_question": r"how do i|compatible",
}


def _coerce(value: Any, spec: dict) -> Any:
    types_ = spec.get("type")
    if isinstance(types_, list):
        types_ = next((t for t in types_ if t != "null"), "string")
    if value is None:
        return None
    try:
        if types_ == "integer":
            return int(float(value))
        if types_ == "number":
            number = float(value)
            # Claude writes 60, not 60.0, for a whole amount in JSON.
            return int(number) if number.is_integer() else number
        if types_ == "boolean":
            return bool(value) if not isinstance(value, str) else value.lower() == "true"
        if types_ == "array" and not isinstance(value, list):
            return [value]
        if types_ == "string" and not isinstance(value, str):
            return str(value)
    except (TypeError, ValueError):
        return value
    return value


def _from_schema(key: str, spec: dict, facts: PromptFacts) -> Any:
    text = facts.effective_user_text
    enum = spec.get("enum")
    if enum:
        best, best_len = None, 0
        for option in enum:
            pat = ENUM_HINTS.get(str(option))
            if not pat:
                continue
            m = re.search(pat, text, re.I)
            if m and len(m.group(0)) > best_len:
                best, best_len = option, len(m.group(0))
        if best is not None:
            return best
        # Nothing in the message pointed anywhere. A neutral option is a more
        # honest default than "unclear", which should mean the message was
        # ambiguous - not that this particular field was never mentioned.
        for neutral in ("normal", "medium", "none", "unclear", "other"):
            if neutral in enum:
                return neutral
        return enum[0]

    types_ = spec.get("type")
    nullable = isinstance(types_, list) and "null" in types_
    if isinstance(types_, list):
        types_ = next((t for t in types_ if t != "null"), "string")

    if types_ == "boolean":
        return False
    if types_ in ("number", "integer"):
        m = re.search(r"(\d+(?:\.\d+)?)", text)
        return _coerce(m.group(1), spec) if m else 0
    if types_ == "array":
        return []
    if nullable:
        return None
    return text[:200] if key in ("customer_message", "message", "description") else ""


#: Products the course's fixtures talk about. A closed list on purpose - guessing a
#: noun out of free text is exactly the kind of thing a simulator should not pretend
#: to do well. Add to it rather than making it clever.
_ITEM_WORDS = (r"\b(shoes|sneakers|boots|jacket|headphones|earbuds|speaker|laptop"
               r"|keyboard|mouse|monitor|tablet|phone|watch|charger|camera|backpack"
               r"|lamp|scarf|dress|shirt)\b")

#: The customer already sent evidence, as opposed to offering to send it later.
_EVIDENCE_SENT = (r"attached|i (?:have |'ve )?sent (?:a |the )?photo|photo is attached"
                  r"|here (?:is|are) (?:a |the )?(?:photo|picture|image)"
                  r"|i(?:'ve| have) uploaded")


def _extract_item(text: str) -> Optional[str]:
    match = re.search(_ITEM_WORDS, text, re.I)
    if match:
        return match.group(1).lower()
    # "I got my shoes yesterday" - the thing right after "my", when it is not one
    # of the words that means something else in this domain.
    match = re.search(r"\bmy\s+([a-z]{3,20})\b", text, re.I)
    if match and match.group(1).lower() not in (
            "order", "money", "refund", "package", "parcel", "account", "card",
            "payment", "item", "purchase", "delivery"):
        return match.group(1).lower()
    return None


def _extract_evidence(text: str) -> bool:
    return bool(re.search(_EVIDENCE_SENT, text, re.I))


#: A message that says two incompatible things. Deliberately narrow - a simulator
#: guessing at contradictions would be worse than one that only spots the shapes
#: the course actually teaches.
_CONTRADICTION = (
    r"works (?:perfectly|fine).{0,80}(?:broken|damaged|does ?n(?:o|')?t work)"
    r"|(?:arrived broken|is damaged).{0,80}works (?:perfectly|fine)"
    r"|\brefund\b.{0,80}(?:send me another|another one|replacement)"
    r"|\breplacement\b.{0,80}\brefund\b"
)


def _extract_conflict(text: str) -> bool:
    return bool(re.search(_CONTRADICTION, text, re.I | re.S))


def _extract_conflict_reason(text: str) -> Optional[str]:
    if not _extract_conflict(text):
        return None
    return "The message asks for two different outcomes at once."


#: Field-name-driven extraction, consulted before the generic schema fallback.
#: Keyed by the names this course's schemas actually use.
FIELD_EXTRACTORS: Dict[str, Callable[[str], Any]] = {
    "item": _extract_item,
    "product": _extract_item,
    "evidence_provided": _extract_evidence,
    "conflict_detected": _extract_conflict,
    "conflict_reason": _extract_conflict_reason,
}


def _fill_missing_information(out: Dict[str, Any], props: Dict[str, Any]) -> None:
    """List the fields that came back null, which is what the field is for.

    Doing it as a post-pass rather than a guess keeps it honest: the list can only
    ever name fields this same extraction really did leave empty.
    """
    spec = props.get("missing_information") or {}
    if not spec or out.get("missing_information"):
        return
    gaps = [name for name, value in out.items()
            if value is None and name != "missing_information"
            and not name.endswith("_detail") and not name.endswith("_reason")]
    out["missing_information"] = gaps


def _field_confidence(value: Any, spec: dict, text: str, scale: dict) -> Any:
    """How sure a field-level confidence says the model is about one value.

    High when the value is written plainly in the document (an order ID, a word an
    enum hint names), middling when an enum fell back to a neutral option because the
    customer was vague, low when nothing was found. A score, not a measurement - which
    is exactly why the lecture calibrates it on labeled data.
    """
    if value in (None, "", "unknown"):
        level = 0.2
    elif spec.get("enum"):
        hint = ENUM_HINTS.get(str(value))
        level = 0.9 if hint and re.search(hint, text, re.I) else 0.5
    else:
        level = 0.9
    top = scale.get("maximum", 1)
    if "integer" in str(scale.get("type")):
        return max(int(scale.get("minimum", 0)), int(round(level * top)))
    return round(level * top, 2)


def _fill_derived(out: Dict[str, Any], props: Dict[str, Any], text: str = "") -> None:
    """Fill confidence and human_review_required from the rest of the extraction.

    Both are computed rather than read out of the text. `confidence` especially:
    the generic number fallback hunts for the first digits in the prompt, which on
    "return order 12345" would confidently report a confidence of 12345.
    """
    if "confidence" in props:
        score = 0.92
        if out.get("conflict_detected"):
            score = 0.45
        elif out.get("reason") in ("unclear", "other"):
            score = 0.55
        elif out.get("desired_action") in ("unclear", "other"):
            score = 0.6
        elif out.get("reason") == "policy_exception":
            score = 0.78
        if out.get("order_id") is None:
            score -= 0.05
        out["confidence"] = round(score, 2)

    # Field-level confidence: "<field>_confidence" next to each field it scores.
    for key, scale in props.items():
        field_name = key[:-len("_confidence")] if key.endswith("_confidence") else None
        if field_name and field_name in props:
            out[key] = _field_confidence(out.get(field_name), props[field_name], text,
                                         scale)

    if "human_review_required" in props:
        confidence = out.get("confidence")
        out["human_review_required"] = bool(
            out.get("conflict_detected")
            or out.get("reason") in ("policy_exception", "unclear")
            or (isinstance(confidence, (int, float))
                and not isinstance(confidence, bool) and confidence < 0.7))


def synthesize_input(tool: dict, facts: PromptFacts, bb: Blackboard) -> dict:
    """Build a tool input that satisfies the schema's `required` list exactly.

    Exactness matters: the course's agentic loop dispatches with `fn(**tool_input)`,
    so a stray key raises TypeError and a missing key raises TypeError too.
    """
    schema = tool.get("input_schema") or {}
    props = schema.get("properties") or {}
    keys = list(schema.get("required") or props.keys())
    if LabConfig.include_optional_tool_inputs:
        keys += [k for k in props if k not in keys]

    out: Dict[str, Any] = {}
    for key in keys:
        spec = props.get(key, {})
        value = bb.facts.get(key)
        if value is None:
            slots = extract_slots(facts.effective_user_text)
            value = slots.get(key)
        if value is None and key in FIELD_EXTRACTORS:
            value = FIELD_EXTRACTORS[key](facts.effective_user_text)
        if value is None:
            value = _from_schema(key, spec, facts)
        out[key] = _coerce(value, spec)
    _fill_missing_information(out, props)
    _fill_derived(out, props, facts.effective_user_text)
    return out


def _find_tool(tools: Sequence[dict], name: str) -> Optional[dict]:
    for tool in tools:
        if tool.get("name") == name:
            return tool
    return None


def decide_tool(tools: List[dict], tool_choice: Any, facts: PromptFacts,
                bb: Blackboard) -> Optional[Tuple[dict, dict, bool]]:
    """Return (tool, synthesized_input, forced) or None to answer with text instead."""
    if not tools:
        return None

    kind = tool_choice.get("type") if isinstance(tool_choice, dict) else (tool_choice or "auto")
    if kind == "none":
        return None

    if kind == "tool":
        name = tool_choice.get("name")
        tool = _find_tool(tools, name)
        if tool is None:
            raise BadRequestError(
                "tool_choice.name {!r} is not present in tools".format(name))
        return tool, synthesize_input(tool, facts, bb), True

    if len(bb.tools_called) >= LabConfig.max_tool_turns:
        return None

    # auto / any: call the first tool that has not been used yet.
    for tool in tools:
        if tool.get("name") not in bb.tools_called:
            return tool, synthesize_input(tool, facts, bb), False

    if kind == "any":
        tool = tools[0]
        return tool, synthesize_input(tool, facts, bb), False
    return None


# ---------------------------------------------------------------------------
# 8b. Agent planner - the tool loop the re-recorded course runs
#
# Reads every tool call and its result in the conversation, decides the next tool
# from what the customer asked and what the tools already returned, and writes the
# final answer from those results. Keyed by the tool names the course's scripts use
# (get_customer, lookup_order, start_return, escalate, list_charges, refund_charge,
# and agents exposed as *_agent tools); anything else falls back to the generic
# behaviour above.
# ---------------------------------------------------------------------------


@dataclass
class ToolEvent:
    name: str
    input: Dict[str, Any]
    result: Any = None
    is_error: bool = False


def _parse_result(content: Any) -> Any:
    if isinstance(content, list):
        content = "".join(str(_block_get(b, "text", "")) for b in content
                          if _block_type(b) == "text")
    if isinstance(content, str):
        try:
            return json.loads(content)
        except ValueError:
            return content
    return content


def tool_history(messages: Sequence[Any]) -> List[ToolEvent]:
    """Every tool call in the conversation, paired with the result sent back."""
    events: List[ToolEvent] = []
    by_id: Dict[str, ToolEvent] = {}
    for msg in messages:
        body = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
        if isinstance(body, str):
            continue
        for block in (body or []):
            kind = _block_type(block)
            if kind == "tool_use":
                event = ToolEvent(str(_block_get(block, "name", "")),
                                  dict(_block_get(block, "input", {}) or {}))
                events.append(event)
                by_id[str(_block_get(block, "id", ""))] = event
            elif kind == "tool_result":
                event = by_id.get(str(_block_get(block, "tool_use_id", "")))
                if event is not None:
                    event.result = _parse_result(_block_get(block, "content"))
                    flagged = bool(_block_get(block, "is_error", False))
                    event.is_error = flagged or (isinstance(event.result, dict)
                                                 and bool(event.result.get("isError")))
    return events


def _not_found(result: Any) -> bool:
    if isinstance(result, dict):
        if result.get("found") is False:
            return True
        return bool(re.search(r"not found", str(result.get("error", "")), re.I))
    return False


def _order_of(result: Any) -> Optional[Dict[str, Any]]:
    """The order record inside a lookup_order result, whichever shape it has."""
    if not isinstance(result, dict) or _not_found(result) or result.get("isError"):
        return None
    if isinstance(result.get("order"), dict):
        return result["order"]
    if "item" in result:
        return result
    return None


def _called(events: List[ToolEvent], name: str) -> List[ToolEvent]:
    return [e for e in events if e.name == name]


def _tool_input(tool: dict, values: Dict[str, Any], facts: PromptFacts) -> Dict[str, Any]:
    """Exactly the schema's properties, filled from `values` where we have them."""
    schema = tool.get("input_schema") or {}
    props = schema.get("properties") or {}
    keys = list(schema.get("required") or props.keys())
    out: Dict[str, Any] = {}
    for key in keys:
        if key in values:
            out[key] = _coerce(values[key], props.get(key, {}))
        else:
            out[key] = _coerce(_from_schema(key, props.get(key, {}), facts), props.get(key, {}))
    return out


def _wants_return(text: str) -> bool:
    return bool(re.search(r"\breturn|\bsend (?:it|them) back|\brefund", text, re.I))


def plan_next(tool_list: List[dict], facts: PromptFacts,
              events: List[ToolEvent]) -> Optional[Tuple[dict, Dict[str, Any]]]:
    """The next tool to call, or None when it is time to answer in text."""
    if len(events) >= LabConfig.max_tool_turns:
        return None
    tools = {t.get("name"): t for t in tool_list}
    text = facts.effective_user_text
    slots = extract_slots(text)
    order_id = slots.get("order_id")
    last = events[-1] if events else None

    if last is not None and last.is_error and isinstance(last.result, dict):
        category = last.result.get("errorCategory")
        if category == "permission":
            return None
        if category == "business" and not last.result.get("isRetryable"):
            if "escalate" in tools and not _called(events, "escalate"):
                reason = (last.result.get("customerMessage") or "A tool refused the "
                          "request") + " The customer still needs help."
                return tools["escalate"], _tool_input(tools["escalate"],
                                                      {"reason": reason}, facts)
            return None
        if category == "validation" and "lookup_order" in tools and order_id:
            if not any(e.name == "lookup_order" and not e.is_error for e in events):
                return tools["lookup_order"], _tool_input(tools["lookup_order"],
                                                          {"order_id": order_id}, facts)

    if "get_customer" in tools and slots.get("email") and not _called(events, "get_customer"):
        return tools["get_customer"], _tool_input(tools["get_customer"],
                                                  {"email": slots["email"]}, facts)

    # A coordinator: delegate to each agent tool once, passing on what came back.
    agents = [t for t in tool_list if str(t.get("name", "")).endswith("_agent")]
    if agents:
        for agent in agents:
            if not _called(events, agent.get("name")):
                values: Dict[str, Any] = {}
                if order_id:
                    values["order_id"] = order_id
                earlier = [e.result for e in events if e.result is not None]
                if earlier:
                    values["facts"] = earlier[-1] if isinstance(earlier[-1], str) \
                        else json.dumps(earlier[-1])
                return agent, _tool_input(agent, values, facts)
        return None

    if "lookup_order" in tools and order_id and not _called(events, "lookup_order"):
        return tools["lookup_order"], _tool_input(tools["lookup_order"],
                                                  {"order_id": order_id}, facts)

    if "list_charges" in tools and order_id and not _called(events, "list_charges"):
        return tools["list_charges"], _tool_input(tools["list_charges"],
                                                  {"order_id": order_id}, facts)
    if "refund_charge" in tools and _called(events, "list_charges") \
            and not _called(events, "refund_charge"):
        charges = _called(events, "list_charges")[-1].result
        amounts = [c.get("amount") for c in charges if isinstance(c, dict)] \
            if isinstance(charges, list) else []
        duplicate = next((a for a in amounts if amounts.count(a) > 1), None)
        if duplicate is not None:
            return tools["refund_charge"], _tool_input(tools["refund_charge"],
                                                       {"order_id": order_id,
                                                        "amount": duplicate}, facts)

    lookups = _called(events, "lookup_order")
    if "start_return" in tools and lookups and not _called(events, "start_return") \
            and _wants_return(text) and _order_of(lookups[-1].result) is not None:
        return tools["start_return"], _tool_input(
            tools["start_return"], {"order_id": lookups[-1].input.get("order_id", order_id)},
            facts)
    return None


def compose_after_tools(facts: PromptFacts, events: List[ToolEvent]) -> Optional[str]:
    """The final answer, written from what the tools returned."""
    if not events:
        return None
    text = facts.effective_user_text
    last = events[-1]
    result = last.result

    if last.name.endswith("_agent") or any(e.name.endswith("_agent") for e in events):
        reports = [e.result for e in events if isinstance(e.result, str)]
        return "I checked your order and the charges. " + " ".join(reports)

    if last.name == "escalate" and isinstance(result, dict):
        refusal = next((e.result.get("customerMessage") for e in reversed(events)
                        if e.is_error and isinstance(e.result, dict)), None)
        refusal = (refusal or "I could not complete that myself").rstrip(".")
        return ("I couldn't do that myself ({}{}), so I've handed your case to a "
                "colleague - ticket {} - who will get back to you.".format(
                    refusal[:1].lower(), refusal[1:], result.get("ticket", "opened")))

    if last.is_error and isinstance(result, dict):
        message = result.get("customerMessage") or "I could not complete that."
        if result.get("errorCategory") == "permission":
            return "{} Then I can look into the order for you.".format(message)
        return "I'm sorry - {}{}".format(message[:1].lower(), message[1:])

    if last.name == "refund_charge" and isinstance(result, dict):
        amount = result.get("refunded")
        if isinstance(amount, float) and amount.is_integer():
            amount = int(amount)
        return ("I found a duplicate charge of {} on order {} and refunded it.".format(
            amount, result.get("order_id")))

    if last.name == "start_return" and isinstance(result, dict):
        order = next((_order_of(e.result) for e in reversed(events)
                      if e.name == "lookup_order" and _order_of(e.result)), None) or {}
        return ("Your return for the {} has started - your return label is {}.".format(
            order.get("item", "item"), result.get("return_label", "on its way")))

    if last.name == "lookup_order":
        order_id = last.input.get("order_id")
        if _not_found(result):
            return ("I couldn't find an order {} - could you check the number and send it "
                    "again?".format(order_id))
        order = _order_of(result)
        if order is None:
            return None
        days = order.get("delivered_days_ago")
        if _wants_return(text) and isinstance(days, int):
            if days <= RETURN_WINDOW_DAYS:
                return ("Yes - the {} from order {} was delivered {} days ago, so you "
                        "still have {} days left in our {}-day return window."
                        .format(order.get("item"), order_id, days,
                                RETURN_WINDOW_DAYS - days, RETURN_WINDOW_DAYS))
            return ("I'm sorry - the {} from order {} was delivered {} days ago, which is "
                    "past our {}-day return window.".format(order.get("item"), order_id,
                                                           days, RETURN_WINDOW_DAYS))
        if order.get("status") == "delivered" and isinstance(days, int):
            return "Order {} is a {}, delivered {} days ago.".format(
                order_id, order.get("item"), days)
        return "Order {} is a {}, {}.".format(order_id, order.get("item"), order.get("status"))

    if last.name == "get_customer" and isinstance(result, dict):
        if result.get("found") is False:
            return "I couldn't find an account with that email - could you check it?"
        return "Thanks, I found your account. What can I help you with?"
    return None


# ---------------------------------------------------------------------------
# 9. Rendering and the create() dispatch
# ---------------------------------------------------------------------------


def _apply_word_limit(text: str, limit: Optional[int]) -> str:
    if not limit:
        return text
    words = text.split()
    if len(words) <= limit:
        return text
    clipped = " ".join(words[:limit])
    return clipped.rstrip(",;:") + "."


def _obey_end_marker(text: str, stop_sequences: Sequence[str], facts: PromptFacts) -> str:
    """A well-behaved model emits the end marker the prompt asked it to emit."""
    for seq in stop_sequences:
        if seq and seq.lower() in facts.effective_user_text.lower():
            return (text.rstrip() + "\n" + seq +
                    "\nText after the stop sequence is never returned by the API.")
    return text


def _earliest_stop(text: str, stop_sequences: Sequence[str]) -> Tuple[Optional[str], Optional[int]]:
    best_seq, best_idx = None, None
    for seq in stop_sequences:
        if not seq:
            continue
        idx = text.find(seq)
        if idx != -1 and (best_idx is None or idx < best_idx):
            best_seq, best_idx = seq, idx
    return best_seq, best_idx


#: Placeholder left in a lab skeleton. `...` is real Python (Ellipsis), so without this
#: check it would travel a long way before failing with something unhelpful.
def _has_blank(value: Any) -> bool:
    if value is Ellipsis:
        return True
    if isinstance(value, dict):
        return any(_has_blank(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_blank(v) for v in value)
    return False


def _reject_blanks(**kwargs: Any) -> None:
    blanks = sorted(name for name, value in kwargs.items() if _has_blank(value))
    if not blanks:
        return
    which = blanks[0] if len(blanks) == 1 else ", ".join(blanks)
    raise BadRequestError(
        "You still have ... in this request ({}). Replace every ... with the value "
        "named in the comment beside it, then run the cell again.".format(which))


#: Model families that, in the course's current API, decide their own reasoning and
#: reject the old sampling controls.
_CURRENT_FAMILIES = re.compile(r"opus|sonnet|fable", re.I)
EFFORT_LEVELS = ("low", "medium", "high", "max")


def _reject_retired_controls(model: Any, temperature: Any, extra: Dict[str, Any],
                             messages: Sequence[Any]) -> None:
    """What the current Opus, Sonnet and Fable models refuse, refused the same way.

    Older code and tutorials set temperature, top_p or top_k, or prefill the start
    of the reply. The course teaches leaving them out; a lab that still sends them
    should fail the way the real API does, with the reason attached.
    """
    if not _CURRENT_FAMILIES.search(str(model)):
        return
    retired = [name for name, value in (("temperature", temperature),
                                        ("top_p", extra.get("top_p")),
                                        ("top_k", extra.get("top_k"))) if value is not None]
    if retired:
        raise BadRequestError(
            "{} is not supported by {}. The current Opus, Sonnet and Fable models "
            "reject temperature, top_p and top_k - leave them out, and choose the "
            "model and the effort level instead.".format(", ".join(retired), model))
    last = messages[-1] if messages else None
    role = last.get("role") if isinstance(last, dict) else getattr(last, "role", None)
    if role == "assistant":
        raise BadRequestError(
            "{} does not support prefilling the assistant's reply: the last message "
            "must come from the user.".format(model))


def _check_tool_protocol(messages: Sequence[Any], tools: Any = None) -> None:
    """The pairing rules the real API enforces on a tool conversation.

    Every tool_use in an assistant turn needs a tool_result in the very next user turn,
    and every tool_result must answer a tool_use from the turn just before it. The
    messages are the same an API call would reject, so a learner meets them here first.
    """
    previous_ids: List[str] = []
    for index, msg in enumerate(messages):
        role = _role_of(msg)
        body = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
        blocks = [] if isinstance(body, str) else list(body or [])
        uses = [str(_block_get(b, "id", "")) for b in blocks if _block_type(b) == "tool_use"]
        results = [b for b in blocks if _block_type(b) == "tool_result"]
        if previous_ids:
            answered = [str(_block_get(b, "tool_use_id", "")) for b in results]
            missing = [i for i in previous_ids if i not in answered]
            if role != "user" or missing:
                raise BadRequestError(
                    "messages.{}: tool_use ids were found without tool_result blocks "
                    "immediately after: {}. Each tool_use block must have a corresponding "
                    "tool_result block in the next message.".format(
                        index, ", ".join(missing or previous_ids)))
        for block in results:
            ref = str(_block_get(block, "tool_use_id", ""))
            if ref not in previous_ids:
                raise BadRequestError(
                    "messages.{}.content: unexpected tool_use_id found in tool_result "
                    "blocks: {}. Each tool_result block must have a corresponding tool_use "
                    "block in the previous message.".format(index, ref))
            content = _block_get(block, "content")
            if content is not None and not isinstance(content, (str, list)):
                raise BadRequestError(
                    "messages.{}.content: tool_result content must be a string or a list "
                    "of content blocks, got {}. Send the result as text: "
                    "json.dumps(result).".format(index, type(content).__name__))
        previous_ids = uses if role == "assistant" else []
        if (uses or results) and not tools:
            raise BadRequestError(
                "Requests which include tool_use or tool_result blocks must define tools. "
                "Send tools=... again with every request of a tool conversation.")


def _read_output_config(output_config: Any) -> Tuple[Optional[str], Any]:
    """(effort, format) from output_config, checked the way the API checks them."""
    if output_config is None:
        return None, None
    if not isinstance(output_config, dict):
        raise BadRequestError("output_config must be a dictionary, for example "
                              '{"effort": "low"}')
    unknown = sorted(set(output_config) - {"effort", "format"})
    if unknown:
        raise BadRequestError("output_config does not accept {}".format(", ".join(unknown)))
    effort = output_config.get("effort")
    if effort is not None and effort not in EFFORT_LEVELS:
        raise BadRequestError("output_config.effort must be one of {}, got {!r}".format(
            ", ".join(EFFORT_LEVELS), effort))
    return effort, output_config.get("format")


class MockMessages:
    def __init__(self, client: "MockAnthropic") -> None:
        self._client = client

    def create(self, *, model: str, max_tokens: int, messages: Sequence[Any],
               system: Any = None, temperature: Any = None,
               stop_sequences: Optional[Sequence[str]] = None,
               tools: Optional[Sequence[dict]] = None, tool_choice: Any = None,
               output_config: Optional[dict] = None,
               stream: bool = False, **extra: Any) -> Message:

        # Labs hand the learner a skeleton with ... in the places to fill in. Catch a
        # forgotten one here, where we can say what to do, rather than letting it fail
        # somewhere deeper with a message about Ellipsis.
        _reject_blanks(model=model, max_tokens=max_tokens, messages=messages,
                       system=system, temperature=temperature,
                       stop_sequences=stop_sequences, tools=tools,
                       tool_choice=tool_choice, output_config=output_config)
        _reject_retired_controls(model, temperature, extra, messages)
        _check_tool_protocol(messages, tools)
        effort, fmt = _read_output_config(output_config)

        if stream:
            raise NotImplementedError(
                "shopassist_lab does not simulate streaming. Set stream=False, or run "
                "with a real FUELIX_API_KEY. See lab_info() for the full gap list.")
        for key in extra:
            if key in ("top_p", "top_k"):
                continue
            _warn_once("parameter {!r} is accepted but ignored by the simulator.".format(key))
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
            raise BadRequestError("max_tokens must be a positive integer")
        if not messages:
            raise BadRequestError("messages must not be empty")

        self._client._calls += 1
        call_index = self._client._calls

        system_text = _flatten(system)
        temp = 0.0 if temperature is None else float(temperature)
        stops = tuple(stop_sequences or ())
        tool_list = list(tools or [])

        facts = analyze_prompt(system_text, messages)
        bb = build_blackboard(messages)

        fingerprint = _fingerprint(system_text, facts.effective_user_text,
                                   facts.turns, [t.get("name") for t in tool_list])
        rng = _rng(fingerprint, call_index if temp > 0 else 0)

        # --- tools first -------------------------------------------------
        kind = tool_choice.get("type") if isinstance(tool_choice, dict) else tool_choice
        if tool_list and kind in ("any", "tool") and _CURRENT_FAMILIES.search(str(model)):
            raise BadRequestError(
                'tool_choice {!r} is not supported by {}. The current Opus, Sonnet and '
                'Fable models reject "any" and a forced tool: use "auto" with tools, or '
                "output_config.format for structured output. (On the exam, forcing a "
                "tool is still the answer the guide teaches.)".format(kind, model))
        if tool_list and kind in (None, "auto"):
            events = tool_history(messages)
            plan = plan_next(tool_list, facts, events)
            if plan is not None:
                tool, tool_input = plan
                block = ToolUseBlock(id="toolu_sim_" + fingerprint[:16],
                                     name=tool["name"], input=tool_input)
                content_ = ([TextBlock("Let me check that.")] if LabConfig.tool_preambles
                            else []) + [block]
                return self._finish(content_, model, "tool_use", None, messages,
                                    "agent.tool:" + tool["name"], temp, call_index,
                                    fingerprint, has_system=facts.has_system,
                                    effort=effort)
            final = compose_after_tools(facts, events)
            if final is not None:
                return self._finish([TextBlock(final)], model, "end_turn", None, messages,
                                    "agent.final", temp, call_index, fingerprint,
                                    has_system=facts.has_system, effort=effort)
            # Nothing to call and nothing to report: answer in text as usual.
            decision = None
        else:
            decision = decide_tool(tool_list, tool_choice, facts, bb)
        if decision is not None:
            tool, tool_input, forced = decision
            block = ToolUseBlock(
                id="toolu_sim_" + fingerprint[:16],
                name=tool["name"],
                input=tool_input,
            )
            if forced or not LabConfig.tool_preambles:
                content: List[Any] = [block]
            else:
                content = [TextBlock("Let me look that up."), block]
            return self._finish(content, model, "tool_use", None, messages,
                                "tool:" + tool["name"], temp, call_index, fingerprint,
                                has_system=facts.has_system, effort=effort)

        # --- structured outputs: JSON that fits the schema it was given ---
        computed = None
        if fmt is not None:
            computed = ("format.json_schema",
                        json.dumps(fill_json_schema(_check_format(fmt), messages)))

        # --- a reply computed from the customer's own words ---------------
        if computed is None:
            computed = compute_reply(facts)
        if computed is not None:
            trace, text = computed
            variant_index = 0
        else:
            # --- otherwise, a fixture -------------------------------------
            scenario = select_scenario(facts, bb, bool(tool_list))
            trace = scenario.id
            text, variant_index = pick_variant(scenario.variants, temp, rng, fingerprint)
            text = _apply_word_limit(text, facts.word_limit)
            text = _obey_end_marker(text, stops, facts)

        # stop_sequences and max_tokens race; whichever comes first wins.
        budget = max_tokens * LabConfig.chars_per_token
        seq, idx = _earliest_stop(text, stops)
        if idx is not None and idx <= budget:
            return self._finish([TextBlock(text[:idx])], model, "stop_sequence", seq,
                                messages, trace, temp, call_index, fingerprint,
                                variant_index, has_system=facts.has_system, effort=effort)
        if len(text) > budget:
            return self._finish([TextBlock(text[:int(budget)])], model, "max_tokens", None,
                                messages, trace, temp, call_index, fingerprint,
                                variant_index, has_system=facts.has_system, effort=effort)
        return self._finish([TextBlock(text)], model, "end_turn", None,
                            messages, trace, temp, call_index, fingerprint,
                            variant_index, has_system=facts.has_system, effort=effort)

    def _finish(self, content: List[Any], model: str, stop_reason: str,
                stop_sequence: Optional[str], messages: Sequence[Any], trace: str,
                temperature: float, call_index: int, fingerprint: str,
                variant_index: int = 0, has_system: bool = False,
                effort: Optional[str] = None) -> Message:
        out_text = "".join(getattr(b, "text", "") for b in content)
        out_tokens = est_tokens(out_text) + sum(
            est_tokens(json.dumps(b.input)) for b in content if b.type == "tool_use")
        in_text = "\n".join(_flatten(
            m.get("content") if isinstance(m, dict) else getattr(m, "content", None))
            for m in messages)

        msg = Message(
            id="msg_sim_" + fingerprint[:12],
            content=content,
            model=model,
            stop_reason=stop_reason,
            stop_sequence=stop_sequence,
            usage=Usage(input_tokens=est_tokens(in_text), output_tokens=out_tokens),
        )
        msg._lab = {
            "scenario": trace,
            "temperature": temperature,
            "call_index": call_index,
            "variant": variant_index,
            "fingerprint": fingerprint,
            "has_system": has_system,
            "effort": effort,
        }
        if LabConfig.verbose:
            print("[sim] {} | temp={} | stop={} | out={}tok".format(
                trace, temperature, stop_reason, out_tokens))
        return msg


class MockAnthropic:
    """Offline stand-in for anthropic.Anthropic."""

    def __init__(self, **kwargs: Any) -> None:
        self._calls = 0
        self.messages = MockMessages(self)
        self.api_key = kwargs.get("api_key")
        _CLIENTS.append(self)
        _banner(live=False)

    def __repr__(self) -> str:
        return "<shopassist_lab.MockAnthropic simulated=True calls={}>".format(self._calls)


def _banner(live: bool) -> None:
    global _BANNER_SHOWN
    if _BANNER_SHOWN:
        return
    _BANNER_SHOWN = True
    if live:
        print("[shopassist_lab] LIVE MODE - real anthropic SDK, real API calls, real cost.")
    else:
        print("[shopassist_lab] SIMULATED MODE - replies are canned fixtures, not Claude.")
        print("[shopassist_lab] No FUELIX_API_KEY found, so everything runs offline.")
        print("[shopassist_lab] Call lab_info() for details, explain(message) for any reply.")


def Anthropic(**kwargs: Any) -> Any:  # noqa: N802 - deliberately mirrors the SDK name
    """Return a real anthropic client when a key is available, otherwise the simulator."""
    if _has_real_key():
        try:
            real = sys.modules.get("anthropic")
            if real is None or getattr(real, "__shopassist_lab__", False):
                import importlib
                # The shim may be standing in for an installed SDK because the key only
                # arrived with load_dotenv(). Drop it, and its anthropic.types, so the
                # real package imports cleanly.
                for name in list(sys.modules):
                    if (name == "anthropic" or name.startswith("anthropic.")) and \
                            getattr(sys.modules[name], "__shopassist_lab__", False):
                        del sys.modules[name]
                real = importlib.import_module("anthropic")
            if not getattr(real, "__shopassist_lab__", False):
                _banner(live=True)
                return real.Anthropic(**_fuelix_kwargs(kwargs))
        except Exception as exc:  # pragma: no cover - depends on the environment
            _warn_once("a key is set but the real anthropic SDK could not be loaded "
                       "({}). Falling back to the simulator.".format(exc))
    return MockAnthropic(**kwargs)


# ---------------------------------------------------------------------------
# 10. dotenv replacement and shim installation
# ---------------------------------------------------------------------------


def _dotenv_values(path: str = ".env") -> Dict[str, str]:
    values: Dict[str, str] = {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                values[key.strip()] = value.strip().strip("'\"")
    except OSError:
        pass
    return values


def load_dotenv(dotenv_path: Optional[str] = None, *_: Any, **__: Any) -> bool:
    """Minimal python-dotenv replacement. Absent .env is not an error."""
    path = dotenv_path or ".env"
    values = _dotenv_values(path)
    for key, value in values.items():
        os.environ.setdefault(key, value)
    return bool(values)


# anthropic 1.0 (August 2026) removed temperature / top_p / top_k from
# messages.create(). A lab notebook taken home and run against that SDK would die
# with a TypeError on the very line the temperature lab is about, so when the
# installed SDK has dropped the parameter we forward it through extra_body and the
# code the student wrote keeps running unchanged. Newer models (Claude Sonnet 5,
# Opus 4.7 and later) reject the parameter server-side; that 400 is left alone.
_SAMPLING_PARAMS = ("temperature", "top_p", "top_k")


def _forward_sampling_params(kwargs: Dict[str, Any], version: str) -> Dict[str, Any]:
    """Move temperature / top_p / top_k out of the keyword arguments into extra_body."""
    moved = {k: kwargs.pop(k) for k in _SAMPLING_PARAMS if k in kwargs}
    if moved:
        _warn_once(
            "anthropic {} no longer accepts {} in messages.create(); forwarding it via "
            "extra_body so this notebook still runs. Claude Sonnet 5 / Opus 4.7+ reject "
            "the parameter altogether - drop it when you move to those models.".format(
                version, ", ".join(sorted(moved))))
        body = dict(kwargs.get("extra_body") or {})
        body.update(moved)
        kwargs["extra_body"] = body
    return kwargs


def _patch_real_sdk() -> None:
    """Let messages.create() on anthropic>=1 keep accepting the 0.x sampling params."""
    try:
        import inspect
        import anthropic as real
        from anthropic.resources.messages import Messages
    except Exception:  # pragma: no cover - depends on the environment
        return
    create = Messages.create
    if getattr(create, "__shopassist_lab__", False):
        return  # already patched in this process
    try:
        if "temperature" in inspect.signature(create).parameters:
            return  # this SDK still takes the parameter natively
    except (TypeError, ValueError):  # pragma: no cover
        return
    version = str(getattr(real, "__version__", "?"))

    def create_compat(self: Any, *args: Any, **kwargs: Any) -> Any:
        return create(self, *args, **_forward_sampling_params(kwargs, version))

    create_compat.__name__ = create.__name__
    create_compat.__doc__ = create.__doc__
    create_compat.__shopassist_lab__ = True  # type: ignore[attr-defined]
    create_compat.__wrapped_create__ = create  # type: ignore[attr-defined]
    Messages.create = create_compat  # type: ignore[assignment]


# -- A web server without the web: stand-ins for fastapi and pydantic ----------
#
# The chat-page lecture runs a FastAPI server. The lab workspace has neither fastapi
# nor pydantic, and no browser either. These stand-ins record the routes the learner
# declares and build the request body the way pydantic would; browser_post() plays the
# page. Installed packages are never shadowed: anthropic itself depends on pydantic.


class _Route:
    def __init__(self, path: str, methods: Sequence[str], endpoint: Any) -> None:
        self.path = path
        self.methods = set(methods)
        self.endpoint = endpoint


class _WebApp:
    """Stands in for fastapi.FastAPI: records the endpoints, serves nothing."""

    def __init__(self, **_: Any) -> None:
        self.routes: List[_Route] = []

    def _route(self, method: str, path: str) -> Callable:
        if not isinstance(path, str) or not path.startswith("/"):
            raise ValueError("A path starts with a slash, like \"/chat\" - got {!r}."
                             .format(path))

        def deco(fn: Callable) -> Callable:
            self.routes.append(_Route(path, (method,), fn))
            return fn
        return deco

    def post(self, path: str, **_: Any) -> Callable:
        return self._route("POST", path)

    def get(self, path: str, **_: Any) -> Callable:
        return self._route("GET", path)

    def mount(self, path: str, app: Any = None, name: Any = None) -> None:
        self.routes.append(_Route(path, (), app))


class _StaticFiles:
    def __init__(self, directory: Any = None, html: bool = False, **_: Any) -> None:
        self.directory, self.html = directory, html


class _BaseModel:
    """Stands in for pydantic.BaseModel: declared fields are required and typed."""

    def __init__(self, **data: Any) -> None:
        fields: Dict[str, Any] = {}
        for klass in reversed(type(self).__mro__):
            fields.update(klass.__dict__.get("__annotations__", {}))
        for name, kind in fields.items():
            if name not in data:
                raise ValueError("{}: Field required - the request body has no {!r}."
                                 .format(name, name))
            if kind in (str, "str") and not isinstance(data[name], str):
                raise ValueError("{}: Input should be a valid string.".format(name))
            setattr(self, name, data[name])
        # Like pydantic, fields the model does not declare are dropped.
        self.__fields_set__ = set(fields)

    def model_dump(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self.__fields_set__}


def _install_web_shims() -> None:
    if not _module_importable("fastapi"):
        fastapi = types.ModuleType("fastapi")
        fastapi.FastAPI = _WebApp
        fastapi.__shopassist_lab__ = True
        static = types.ModuleType("fastapi.staticfiles")
        static.StaticFiles = _StaticFiles
        static.__shopassist_lab__ = True
        fastapi.staticfiles = static
        sys.modules["fastapi"] = fastapi
        sys.modules["fastapi.staticfiles"] = static
    if not _module_importable("pydantic"):
        pydantic = types.ModuleType("pydantic")
        pydantic.BaseModel = _BaseModel
        pydantic.__shopassist_lab__ = True
        sys.modules["pydantic"] = pydantic


def browser_post(app: Any, path: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Play the chat page: POST JSON to the app's endpoint and return the JSON reply.

    Works on the stand-in app and on a real FastAPI app alike: it finds the endpoint
    by path, builds the body model from the endpoint's annotation, and sends the
    result through json, as it would travel over the wire.
    """
    route = next((r for r in reversed(list(getattr(app, "routes", [])))
                  if getattr(r, "path", None) == path
                  and "POST" in (getattr(r, "methods", None) or ())), None)
    if route is None:
        raise LookupError("The server has no POST {0} endpoint. Decorate the function "
                          "with @app.post({0!r}).".format(path))
    import inspect
    params = list(inspect.signature(route.endpoint).parameters.values())
    model = params[0].annotation if params else None
    if isinstance(model, str):   # postponed annotations: look the name up
        model = getattr(route.endpoint, "__globals__", {}).get(model, model)
    body = model(**data) if isinstance(model, type) else data
    print("  browser -> server   POST {} {}".format(path, json.dumps(data)))
    result = route.endpoint(body)
    wire = json.dumps(result)
    print('  server log          "POST {} HTTP/1.1" 200 OK'.format(path))
    print("  server -> browser   {}".format(wire))
    return json.loads(wire)


def install(force: bool = False) -> None:
    """Make `from anthropic import Anthropic` and `from dotenv import load_dotenv` work.

    An installed SDK is used as-is when a key is already set. Without a key the
    simulator stands in even for an installed SDK - otherwise a learner who installed
    anthropic at home would hit an authentication error on the first request. If the key
    arrives later, from load_dotenv(), Anthropic() still returns the real client.
    """
    _export_fuelix_env()
    real_sdk = _module_importable("anthropic")
    if not force and real_sdk:
        _patch_real_sdk()
    if force or not real_sdk or not _has_real_key():
        shim = types.ModuleType("anthropic")
        shim.Anthropic = Anthropic
        shim.APIError = APIError
        shim.BadRequestError = BadRequestError
        shim.AuthenticationError = AuthenticationError
        shim.__version__ = __version__ + "+shopassist-lab"
        shim.__shopassist_lab__ = True

        types_mod = types.ModuleType("anthropic.types")
        types_mod.__shopassist_lab__ = True
        types_mod.TextBlock = TextBlock
        types_mod.ToolUseBlock = ToolUseBlock
        types_mod.Message = Message
        types_mod.Usage = Usage
        shim.types = types_mod

        sys.modules["anthropic"] = shim
        sys.modules["anthropic.types"] = types_mod

    _install_web_shims()

    if force or not _module_importable("dotenv"):
        shim = types.ModuleType("dotenv")
        shim.load_dotenv = load_dotenv
        shim.dotenv_values = _dotenv_values
        shim.find_dotenv = lambda *a, **k: ""
        shim.__shopassist_lab__ = True
        sys.modules["dotenv"] = shim


# ---------------------------------------------------------------------------
# 11. Introspection helpers
# ---------------------------------------------------------------------------


def lab_info() -> None:
    """Print exactly what is real and what is simulated."""
    live = _has_real_key()
    print("shopassist_lab {}".format(__version__))
    print("  mode: {}".format("LIVE (real Claude API)" if live else "SIMULATED (offline)"))
    print("")
    print("  Faithfully simulated:")
    print("    max_tokens -> truncation and stop_reason='max_tokens'")
    print("    output_config -> effort recorded; format returns JSON fitting the schema")
    print("    system -> detected rules change which reply is returned")
    print("    tools -> tool_use blocks; the next step follows your tool_results")
    print("    tool protocol -> every tool_use needs its tool_result, as in the API")
    print("    temperature, top_p, top_k, prefill, forced tool_choice -> rejected by")
    print("      current Opus/Sonnet/Fable models, as in the API")
    print("    message shape -> .content, .stop_reason, .usage, .model_dump_json()")
    print("")
    print("  NOT simulated (raises or is ignored):")
    print("    streaming, prompt caching, extended thinking, batches, vision")
    print("    genuine language understanding - replies are fixtures, not generated")
    print("")
    print("  Scenarios loaded: {}".format(len(SCENARIOS)))
    print("  Checks available: {}".format(", ".join(str(k) for k in sorted(TASKS))))
    print("")
    print("  To run against the real API: set FUELIX_API_KEY and re-run the notebook.")


def explain(message: Any) -> None:
    """Explain why the simulator returned this particular reply."""
    if not getattr(message, "simulated", False):
        print("This is a real Claude response, not a simulated one - nothing to explain.")
        return
    meta = getattr(message, "_lab", {}) or {}
    trace = meta.get("scenario", "?")
    scenario = next((s for s in SCENARIOS + [FALLBACK] if s.id == trace), None)
    if scenario is None and trace in COMPUTED_NOTES:
        print("Reply:       {} (computed, not a fixture)".format(trace))
        print("Why:         {}".format(COMPUTED_NOTES[trace]))
    else:
        print("Scenario:    {} (lesson {})".format(
            trace, scenario.lesson if scenario else "?"))
        if scenario and scenario.note:
            print("Why:         {}".format(scenario.note))
    if scenario:
        print("Variant:     {} of {}{}".format(
            meta.get("variant", 0), len(scenario.variants),
            "  (temperature=0 always picks 0)" if meta.get("temperature", 0) == 0 else ""))
    print("stop_reason: {}".format(message.stop_reason))
    print("Tokens:      {} in, {} out".format(
        message.usage.input_tokens, message.usage.output_tokens))
    print("")
    print("Try next:    drop the system= argument and re-run, or set max_tokens=15,")
    print("             or raise temperature to 1.0 and call it three times.")


# ---------------------------------------------------------------------------
# 12. Self-check harness
#
# These checks are deliberately written against BEHAVIOUR, never against exact
# fixture strings, so every one of them also passes when a student re-runs the
# notebook with a real FUELIX_API_KEY. They are themselves a worked example
# of the course's Domain 4 topic: code-based grading.
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    task_id: str
    passed: bool
    title: str
    message: str = ""
    hint: str = ""

    def __bool__(self) -> bool:
        return self.passed

    def __repr__(self) -> str:
        head = "[{}] Task {} - {}".format(
            "PASS" if self.passed else "FAIL", self.task_id, self.title)
        body = "\n  " + self.message if self.message else ""
        tail = "\n  Hint: " + self.hint if (self.hint and not self.passed) else ""
        return head + body + tail


@dataclass
class _Task:
    task_id: str
    title: str
    #: Lab slug, e.g. "first_request". Deliberately not a lesson number - lessons get
    #: renumbered as the course grows, and these ids are shown to the learner.
    lab: str
    needs: Tuple[str, ...]
    fn: Callable


TASKS: Dict[str, _Task] = {}


def task(task_id: str, title: str, lab: str, needs: Sequence[str] = ()) -> Callable:
    def deco(fn: Callable) -> Callable:
        TASKS[str(task_id)] = _Task(str(task_id), title, lab, tuple(needs), fn)
        return fn
    return deco


def _ok(msg: str) -> Tuple[bool, str, str]:
    return True, msg, ""


def _fail(msg: str, hint: str = "") -> Tuple[bool, str, str]:
    return False, msg, hint


def check(task_id: Any, **kwargs: Any) -> CheckResult:
    """Verify one lab task. The result is truthy when the task passes."""
    key = str(task_id)
    entry = TASKS.get(key)
    if entry is None:
        return CheckResult(key, False, "unknown task",
                           "There is no check with id {!r}.".format(key),
                           "Available: " + ", ".join(sorted(TASKS)))
    missing = [n for n in entry.needs if n not in kwargs]
    if missing:
        return CheckResult(key, False, entry.title,
                           "This check needs {}.".format(", ".join(repr(m) for m in missing)),
                           "Call it as check({!r}, {})".format(
                               task_id, ", ".join(n + "=..." for n in entry.needs)))
    try:
        passed, msg, hint = entry.fn(**kwargs)
    except Exception as exc:
        return CheckResult(key, False, entry.title,
                           "The check raised {}: {}".format(type(exc).__name__, exc),
                           "That usually means a variable holds a different kind of "
                           "value than the task expected.")
    return CheckResult(key, passed, entry.title, msg, hint)


def check_all(**artifacts: Any) -> List[CheckResult]:
    """Run every check for which the required artifacts were supplied."""
    results = []
    # Registration order is notebook order. Sorting by id would scramble it now that
    # ids are names rather than numbers.
    for key in TASKS:
        entry = TASKS[key]
        if all(n in artifacts for n in entry.needs):
            result = check(key, **{n: artifacts[n] for n in entry.needs})
            results.append(result)
            print(result)
    passed = sum(1 for r in results if r.passed)
    print("\n{} of {} checks passed.".format(passed, len(results)))
    return results


# -- shared text predicates --------------------------------------------------


def _asks_for_order_number(text: str) -> bool:
    return bool(re.search(r"order\s*(?:number|id|#)", text, re.I))


def _promises_refund(text: str) -> bool:
    return bool(re.search(
        r"(?:i(?:'ve| have)|we(?:'ve| have))\s+(?:gone ahead and\s+)?"
        r"(?:approved|processed|issued|refunded)"
        r"|refund is approved|consider it refunded|i can refund that"
        r"|approved a full refund", text, re.I))


def _text_of(message: Any) -> str:
    if isinstance(message, str):
        return message
    blocks = getattr(message, "content", None) or []
    return "".join(getattr(b, "text", "") for b in blocks if getattr(b, "type", "") == "text")


def _role_of(msg: Any) -> str:
    return (msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "")) or ""


# -- Lab 1 tasks -------------------------------------------------------------


def _effort_of(message: Any) -> Optional[str]:
    return (getattr(message, "_lab", {}) or {}).get("effort")


@task("first_request", title="Make your first Claude request", lab="first_request",
      needs=("response",))
def _task1(response: Any, **_: Any) -> Tuple[bool, str, str]:
    if not hasattr(response, "content"):
        return _fail("`response` does not look like a Message object.",
                     "Assign the whole result: response = client.messages.create(...). "
                     "The next task reads several fields from it.")
    if not response.content:
        return _fail("The response came back with no content blocks.")
    first = response.content[0]
    if getattr(first, "type", None) != "text":
        return _fail("content[0] is a {!r} block, not a text block.".format(
            getattr(first, "type", "?")))
    if not first.text.strip():
        return _fail("The reply text is empty.")
    if getattr(response, "simulated", False) and _effort_of(response) != "low":
        return _fail("The request did not set effort to low.",
                     'Pass output_config={"effort": "low"} - a simple question does not '
                     "need Claude to think before it answers.")
    if response.stop_reason != "end_turn":
        return _fail("stop_reason is {!r}, so the reply is not complete.".format(
            response.stop_reason),
            "Use max_tokens=300 here so Claude can finish its two sentences.")
    return _ok("One call: the model, a ceiling for the reply, the effort, and the "
               "conversation so far. Every Claude feature in this course starts with "
               "exactly this.")


@task("response_object", title="Read the text, stop_reason and usage",
      lab="first_request",
      needs=("response", "reply_text", "stop_reason", "input_tokens", "output_tokens"))
def _task12_3(response: Any, reply_text: Any, stop_reason: Any, input_tokens: Any,
              output_tokens: Any, **_: Any) -> Tuple[bool, str, str]:
    values = (("reply_text", reply_text), ("stop_reason", stop_reason),
              ("input_tokens", input_tokens), ("output_tokens", output_tokens))
    blanks = [n for n, v in values if v is Ellipsis]
    if blanks:
        return _fail("You still have ... in {}.".format(", ".join(blanks)),
                     "Replace each ... with the field named in the comment beside it, "
                     "then run the cell again.")
    if reply_text != _text_of(response):
        return _fail("`reply_text` is not the text carried by this response.",
                     "Read it out of the object: response.content[0].text")
    if stop_reason != response.stop_reason:
        return _fail("`stop_reason` is {!r}, but response.stop_reason is {!r}.".format(
            stop_reason, response.stop_reason))
    usage = getattr(response, "usage", None)
    if usage is None:
        return _fail("This response has no `usage` field to read.")
    if input_tokens != usage.input_tokens or output_tokens != usage.output_tokens:
        return _fail("The token counts do not match response.usage.",
                     "Read them from the object - response.usage.input_tokens and "
                     "response.usage.output_tokens - rather than typing the numbers in.")
    return _ok("The text is what the customer sees. stop_reason says why Claude stopped "
               "writing, and usage counts what you sent and what came back - which is "
               "what you pay for.")


@task("stop_reason", title="max_tokens is a ceiling: check stop_reason",
      lab="first_request", needs=("short", "long", "safe_reply"))
def _task2(short: Any, long: Any, safe_reply: Any, **_: Any) -> Tuple[bool, str, str]:
    if short.stop_reason != "max_tokens":
        return _fail("The short call returned stop_reason={!r}, not 'max_tokens'.".format(
            short.stop_reason),
            "Send the same request with max_tokens=20.")
    if long.stop_reason != "end_turn":
        return _fail("The 300-token reply from Task 1 did not end on its own.")
    if not callable(safe_reply):
        return _fail("`safe_reply` is not a function.")
    try:
        shown_long, shown_short = safe_reply(long), safe_reply(short)
    except Exception as exc:                                # noqa: BLE001
        return _fail("safe_reply() raised {}: {}".format(type(exc).__name__, exc))
    if shown_short is not None:
        return _fail("safe_reply() let the cut reply through.",
                     'Only a reply whose stop_reason is "end_turn" is complete. A '
                     '"max_tokens" reply ends mid-sentence and must not reach the '
                     "customer.")
    if shown_long != _text_of(long):
        return _fail("safe_reply() did not return the text of the complete reply.",
                     'Compare with "end_turn" - that is the value a finished reply has.')
    return _ok("max_tokens is a ceiling, not a target: at 20 tokens the reply stops "
               "mid-sentence and stop_reason says so. Nothing failed - the API did what "
               "you asked - which is exactly why your code has to check stop_reason before "
               "anything reaches a customer.")


@task("stateless", title="Two questions sent on their own", lab="conversation",
      needs=("first_reply", "second_reply"))
def _task_conv_1(first_reply: Any, second_reply: Any, **_: Any) -> Tuple[bool, str, str]:
    for name, value in (("first_reply", first_reply), ("second_reply", second_reply)):
        if value is Ellipsis or value is None:
            return _fail("`{}` is empty.".format(name),
                         "ask() has to return the reply text it found.")
        if not isinstance(value, str):
            return _fail("`{}` is a {}, not text.".format(name, type(value).__name__),
                         'Keep block.text from the block whose type is "text".')
    if not re.search(r"\b10 days\b|ten days", first_reply, re.I):
        return _fail("The first answer does not mention the days left.",
                     "Pass system=SYSTEM - the 30-day rule lives in the system line.")
    if "jacket" in second_reply.lower():
        return _fail("The second answer already knows about the jacket.",
                     "Send the second question on its own - a list with just that one "
                     "message. This task is the 'before' picture.")
    return _ok("The second answer asks which item you mean. Claude did not forget the "
               "jacket - it was never told. Every request stands alone, and Claude sees "
               "only what that request contains.")


@task("history", title="Keep the conversation in a list", lab="conversation",
      needs=("history",))
def _task_conv_2(history: Any, **_: Any) -> Tuple[bool, str, str]:
    if not isinstance(history, list) or len(history) < 4:
        return _fail("`history` should hold at least four messages after two turns, "
                     "got {}.".format(len(history) if isinstance(history, list) else
                                      type(history).__name__),
                     "say() appends the customer's message, then Claude's reply.")
    roles = [_role_of(m) for m in history]
    if roles[0] != "user" or any(roles[i] == roles[i - 1] for i in range(1, len(roles))):
        return _fail("The roles came out as {}.".format(roles),
                     'The customer\'s message goes in as "user", Claude\'s reply as '
                     '"assistant" - one after the other.')
    last = history[-1].get("content") if isinstance(history[-1], dict) else None
    if not isinstance(last, str) or "jacket" not in last.lower():
        return _fail("The last reply does not know about the jacket.",
                     "say() must send the WHOLE history list to ask(), not just the "
                     "newest message.")
    return _ok("Now the second answer knows the jacket and the days left. The memory of "
               "the conversation lives in your code: a list you send in full, every "
               "turn.")


@task("tokens_grow", title="Watch the input tokens grow", lab="conversation",
      needs=("history", "token_log", "growth"))
def _task_conv_3(history: Any, token_log: Any, growth: Any, **_: Any) -> Tuple[bool, str, str]:
    if not isinstance(history, list) or len(history) < 6:
        return _fail("`history` should hold six messages after three turns.",
                     "Run the cell with the third question once, after Task 2.")
    if not isinstance(token_log, list) or len(token_log) < 3:
        return _fail("`TOKEN_LOG` should hold the input tokens of the three turns.",
                     "Run Task 2 and then this cell, in order - Task 2 starts a fresh log.")
    if growth is Ellipsis:
        return _fail("`growth` still has ... in it.",
                     "The last turn's count minus the first: TOKEN_LOG[-1] - TOKEN_LOG[0].")
    if any(b <= a for a, b in zip(token_log, token_log[1:])):
        return _fail("The input tokens did not grow on every turn: {}.".format(token_log),
                     "Each turn must send the whole history, so each request is bigger.")
    if growth != token_log[-1] - token_log[0]:
        return _fail("`growth` is {!r}, but the last turn minus the first is {}.".format(
            growth, token_log[-1] - token_log[0]),
            "Use TOKEN_LOG[-1] - TOKEN_LOG[0].")
    return _ok("Input tokens per turn: {}. Every turn sends the whole history again, so "
               "each costs more than the last - the price of memory. A long chat can "
               "reach the context window, the most one request can hold.".format(token_log))


# -- Prompt evaluation tasks -------------------------------------------------


EVAL_EXPECTED = ("YES", "YES", "NO", "NO", "NO", "YES")


@task("test_set", title="Write the test set", lab="evaluation", needs=("cases",))
def _task17_1(cases: Any, **_: Any) -> Tuple[bool, str, str]:
    if not isinstance(cases, list) or len(cases) != 6:
        return _fail("CASES should hold the six cases, got {}.".format(
            len(cases) if isinstance(cases, list) else type(cases).__name__),
            "Do not add or remove cases - just fill in the expected verdicts.")
    for i, (case, wanted) in enumerate(zip(cases, EVAL_EXPECTED), 1):
        if not isinstance(case, tuple) or len(case) != 2:
            return _fail("Case {} is not a (question, verdict) pair.".format(i))
        question, verdict = case
        if verdict is Ellipsis:
            return _fail("Case {} still has ... for its verdict.".format(i),
                         'Write "YES" or "NO", in quotes.')
        if verdict not in ("YES", "NO"):
            return _fail("Case {} expects {!r}.".format(i, verdict),
                         'A verdict is "YES" or "NO", in capitals and quotes.')
        if verdict != wanted:
            return _fail("Case {} expects {}: {!r}.".format(i, verdict, question[:60]),
                         "Read it against the store's rules: returns within 30 days of "
                         "delivery, and no gift cards or underwear.")
    return _ok("Six questions, six answers you committed to before anything ran. That is "
               "what turns 'this reply looks fine' into something a program can check.")


@task("score", title="Score a prompt in code", lab="evaluation",
      needs=("evaluate", "v1_score"))
def _task17_2(evaluate: Any, v1_score: Any, **_: Any) -> Tuple[bool, str, str]:
    if not callable(evaluate):
        return _fail("`evaluate` is not a function.")
    if v1_score is Ellipsis or not isinstance(v1_score, int):
        return _fail("evaluate() did not return the score, got {!r}.".format(v1_score),
                     "Count a pass per matching reply and return the count at the end.")
    if v1_score == 6:
        return _fail("The first prompt scored 6 of 6.",
                     "It should not: it never mentions underwear. Check that a reply "
                     "passes only when it starts with the EXPECTED verdict.")
    if v1_score != 5:
        return _fail("The first prompt scored {} of 6, expected 5.".format(v1_score),
                     "A pass is reply.strip().upper().startswith(expected), and the score "
                     "adds one for each pass.")
    return _ok("5 of 6. The underwear case fails: Claude says yes, because the prompt "
               "never mentions underwear. The prompt has a gap, and the test set found it "
               "in seconds.")


@task("compare", title="Compare two prompts by their scores", lab="evaluation",
      needs=("v1_score", "v2_score", "prompt_v2"))
def _task17_3(v1_score: Any, v2_score: Any, prompt_v2: Any, **_: Any) -> Tuple[bool, str, str]:
    if not isinstance(prompt_v2, str) or not re.search(r"underwear", prompt_v2, re.I):
        return _fail("PROMPT_V2 does not mention underwear.",
                     "PROMPT_V2 is PROMPT_V1 plus NO_RETURN_RULE.")
    if v2_score != 6:
        return _fail("The second prompt scored {!r} of 6.".format(v2_score),
                     "Evaluate PROMPT_V2 with the same function and the same cases.")
    if v1_score != 5:
        return _fail("The first prompt's score should still be 5 - run Task 2 first.")
    return _ok("5 of 6, then 6 of 6, on the same six cases. The change was an improvement, "
               "and you have a number that says so, not a feeling. Six cases prove "
               "little on their own - the value is the comparison, and the set grows with "
               "every real question that went wrong.")


# -- Structured output tasks -------------------------------------------------


RETURN_REASONS = ("too_small", "too_big", "damaged", "other")


@task("schema", title="Describe the shape with a JSON schema", lab="structured_output",
      needs=("schema",))
def _task25_1(schema: Any, **_: Any) -> Tuple[bool, str, str]:
    if not isinstance(schema, dict) or schema.get("type") != "object":
        return _fail("SCHEMA should be an object schema.")
    if _has_blank(schema):
        return _fail("SCHEMA still has ... in it.",
                     "Replace each ... with the value named in the comment beside it.")
    props = schema.get("properties") or {}
    reason = props.get("reason") or {}
    if tuple(reason.get("enum") or ()) != RETURN_REASONS:
        return _fail("reason's allowed values are {!r}.".format(reason.get("enum")),
                     "Use REASONS: too_small, too_big, damaged, other.")
    if sorted(schema.get("required") or []) != ["item", "order_id", "reason"]:
        return _fail("required is {!r}.".format(schema.get("required")),
                     'All three fields are required: ["order_id", "item", "reason"].')
    if schema.get("additionalProperties") is not False:
        return _fail("additionalProperties is {!r}.".format(schema.get("additionalProperties")),
                     "Set it to False - no fields other than the three are allowed.")
    return _ok("An object with three properties, a reason limited to four values, all "
               "three required and nothing else allowed. That is the shape your code "
               "will be able to read without guessing.")


@task("format", title="Ask for output that fits the schema", lab="structured_output",
      needs=("first", "data"))
def _task25_2(first: Any, data: Any, **_: Any) -> Tuple[bool, str, str]:
    lab = getattr(first, "_lab", {}) or {}
    if getattr(first, "simulated", False) and lab.get("scenario") != "format.json_schema":
        return _fail("The request did not ask for structured output.",
                     'Pass output_config={"effort": "low", "format": {"type": "json_schema", '
                     '"schema": SCHEMA}}.')
    if not isinstance(data, dict):
        return _fail("`data` is a {}, not a dictionary.".format(type(data).__name__),
                     "Parse the text block with json.loads(...).")
    if data != {"order_id": "ORD-12345678", "item": "jacket", "reason": "too_small"}:
        return _fail("The first email gave {!r}.".format(data))
    return _ok("JSON that matches the schema, parsed with json.loads - three correct "
               "fields: the order, the jacket, too_small. With the format set, it parses "
               "every time.")


@task("valid_not_correct", title="Valid is not the same as correct",
      lab="structured_output", needs=("second", "looks_like_order"))
def _task25_3(second: Any, looks_like_order: Any, **_: Any) -> Tuple[bool, str, str]:
    if not isinstance(second, dict) or second.get("order_id") is None:
        return _fail("`second` should be the parsed result for the second email.")
    if not callable(looks_like_order):
        return _fail("`looks_like_order` is not a function.")
    try:
        verdicts = (looks_like_order("ORD-12345678"), looks_like_order(second["order_id"]))
    except Exception as exc:                                # noqa: BLE001
        return _fail("looks_like_order() raised {}: {}".format(type(exc).__name__, exc))
    if verdicts != (True, False):
        return _fail("looks_like_order() said {} for a real ID and {} for {!r}.".format(
            verdicts[0], verdicts[1], second["order_id"]),
            'A real order ID starts with "ORD-".')
    return _ok("The second email never names an order, and the schema still got its "
               "string: {!r}. Well formed, and useless - stored, it would be a return "
               "request pointing at no order. A schema makes the answer parse; your code "
               "still checks the values.".format(second["order_id"]))


# -- Validation tasks --------------------------------------------------------


@task("nullable", title="Allow null instead of a made-up value", lab="validation",
      needs=("schema", "results"))
def _task26_1(schema: Any, results: Any, **_: Any) -> Tuple[bool, str, str]:
    spec = ((schema or {}).get("properties") or {}).get("order_id") or {}
    types_ = spec.get("type")
    if not (isinstance(types_, list) and "null" in types_ and "string" in types_):
        return _fail("order_id's type is {!r}.".format(types_),
                     'Allow a string or nothing: ["string", "null"].')
    if not isinstance(results, list) or len(results) < 2:
        return _fail("Expected the results for the two emails.")
    if results[1].get("order_id") is not None:
        return _fail("The email without an order number gave order_id {!r}.".format(
            results[1].get("order_id")),
            "Send INSTRUCTION with the email: it says to use null and never invent one.")
    return _ok("The email without an order number now gives null, not a made-up word. "
               "That handles the honest case: missing is said out loud, and nothing "
               "invented gets stored.")


@task("validate", title="Check the values in code", lab="validation",
      needs=("validate",))
def _task26_2(validate: Any, **_: Any) -> Tuple[bool, str, str]:
    if not callable(validate):
        return _fail("`validate` is not a function.")
    base = {"item": "jacket", "reason": "too_small"}
    try:
        good = validate(dict(base, order_id="ORD-12345678"))
        none = validate(dict(base, order_id=None))
        bad = validate(dict(base, order_id="ORD-7788"))
    except Exception as exc:                                # noqa: BLE001
        return _fail("validate() raised {}: {}".format(type(exc).__name__, exc),
                     r'The pattern is r"ORD-\d{8}" - ORD- and exactly eight digits.')
    if good is not None or none is not None:
        return _fail("validate() rejected a valid order ID or a null.",
                     "Return None when the data is fine - null is allowed.")
    if not (isinstance(bad, str) and "ORD-7788" in bad):
        return _fail("validate() let ORD-7788 through, or its error does not name it.",
                     r'Use the exact pattern r"ORD-\d{8}", and return the message that '
                     "says which value failed.")
    return _ok("A typo the schema cannot see - four digits instead of eight - is caught "
               "by code, with an exact message: which field, which value. That message "
               "is what the retry sends back.")


@task("retry", title="Retry with the exact error", lab="validation",
      needs=("results", "rejected"))
def _task26_3(results: Any, rejected: Any, **_: Any) -> Tuple[bool, str, str]:
    if not isinstance(results, list) or len(results) != 3:
        return _fail("Expected the results for the three emails.")
    if results[2] is None:
        return _fail("The third email gave up after every attempt.",
                     "Send Claude its own answer as the assistant turn, then the exact "
                     "error as the user turn, and try again.")
    if results[2].get("order_id") is not None:
        return _fail("The third email kept order_id {!r}.".format(results[2]["order_id"]))
    if not isinstance(rejected, list) or len(rejected) != 1:
        return _fail("Expected exactly one rejected attempt, got {!r}.".format(rejected),
                     "Validate every attempt; the third email is rejected once, then "
                     "fixed.")
    return _ok("Attempt one returned the four-digit number from the email; your validator "
               "rejected it and sent the answer and the error back. Attempt two returned "
               "null. Sending the exact error turns a retry from a coin flip into a "
               "correction - bounded at three attempts.")


# -- BUILD: the extraction module --------------------------------------------


BUILD_EXPECTED = (
    ("001.txt", "ORD-12345678", "ok"),
    ("002.txt", None, "needs review"),
    ("003.txt", "ORD-87654321", "ok"),
    ("004.txt", None, "needs review"),
    ("005.txt", "ORD-11112222", "ok"),
)


@task("read_folder", title="Run the extraction over the folder", lab="extract_returns",
      needs=("requests",))
def _task28_1(requests: Any, **_: Any) -> Tuple[bool, str, str]:
    if not isinstance(requests, list) or len(requests) != 5:
        return _fail("Expected five records, one per email, got {}.".format(
            len(requests) if isinstance(requests, list) else type(requests).__name__),
            'Walk the folder with Path("emails").glob("*.txt"), in sorted order.')
    for row, (name, order_id, _) in zip(requests, BUILD_EXPECTED):
        if not isinstance(row, dict) or row.get("email") != name:
            return _fail("The records are not in file order, or miss the file name.",
                         "sorted(...) the paths, and set data[\"email\"] = path.name.")
        if row.get("order_id") != order_id:
            return _fail("{} gave order_id {!r}.".format(name, row.get("order_id")),
                         "Pass the email's text to extract(): path.read_text().")
    return _ok("Five files, five records, every one through the same path you built: the "
               "schema, the validation and the retry. The four-digit order number in "
               "004.txt was rejected once and came back null.")


@task("status", title="A status for every record", lab="extract_returns",
      needs=("requests",))
def _task28_2(requests: Any, **_: Any) -> Tuple[bool, str, str]:
    if not isinstance(requests, list) or len(requests) != 5:
        return _fail("Run Task 1 first - it builds the five records.")
    for row, (name, _, status) in zip(requests, BUILD_EXPECTED):
        if row.get("status") is Ellipsis:
            return _fail("A status is still ... .")
        if row.get("status") != status:
            return _fail("{} has status {!r}, expected {!r}.".format(
                name, row.get("status"), status),
                'The status is "ok" when there is an order ID, "needs review" when it is '
                "null.")
    return _ok("Three complete, two for a person: the email with no order number, and "
               "the one whose number was wrong and became null after the retry. A missing "
               "value is information - it tells the team which emails a person must read.")


@task("save", title="Save the batch and count the reviews", lab="extract_returns",
      needs=("saved", "need_review"))
def _task28_3(saved: Any, need_review: Any, **_: Any) -> Tuple[bool, str, str]:
    if not isinstance(saved, list) or len(saved) != 5:
        return _fail("return_requests.json does not hold the five records.",
                     "Write json.dumps(requests, indent=2) to the file.")
    if need_review != 2:
        return _fail("need_review is {!r}, expected 2.".format(need_review),
                     'Count the records whose status is not "ok".')
    return _ok("All five saved to return_requests.json, ready for the order system or a "
               "person, and two flagged for review. Pushing Claude for an order number it "
               "does not have would produce one - and it would be wrong.")


@task("scope", title="A system prompt with role and scope", lab="system_prompt",
      needs=("system", "poem_reply"))
def _task_rules_1(system: Any, poem_reply: Any, **_: Any) -> Tuple[bool, str, str]:
    if not isinstance(system, str) or "..." in system or system is Ellipsis:
        return _fail("SYSTEM is not finished yet.",
                     "Replace each ... with the constant named beside it.")
    if not re.search(r"help only with|only help with", system, re.I):
        return _fail("SYSTEM does not set a scope.",
                     "Add SCOPE_RULE - it says what ShopAssist helps with.")
    if not re.search(r"politely|decline|cannot help", system, re.I):
        return _fail("SYSTEM does not say what to do outside the scope.",
                     "Add DECLINE_RULE - otherwise Claude has no instruction for the rest.")
    if not isinstance(poem_reply, str) or not re.search(r"only help|can only|cannot help",
                                                        poem_reply, re.I):
        return _fail("The poem request was not declined.",
                     "Run the cell again after SYSTEM has both lines.")
    return _ok("Claude declined the poem and offered help with orders instead. Role, "
               "tone, scope and the style of an answer: this is what a system prompt "
               "does well.")


@task("gate", title="A rule that must hold lives in code", lab="system_prompt",
      needs=("sent_before", "sent_after", "say"))
def _task_rules_2(sent_before: Any, sent_after: Any, say: Any, **_: Any) -> Tuple[bool, str, str]:
    if not callable(say):
        return _fail("`say` is not a function.")
    if sent_after != sent_before:
        return _fail("{} request(s) went to Claude for the refund message.".format(
            sent_after - sent_before),
            'The gate has to answer by itself and return BEFORE ask() is called - check '
            '"refund" in the text and `not verified`.')
    return _ok("No request left your application for that turn: your code answered alone. "
               "A prompt is a request, and a customer who insists may talk the model into "
               "anything. A rule that must always hold belongs in code.")


@task("verified", title="Verified, the same request passes", lab="system_prompt",
      needs=("verified", "sent_before", "sent_after", "refund_reply"))
def _task_rules_3(verified: Any, sent_before: Any, sent_after: Any, refund_reply: Any,
                  **_: Any) -> Tuple[bool, str, str]:
    if verified is not True:
        return _fail("`verified` is {!r}.".format(verified),
                     "Set it to True - the customer has confirmed the email.")
    if sent_after - sent_before != 1:
        return _fail("{} request(s) went to Claude, expected exactly 1.".format(
            sent_after - sent_before),
            "Once verified, the refund message passes the gate and goes to Claude.")
    if not isinstance(refund_reply, str) or not re.search(r"30-day|30 days", refund_reply):
        return _fail("The reply does not answer about the 30-day window.")
    return _ok("The same words, a different outcome: verified, the message passed the gate "
               "and Claude answered about the 30-day window. Role, tone and scope go in the "
               "system prompt; every rule that must always hold goes in code, before the "
               "request leaves.")


@task("model_per_step", title="A model for each step", lab="model_settings",
      needs=("summary", "answer"))
def _task_ms_1(summary: Any, answer: Any, **_: Any) -> Tuple[bool, str, str]:
    for name, resp, family, why in (
            ("summary", summary, "haiku", "the log note is simple and runs after every "
                                          "conversation - Haiku fits"),
            ("answer", answer, "sonnet", "the conversation needs policy judgment - Sonnet "
                                         "runs it")):
        model = str(getattr(resp, "model", ""))
        if family not in model.lower():
            return _fail("The {} step ran on {!r}.".format(name, model),
                         "Use the {} constant: {}.".format(
                             "MODEL_SUMMARY" if family == "haiku" else "MODEL_CHAT", why))
    return _ok("Two steps, two families: the two-line log note on Haiku, fast and cheap; "
               "the customer conversation on Sonnet. Each model ID appears once, as a "
               "constant, so a new version is a one-line change.")


@task("effort_per_step", title="An effort level for each step", lab="model_settings",
      needs=("quick", "tangled"))
def _task_ms_2(quick: Any, tangled: Any, **_: Any) -> Tuple[bool, str, str]:
    sim = getattr(quick, "simulated", False)
    if sim and _effort_of(quick) != "low":
        return _fail("The shipping question ran at effort {!r}.".format(_effort_of(quick)),
                     'Confirming that an order shipped is simple: "low", so the answer '
                     "comes back fast.")
    if sim and _effort_of(tangled) != "high":
        return _fail("The tangled case ran at effort {!r}.".format(_effort_of(tangled)),
                     'Three overlapping orders and a partial refund deserve more work: '
                     '"high".')
    return _ok("Low effort for a quick confirmation, high effort for the tangled case. "
               "You do not control Claude's reasoning directly - you tell it how much "
               "work each step deserves, and like the model, you choose it per step.")


@task("check_before_send", title="Check stop_reason before the customer sees it",
      lab="model_settings", needs=("first", "delivered"))
def _task_ms_3(first: Any, delivered: Any, **_: Any) -> Tuple[bool, str, str]:
    if getattr(first, "stop_reason", None) != "max_tokens":
        return _fail("The first reply was not cut, stop_reason is {!r}.".format(
            getattr(first, "stop_reason", None)),
            "Keep max_tokens=300 for the first try - that is the ceiling the store set.")
    if delivered is None or delivered is Ellipsis or not isinstance(delivered, str):
        return _fail("deliver() returned nothing to show.",
                     'When stop_reason is "max_tokens", retry with more room and return '
                     "the new text.")
    if delivered == _text_of(first):
        return _fail("deliver() returned the cut reply.",
                     'Fill in both blanks in deliver(): compare stop_reason with '
                     '"max_tokens", and retry with max_tokens=1000.')
    if not delivered.rstrip().endswith((".", "!", "?")):
        return _fail("The delivered reply still stops mid-sentence.",
                     "Give the retry enough room: max_tokens=1000.")
    return _ok("The first reply hit the 300-token ceiling halfway through the second "
               "order. Your code read stop_reason first, saw max_tokens, and retried with "
               "more room - the customer only ever saw the complete answer.")


# -- chat page and server ----------------------------------------------------


def _has_post(app: Any, path: str) -> bool:
    return any(getattr(r, "path", None) == path and "POST" in (getattr(r, "methods", None)
                                                                or ())
               for r in getattr(app, "routes", []))


@task("endpoint", title="The server's chat endpoint", lab="chat_server",
      needs=("app", "history", "page"))
def _task_web_1(app: Any, history: Any, page: Any, **_: Any) -> Tuple[bool, str, str]:
    if not _has_post(app, "/chat"):
        return _fail("The app has no POST /chat endpoint.",
                     'Decorate chat() with @app.post("/chat") - the address the page '
                     "sends to.")
    if not isinstance(page, dict) or not isinstance(page.get("reply"), str) or \
            not isinstance(page.get("input_tokens"), int):
        return _fail("The page got {!r}.".format(page),
                     "Return the reply and the input tokens as a dict - it travels as "
                     "JSON.")
    if not isinstance(history, list) or len(history) != 2:
        return _fail("After one message the server holds {} turn(s), expected 2.".format(
            len(history) if isinstance(history, list) else "?"))
    if history[0].get("content") != "I bought a jacket twenty days ago.":
        return _fail("The first turn in the history is {!r}.".format(
            history[0].get("content")),
            "Store the customer's text: body.text.")
    if history[1].get("role") != "assistant" or history[1].get("content") != page["reply"]:
        return _fail("The second turn is not Claude's reply: {!r}.".format(history[1]),
                     'Add the reply to the history as role "assistant": reply.')
    return _ok("One POST to /chat: the server added the text to its history, sent the whole "
               "history with the system line to Claude, stored the reply, and sent back "
               "JSON with the reply and the input tokens.")


@task("server_memory", title="The server keeps the conversation", lab="chat_server",
      needs=("first", "second", "history"))
def _task_web_2(first: Any, second: Any, history: Any, **_: Any) -> Tuple[bool, str, str]:
    if not all(isinstance(x, dict) and "reply" in x for x in (first, second)):
        return _fail("first and second should be what the page received.")
    reply = second["reply"]
    if not re.search(r"jacket", reply, re.I) or re.search(r"which item", reply, re.I):
        return _fail("The second reply does not know about the jacket: {!r}.".format(reply),
                     "The page sends only the new question; the server's history must "
                     "hold the first message. Run the cell as it is.")
    if not isinstance(history, list) or len(history) != 4:
        return _fail("The server holds {} turn(s) after two messages, expected 4.".format(
            len(history) if isinstance(history, list) else "?"))
    if not second.get("input_tokens", 0) > first.get("input_tokens", 0):
        return _fail("The input tokens did not grow: {} -> {}.".format(
            first.get("input_tokens"), second.get("input_tokens")))
    return _ok("The page sent only \"Can I still return it?\" - the server kept the history, "
               "so Claude knew about the jacket. And the input tokens on the page grew, "
               "{} -> {}, just like in the script.".format(first["input_tokens"],
                                                          second["input_tokens"]))


@task("server_side", title="Only the server talks to Claude", lab="chat_server",
      needs=("sent", "page", "history"))
def _task_web_3(sent: Any, page: Any, history: Any, **_: Any) -> Tuple[bool, str, str]:
    if not isinstance(sent, dict) or _has_blank(sent):
        return _fail("The edited page's request still has a ... in it.",
                     "Fill in the rule the edited page tries to set.")
    forged = [v for k, v in sent.items() if k != "text" and isinstance(v, str) and v]
    if not forged:
        return _fail("The request only carries text - nothing tries to change the rules.",
                     'Keep the "system" field: that is what an edited page would send.')
    if not isinstance(page, dict) or set(page) != {"reply", "input_tokens"}:
        return _fail("The page received the keys {!r}.".format(
            sorted(page) if isinstance(page, dict) else page),
            "The server returns only the reply and the input tokens.")
    stored = json.dumps(history)
    if any(f in stored for f in forged):
        return _fail("The edited page's rule ended up in the conversation.")
    if not re.search(r"past|no longer|cannot|can't|30-day", page["reply"], re.I):
        return _fail("The reply does not follow the 30-day rule: {!r}.".format(page["reply"]))
    return _ok("The edited page sent its own rule, and the server dropped it: ChatIn "
               "declares only text. Claude answered under the server's 30-day rule, and the "
               "page got back only the reply and the token count - no key, no system line, "
               "no history. The browser talks to your server, and only your server talks "
               "to Claude.")


# -- confidence calibration and human review ---------------------------------


@task("calibrate", title="Calibrate field-level confidence", lab="human_review",
      needs=("extraction", "cutoffs"))
def _task_review_1(extraction: Any, cutoffs: Any, **_: Any) -> Tuple[bool, str, str]:
    if not isinstance(extraction, dict) or not all(
            isinstance(extraction.get(k), int) for k in ("order_id_confidence",
                                                         "reason_confidence")):
        return _fail("The extraction has no field-level confidence scores: {!r}.".format(
            extraction))
    if not isinstance(cutoffs, dict) or _has_blank(cutoffs):
        return _fail("CUTOFFS is not finished: {!r}.".format(cutoffs),
                     "Replace each ... in calibrate() with the value named beside it.")
    if cutoffs != {"order_id": 7, "reason": 9}:
        return _fail("The cut-offs are {!r}, expected order_id 7 and reason 9.".format(
            cutoffs),
            "Walk down from 10: take the rows with exactly that score, stop at the "
            "first score whose accuracy is below the target, and keep the last score "
            "that held.")
    return _ok("Claude scored the order number {} and the vague reason {}. On the labeled "
               "set, order numbers hold from a score of 7, while reasons scored 7 are right "
               "only about four times in five - their cut-off is 9. Setting each field's "
               "cut-off from measured results is calibration.".format(
                   extraction["order_id_confidence"], extraction["reason_confidence"]))


@task("per_group", title="Accuracy per document type and field", lab="human_review",
      needs=("by_type", "invoice_fields", "automate"))
def _task_review_2(by_type: Any, invoice_fields: Any, automate: Any,
                   **_: Any) -> Tuple[bool, str, str]:
    if not isinstance(by_type, dict) or _has_blank(by_type) or \
            set(by_type) != {"return_form", "email", "invoice"}:
        return _fail("by_type is {!r}.".format(by_type),
                     "One accuracy per document type: the rows whose doc_type is t.")
    if not abs(by_type["invoice"] - 0.8) < 0.001:
        return _fail("Invoices come out at {:.0%}, expected 80%.".format(by_type["invoice"]),
                     'Filter on r["doc_type"] == t.')
    if not isinstance(invoice_fields, dict) or _has_blank(invoice_fields) or \
            min(invoice_fields, key=invoice_fields.get) != "refund_amount":
        return _fail("invoice_fields is {!r}.".format(invoice_fields),
                     'Keep only invoices, then one accuracy per field: r["field"] == f.')
    if automate != ["return_form", "email"]:
        return _fail("AUTOMATE is {!r}, expected the return forms and the emails.".format(
            automate), "Automate only the groups that reach TARGET.")
    return _ok("97% overall hid an 80% group: invoices, one document in twenty, and on "
               "them the refund amount - the field that sends money - is right only {:.0%} "
               "of the time. Return forms and emails hold up and can be automated; "
               "invoices stay under review.".format(invoice_fields["refund_amount"]))


@task("stratified", title="Keep sampling every group", lab="human_review",
      needs=("sample", "items", "found"))
def _task_review_3(sample: Any, items: Any, found: Any, **_: Any) -> Tuple[bool, str, str]:
    if not isinstance(sample, list) or not sample:
        return _fail("The sample is empty.")
    groups: Dict[Tuple[str, str], int] = {}
    for item in items:
        key = (item["doc_type"], item["level"])
        groups[key] = groups.get(key, 0) + 1
    picked: Dict[Tuple[str, str], int] = {}
    for item in sample:
        key = (item.get("doc_type"), item.get("level"))
        picked[key] = picked.get(key, 0) + 1
    for key, size in sorted(groups.items()):
        want = min(20, size)
        if picked.get(key, 0) != want:
            return _fail("Group {} got {} picks, expected {}.".format(
                key, picked.get(key, 0), want),
                "Group by document type AND confidence level, and take "
                "min(per_group, len(members)) from every group.")
    if len({item["id"] for item in sample}) != len(sample):
        return _fail("The sample picks some items twice.")
    caught = (found or {}).get(("invoice", "high"), 0) if isinstance(found, dict) else 0
    return _ok("Twenty from every group, {} items in all - the high-confidence invoices "
               "included, which a plain random pick barely touches. The sample caught {} "
               "wrong date(s) there: a new supplier writes the day before the month. Claude "
               "scored those dates high, so no cut-off could catch them.".format(
                   len(sample), caught))


# -- tool call ---------------------------------------------------------------


def _blocks_of(content: Any) -> List[Any]:
    return [] if isinstance(content, str) else list(content or [])


def _tool_uses(message: Any) -> List[Any]:
    return [b for b in _blocks_of(getattr(message, "content", None))
            if _block_type(b) == "tool_use"]


@task("define_tool", title="Describe the tool and send it", lab="tool_call",
      needs=("tools", "response"))
def _task_tc_1(tools: Any, response: Any, **_: Any) -> Tuple[bool, str, str]:
    if not isinstance(tools, list) or not tools or not isinstance(tools[0], dict):
        return _fail("TOOLS is not a list holding one tool definition.")
    tool = tools[0]
    if tool.get("name") != "lookup_order":
        return _fail("The tool is named {!r}.".format(tool.get("name")),
                     'Name it "lookup_order": the name Claude asks for, and the name of '
                     "your function.")
    schema = tool.get("input_schema") or {}
    if schema.get("required") != ["order_id"]:
        return _fail("input_schema requires {!r}.".format(schema.get("required")),
                     'List the one field the tool needs: "required": ["order_id"].')
    if getattr(response, "stop_reason", None) != "tool_use":
        return _fail("stop_reason is {!r}, not 'tool_use'.".format(
            getattr(response, "stop_reason", None)),
            "Pass tools=TOOLS to the request - without it Claude does not know the tool "
            "exists, and answers from the policy alone.")
    uses = _tool_uses(response)
    if not uses or _block_get(uses[0], "name") != "lookup_order":
        return _fail("The reply holds no tool_use block for lookup_order.")
    return _ok("stop_reason is tool_use: Claude did not answer. It asked for lookup_order "
               "with the input it chose, {}. Nothing has run yet - Claude never runs a "
               "tool, your code does.".format(json.dumps(_block_get(uses[0], "input"))))


@task("run_tool", title="Run the function Claude asked for", lab="tool_call",
      needs=("result", "messages"))
def _task_tc_2(result: Any, messages: Any, **_: Any) -> Tuple[bool, str, str]:
    if not isinstance(result, dict) or result.get("item") != "jacket":
        return _fail("result is {!r}, not the jacket's order record.".format(result),
                     "Call lookup_order(**block.input) - the input Claude chose.")
    if not isinstance(messages, list) or len(messages) != 3:
        return _fail("messages holds {} turn(s), expected 3: the question, Claude's "
                     "reply and your tool result.".format(
                         len(messages) if isinstance(messages, list) else "?"))
    assistant, user = messages[1], messages[2]
    if _role_of(assistant) != "assistant":
        return _fail("The second turn has role {!r}.".format(_role_of(assistant)),
                     'Claude\'s reply goes back as role "assistant".')
    body = assistant.get("content") if isinstance(assistant, dict) else None
    uses = [b for b in _blocks_of(body if body is not Ellipsis else None)
            if _block_type(b) == "tool_use"]
    if not uses:
        return _fail("The assistant turn does not hold Claude's tool_use block.",
                     "Send Claude's reply back exactly as it came: response.content.")
    results = [b for b in _blocks_of(user.get("content") if isinstance(user, dict) else None)
               if _block_type(b) == "tool_result"]
    if not results:
        return _fail("The third turn holds no tool_result block.")
    ref = _block_get(results[0], "tool_use_id")
    if ref != _block_get(uses[0], "id"):
        return _fail("tool_use_id is {}, not the id of Claude's tool_use block.".format(
            "still ..." if ref is Ellipsis else repr(ref)),
                     "Use block.id - it tells Claude which request this result answers.")
    content = _block_get(results[0], "content")
    if not isinstance(content, str):
        return _fail("The tool_result content is a {}, not text.".format(
            type(content).__name__), "Send json.dumps(result).")
    return _ok("Your code ran lookup_order and got the record: a {}, {} {} days ago. Two "
               "turns went into the conversation: Claude's own reply, exactly as it came, "
               "and a tool_result that carries the id of the tool_use block and the result "
               "as JSON.".format(result.get("item"), result.get("status"),
                                 result.get("delivered_days_ago")))


@task("round_trip", title="Send the result back for the answer", lab="tool_call",
      needs=("final",))
def _task_tc_3(final: Any, **_: Any) -> Tuple[bool, str, str]:
    stop = getattr(final, "stop_reason", None)
    if stop == "tool_use":
        return _fail("Claude asked for a tool again instead of answering.",
                     "Send messages=messages - the list that now holds your tool_result.")
    if stop != "end_turn":
        return _fail("stop_reason is {!r}, expected 'end_turn'.".format(stop))
    text = _text_of(final)
    if not re.search(r"\b(?:20|twenty)\b", text, re.I):
        return _fail("The answer does not use the order's data: {!r}.".format(text[:120]),
                     "Run Task 2 first, then this cell - the conversation must hold the "
                     "tool result.")
    return _ok("stop_reason is end_turn, and the answer uses the real data: delivered 20 "
               "days ago, ten days left. A tool call is a round trip - Claude asks with "
               "tool_use, your code answers with tool_result, and only then does Claude "
               "answer the customer.")


# -- agentic loop ------------------------------------------------------------


def _tool_names_in(messages: Any) -> List[str]:
    names: List[str] = []
    for msg in messages if isinstance(messages, list) else []:
        if _role_of(msg) == "assistant":
            body = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
            names += [str(_block_get(b, "name")) for b in _blocks_of(body)
                      if _block_type(b) == "tool_use"]
    return names


@task("tool_map", title="Map each tool name to its function", lab="agentic_loop",
      needs=("tool_functions", "output"))
def _task_loop_1(tool_functions: Any, output: Any, **_: Any) -> Tuple[bool, str, str]:
    if not isinstance(tool_functions, dict) or \
            set(tool_functions) != {"lookup_order", "start_return"}:
        return _fail("TOOL_FUNCTIONS has the keys {!r}.".format(
            sorted(tool_functions) if isinstance(tool_functions, dict) else tool_functions),
            "One entry per tool, keyed by the tool's name: lookup_order and start_return.")
    for name, fn in tool_functions.items():
        if not callable(fn) or getattr(fn, "__name__", name) != name:
            return _fail("TOOL_FUNCTIONS[{!r}] is {!r}.".format(name, fn),
                         "Each name maps to the function of the same name - no quotes, "
                         "no parentheses: {}.".format(name))
    if not isinstance(output, dict) or output.get("item") != "jacket":
        return _fail("The lookup returned {!r}.".format(output),
                     "Look the function up by the name the block carries, then call it "
                     "with its input: TOOL_FUNCTIONS[name](**args).")
    return _ok("One dictionary, and your code can run whichever tool Claude asks for: the "
               "name from the tool_use block picks the function, its input becomes the "
               "arguments.")


@task("every_result", title="A tool_result for every tool_use block", lab="agentic_loop",
      needs=("run_tools", "results"))
def _task_loop_2(run_tools: Any, results: Any, **_: Any) -> Tuple[bool, str, str]:
    if not callable(run_tools):
        return _fail("run_tools is not a function.")
    if not isinstance(results, list) or len(results) != 1:
        return _fail("run_tools returned {!r}, expected a list with one tool_result.".format(
            results), "Collect each tool_result in `results`, and return the list.")
    first = results[0]
    if not isinstance(first, dict) or first.get("type") != "tool_result":
        return _fail("The list holds {!r}, not a tool_result block.".format(first))
    if not isinstance(first.get("content"), str):
        return _fail("The tool_result content is not text.", "Send json.dumps(output).")
    # Two blocks in one reply: Claude may ask for more than one tool at a time.
    two = Message("msg_check", [
        ToolUseBlock("toolu_check_a", "lookup_order", {"order_id": "ORD-12345678"}),
        ToolUseBlock("toolu_check_b", "start_return", {"order_id": "ORD-12345678"})],
        "check", "tool_use", None, Usage(0, 0))
    buffer = sys.stdout
    try:
        sys.stdout = io.StringIO()
        both = run_tools(two)
    finally:
        sys.stdout = buffer
    ids = [r.get("tool_use_id") for r in both if isinstance(r, dict)] \
        if isinstance(both, list) else []
    if ids != ["toolu_check_a", "toolu_check_b"]:
        return _fail("For a reply with two tool_use blocks, run_tools gave the ids {!r}."
                     .format(ids),
                     "Run every tool_use block and take the id from each block: "
                     "block.id - and return the list after the loop, not inside it.")
    return _ok("One tool_result per tool_use block, each with the id of its block - and a "
               "reply with two blocks got two results. They all go back to Claude in one "
               "user message.")


@task("loop", title="Loop until Claude stops asking for tools", lab="agentic_loop",
      needs=("response", "messages", "turn"))
def _task_loop_3(response: Any, messages: Any, turn: Any, **_: Any) -> Tuple[bool, str, str]:
    stop = getattr(response, "stop_reason", None)
    if stop == "tool_use":
        return _fail("The loop ended while Claude was still asking for a tool.",
                     'Leave the loop only when stop_reason != "tool_use".')
    if stop != "end_turn":
        return _fail("The last reply ended with stop_reason {!r}.".format(stop))
    names = _tool_names_in(messages)
    if names[:2] != ["lookup_order", "start_return"]:
        return _fail("The tools ran in the order {!r}.".format(names),
                     "Add Claude's reply and then the results to `messages` on every "
                     "turn, so Claude sees what the last tool returned.")
    if not re.search(r"RET-2048", _text_of(response)):
        return _fail("The final answer does not give the return label.")
    return _ok("{} turns: lookup_order, then start_return, then end_turn with the label. "
               "Claude decided which tool to call and in what order; your loop ran them, "
               "and your loop decided when to stop, by reading stop_reason.".format(turn))


# -- tool errors -------------------------------------------------------------


def _tool_results_in(messages: Any) -> List[Tuple[str, Any, bool]]:
    """(tool name, parsed result, is_error flag) for every tool call in a conversation."""
    names: Dict[str, str] = {}
    out: List[Tuple[str, Any, bool]] = []
    for msg in messages if isinstance(messages, list) else []:
        body = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
        for b in _blocks_of(body):
            if _block_type(b) == "tool_use":
                names[str(_block_get(b, "id"))] = str(_block_get(b, "name"))
            elif _block_type(b) == "tool_result":
                out.append((names.get(str(_block_get(b, "tool_use_id")), "?"),
                            _parse_result(_block_get(b, "content")),
                            _block_get(b, "is_error", None) is True))
    return out


@task("not_found", title="Not found is data, not an error", lab="tool_errors",
      needs=("lookup_order",))
def _task_err_1(lookup_order: Any, **_: Any) -> Tuple[bool, str, str]:
    try:
        missing = lookup_order("ORD-99999999")
        found = lookup_order("ORD-87654321")
    except Exception as exc:
        return _fail("lookup_order raised {}: {}.".format(type(exc).__name__, exc),
                     "A tool never raises for a missing order: it returns an answer.")
    if _has_blank(missing) or _has_blank(found):
        return _fail("lookup_order still returns a ... value.",
                     "Replace each ... with the value named beside it.")
    if not isinstance(missing, dict) or missing.get("found") is not False:
        return _fail("For an unknown order, lookup_order returned {!r}.".format(missing),
                     'An order that does not exist is a normal, empty answer: '
                     '"found": False and a short message.')
    if missing.get("isError"):
        return _fail("The unknown order comes back flagged as an error.",
                     "Not found is data: no isError flag.")
    if not isinstance(found, dict) or found.get("found") is not True or \
            not isinstance(found.get("order"), dict):
        return _fail("For the lamp, lookup_order returned {!r}.".format(found),
                     'An existing order returns "found": True and the order.')
    return _ok("An unknown order is not an error - it is a normal answer, just an empty "
               "one: found False and a short message. The lamp comes back found, with its "
               "order record.")


@task("business_error", title="A structured error when the policy blocks it",
      lab="tool_errors", needs=("start_return",))
def _task_err_2(start_return: Any, **_: Any) -> Tuple[bool, str, str]:
    try:
        refused = start_return("ORD-87654321")
        allowed = start_return("ORD-12345678")
    except Exception as exc:
        return _fail("start_return raised {}: {}.".format(type(exc).__name__, exc),
                     "Return the error as data - an exception would crash the loop.")
    if not isinstance(refused, dict):
        return _fail("For the lamp, start_return returned {!r}.".format(refused))
    if _has_blank(refused):
        return _fail("The error still holds a ... value.",
                     "Replace each ... with the value named beside it.")
    for key, want, hint in (
            ("isError", True, 'Set "isError": True - the call failed.'),
            ("errorCategory", "business", 'The policy blocked it: category "business".'),
            ("isRetryable", False, 'Trying again cannot change the delivery date: '
                                   '"isRetryable": False.')):
        if refused.get(key) != want or type(refused.get(key)) is not type(want):
            return _fail("{} is {!r}.".format(key, refused.get(key)), hint)
    for key in ("customerMessage", "developerMessage"):
        if not isinstance(refused.get(key), str) or not refused[key].strip():
            return _fail("The error has no {}.".format(key))
    if not isinstance(allowed, dict) or "return_label" not in allowed:
        return _fail("The jacket, inside the window, did not get a return label.")
    return _ok("The lamp is refused with a structured error: isError, category business, "
               "not retryable, a message that is safe to show the customer and one for the "
               "logs. The jacket still gets its label.")


@task("is_error", title="Flag the tool_result, and let Claude explain", lab="tool_errors",
      needs=("cases",))
def _task_err_3(cases: Any, **_: Any) -> Tuple[bool, str, str]:
    if not isinstance(cases, list) or len(cases) != 2 or \
            not all(isinstance(c, tuple) and len(c) == 2 for c in cases):
        return _fail("Expected two (response, messages) pairs, one per case.",
                     "Run the cell as it is: handle() returns response, messages.")
    (unknown, unknown_msgs), (lamp, lamp_msgs) = cases
    if any(getattr(r, "stop_reason", None) == "tool_use" for r in (unknown, lamp)):
        return _fail("handle() returned while Claude was still asking for a tool.",
                     'Leave the loop only when stop_reason != "tool_use".')
    first = _tool_results_in(unknown_msgs)
    if any(flag for _, _, flag in first):
        return _fail("The unknown order went back to Claude flagged is_error.",
                     "Not found is data: is_error stays False for it.")
    if not re.search(r"check|couldn't find|could not find|no order", _text_of(unknown), re.I):
        return _fail("For the unknown order, the answer does not ask to check the number.")
    second = _tool_results_in(lamp_msgs)
    refusals = [(r, flag) for name, r, flag in second if name == "start_return"]
    if not refusals:
        return _fail("For the lamp, start_return never ran.")
    result, flag = refusals[-1]
    if not flag:
        return _fail("start_return's tool_result went back with is_error False.",
                     'Take the flag from the output: bool(output.get("isError")).')
    text = _text_of(lamp)
    if not re.search(r"30-day|30 day|return window", text, re.I):
        return _fail("The answer does not tell the customer the window has passed: {!r}"
                     .format(text[:120]))
    if isinstance(result, dict) and "delivered_days_ago" in text:
        return _fail("The developer message reached the customer.")
    return _ok("The unknown order: found False, and Claude asked the customer to check the "
               "number - no error, no crash. The lamp: start_return refused, the tool_result "
               "carried is_error, and Claude answered in the words of the customer message. "
               "The developer message never reached the customer.")


# -- BUILD: support agent ----------------------------------------------------


def _quiet(fn: Callable, *args: Any) -> Any:
    buffer = sys.stdout
    try:
        sys.stdout = io.StringIO()
        return fn(*args)
    finally:
        sys.stdout = buffer


@task("case_facts", title="Case facts, filled by the tools", lab="support_agent",
      needs=("case", "get_customer", "lookup_order", "escalate"))
def _task_agent_1(case: Any, get_customer: Any, lookup_order: Any, escalate: Any,
                  **_: Any) -> Tuple[bool, str, str]:
    if not isinstance(case, dict) or set(case) != {"customer_id", "order_id", "escalated"}:
        return _fail("case is {!r}.".format(case),
                     "Keep the three facts: customer_id, order_id and escalated.")
    saved = dict(case)
    try:
        case.update(customer_id=None, order_id=None, escalated=False)
        _quiet(get_customer, "nobody@example.com")
        _quiet(lookup_order, "ORD-00000000")
        if case != {"customer_id": None, "order_id": None, "escalated": False}:
            return _fail("A lookup that found nothing changed the case: {!r}.".format(case),
                         "Set a fact only after the tool found what it looked for.")
        _quiet(get_customer, "alex@example.com")
        if case["customer_id"] != "CUS-1001":
            return _fail("After get_customer, customer_id is {!r}.".format(case["customer_id"]),
                         'get_customer sets case["customer_id"] = customer_id.')
        _quiet(lookup_order, "ORD-87654321")
        if case["order_id"] != "ORD-87654321":
            return _fail("After lookup_order, order_id is {!r}.".format(case["order_id"]),
                         'lookup_order sets case["order_id"] = order_id.')
        _quiet(escalate, "check")
        if case["escalated"] is not True:
            return _fail("After escalate, escalated is {!r}.".format(case["escalated"]),
                         'escalate sets case["escalated"] = True.')
    finally:
        case.clear()
        case.update(saved)
    return _ok("The case lives in your code, and the tools fill it as they run: "
               "get_customer the customer ID, lookup_order the order ID, escalate the flag. "
               "A lookup that finds nothing leaves it as it was.")


@task("hook", title="A hook that checks every tool call", lab="support_agent",
      needs=("before_tool", "case"))
def _task_agent_2(before_tool: Any, case: Any, **_: Any) -> Tuple[bool, str, str]:
    if not callable(before_tool) or not isinstance(case, dict):
        return _fail("before_tool is not a function, or case is not the dictionary.")
    saved = dict(case)
    order = {"order_id": "ORD-87654321"}
    try:
        case.update(customer_id=None, order_id=None, escalated=False)
        if before_tool("get_customer", {"email": "alex@example.com"}) is not None:
            return _fail("The hook blocks get_customer for an unverified customer.",
                         "get_customer is the one tool that runs before verification - "
                         'the gate skips it: name != "get_customer".')
        blocked = before_tool("lookup_order", order)
        if not isinstance(blocked, dict) or blocked.get("errorCategory") != "permission":
            return _fail("Unverified, lookup_order got {!r}.".format(blocked),
                         'No tool except get_customer runs while customer_id is empty: '
                         'a "permission" error.')
        case["customer_id"] = "CUS-1001"
        if before_tool("lookup_order", order) is not None:
            return _fail("Verified, the hook still blocks lookup_order.")
        early = before_tool("start_return", order)
        if not isinstance(early, dict) or early.get("errorCategory") != "validation":
            return _fail("start_return before any lookup got {!r}.".format(early),
                         'start_return only for the order that was looked up: compare '
                         'args["order_id"] with case["order_id"].')
        case["order_id"] = "ORD-87654321"
        if before_tool("start_return", order) is not None:
            return _fail("After the lookup, the hook still blocks start_return.")
    finally:
        case.clear()
        case.update(saved)
    return _ok("The gate: nothing but get_customer runs for an unverified customer. The "
               "order rule: start_return only for the order that was looked up. A failed "
               "rule returns a structured error - and the tool never runs.")


@task("handoff", title="Run the loop through the hook", lab="support_agent",
      needs=("runs",))
def _task_agent_3(runs: Any, **_: Any) -> Tuple[bool, str, str]:
    if not isinstance(runs, list) or len(runs) != 2 or \
            not all(isinstance(r, tuple) and len(r) == 3 for r in runs):
        return _fail("Expected two runs, each (response, messages, case).",
                     "Run the cell as it is.")
    (anon, anon_msgs, anon_case), (alex, alex_msgs, alex_case) = runs
    if any(getattr(r, "stop_reason", None) == "tool_use" for r in (anon, alex)):
        return _fail("run() returned while Claude was still asking for a tool.")
    ran = _tool_results_in(anon_msgs)
    if anon_case != {"customer_id": None, "order_id": None, "escalated": False} or \
            any(not flag for _, _, flag in ran):
        return _fail("For the unverified customer, a tool ran: {!r}.".format(anon_case),
                     "Every call goes through before_tool first; the function runs only "
                     "when the hook returns None.")
    if not re.search(r"email", _text_of(anon), re.I):
        return _fail("For the unverified customer, the answer does not ask for the email.")
    names = [name for name, _, _ in _tool_results_in(alex_msgs)]
    if "escalate" not in names:
        return _fail("The verified case was never escalated: {!r}.".format(names),
                     "start_return refuses the lamp; the customer still needs help, so the "
                     "case goes to a person through escalate.")
    if alex_case != {"customer_id": "CUS-1001", "order_id": "ORD-87654321",
                     "escalated": True}:
        return _fail("The case at the end is {!r}.".format(alex_case))
    if not re.search(r"ESC-\d+", _text_of(alex)):
        return _fail("The answer does not give the ticket number.")
    return _ok("Unverified, the hook blocked the call and Claude asked for the email - "
               "nothing looked up, nothing started. Verified: {} - and the case ends with "
               "the customer, the order and escalated True. A prompt is advice; the hook "
               "is code, and it cannot be talked around.".format(" -> ".join(names)))


# -- coordinator and subagents ------------------------------------------------


def _received(received: Any, agent: str) -> List[Dict[str, Any]]:
    return [r for r in received if isinstance(r, dict) and r.get("agent") == agent] \
        if isinstance(received, list) else []


@task("one_loop", title="One function runs any agent", lab="coordinator",
      needs=("report", "received"))
def _task_coord_1(report: Any, received: Any, **_: Any) -> Tuple[bool, str, str]:
    runs = _received(received, "order")
    if not runs:
        return _fail("RECEIVED has no entry for the order agent.",
                     "Run the cell as it is - run_agent() records what each agent gets.")
    if runs[-1].get("tools") != ["lookup_order"]:
        return _fail("The order agent got the tools {!r}.".format(runs[-1].get("tools")))
    if not isinstance(report, str) or not re.search(r"lamp", report, re.I):
        return _fail("The order agent's report is {!r}.".format(report),
                     "run_agent() returns the text of the last reply.")
    return _ok("One function, one loop: the order agent ran with its own system prompt, "
               "its own tool and its own task, and returned a short report: {!r}. Every "
               "agent - the coordinator too - is this function with different "
               "arguments.".format(report))


@task("explicit_context", title="Pass the facts explicitly", lab="coordinator",
      needs=("received", "facts"))
def _task_coord_2(received: Any, facts: Any, **_: Any) -> Tuple[bool, str, str]:
    order, billing = _received(received, "order"), _received(received, "billing")
    if not order or not billing:
        return _fail("RECEIVED does not hold a run of both subagents.",
                     "Run the cell as it is: it calls order_agent, then billing_agent.")
    if order[-1].get("tools") != ["lookup_order"]:
        return _fail("The order agent got the tools {!r}.".format(order[-1].get("tools")),
                     "The order agent can only look up orders: ORDER_TOOLS.")
    if sorted(billing[-1].get("tools") or []) != ["list_charges", "refund_charge"]:
        return _fail("The billing agent got the tools {!r}.".format(
            billing[-1].get("tools")),
            "The billing agent lists and refunds charges: BILLING_TOOLS.")
    task_text = str(billing[-1].get("task", ""))
    if not isinstance(facts, str) or facts not in task_text:
        return _fail("The billing agent received: {!r} - not the order facts.".format(
            task_text[:140]),
            "A subagent starts with an empty conversation. Put the facts in its task: "
            "f\"Order facts: {facts}\\n...\".")
    return _ok("Each subagent got only the tools for its job, and the billing agent "
               "received the order facts in its task text - the only thing it sees. "
               "What the coordinator forgets to pass, the subagent never knows.")


@task("delegate", title="The coordinator delegates through agent tools",
      lab="coordinator", needs=("answer", "received"))
def _task_coord_3(answer: Any, received: Any, **_: Any) -> Tuple[bool, str, str]:
    order = [r.get("agent") for r in received if isinstance(r, dict)] \
        if isinstance(received, list) else []
    if order[:1] != ["coordinator"]:
        return _fail("The coordinator did not run first: {!r}.".format(order))
    coordinator = _received(received, "coordinator")[-1]
    if sorted(coordinator.get("tools") or []) != ["billing_agent", "order_agent"]:
        return _fail("The coordinator got the tools {!r}.".format(coordinator.get("tools")),
                     "The coordinator has no data tools - only the two agents: "
                     "COORDINATOR_TOOLS.")
    if order[1:] != ["order", "billing"]:
        return _fail("The subagents ran in the order {!r}.".format(order[1:]),
                     "Map the agent tools to their functions: order_agent and "
                     "billing_agent in FUNCTIONS.")
    if not isinstance(answer, str) or not re.search(r"refund", answer, re.I):
        return _fail("The answer does not report the refund: {!r}.".format(answer))
    return _ok("The coordinator delegated to the order agent, passed its report to the "
               "billing agent, and wrote the answer from the two reports. Delegating is a "
               "tool call - and each subagent started empty, with only the task text it "
               "was given.")


# ---------------------------------------------------------------------------
# Auto-install on import, so a lab notebook only needs `import shopassist_lab`.
# ---------------------------------------------------------------------------

install()
