import json
from fuelix import make_client

client = make_client()

MODEL = "claude-sonnet-5-5"

SCHEMA = {
    "type": "object",
    "properties": {
        "order_id": {
            "type": ["string", "null"],
            "description": "null if there is no order number",
        },
        "item": {"type": "string"},
        "reason": {
            "type": "string",
            "enum": ["too_small", "too_big", "damaged", "other"],
        },
    },
    "required": ["order_id", "item", "reason"],
    "additionalProperties": False,
}


def ask_claude(messages):
    response = client.messages.create(
        model=MODEL,
        max_tokens=300,
        output_config={
            "effort": "low",
            "format": {"type": "json_schema", "schema": SCHEMA},
        },
        messages=messages,
    )
    for block in response.content:
        if block.type == "text":
            return block.text


def is_valid(order_id):
    if order_id is None:
        return True
    if order_id.startswith("ORD-") and len(order_id) == 12:
        return True
    return False


def extract(email):
    prompt = ("Extract the return request from this "
              "customer email.\n\n" + email)
    messages = [{"role": "user", "content": prompt}]
    for attempt in [1, 2, 3]:
        text = ask_claude(messages)
        data = json.loads(text)
        order_id = data["order_id"]
        if is_valid(order_id):
            return data
        error = order_id + " is not a valid order ID"
        print("  attempt", attempt, "rejected:", error)
        messages.append({"role": "assistant", "content": text})
        feedback = error + ". Fix it, or use null."
        messages.append({"role": "user", "content": feedback})
    return None


EMAILS = [
    "Hi, I want to return the blue jacket from order "
    "ORD-12345678. It arrived last week but it is too small. Alex",
    "Hello, the shoes I got are damaged. Please tell me how "
    "to send them back. Thanks, Sam",
    "The headphones from order ORD-7788 stopped working after "
    "two days. I want a return. Kim",
]

if __name__ == "__main__":
    for email in EMAILS:
        data = extract(email)
        print(data)
