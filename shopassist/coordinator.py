import json
from fuelix import make_client

client = make_client()

MODEL = "claude-sonnet-5-5"
FUNCTIONS = {}


def make_tool(name, description, fields):
    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": fields,
            "required": list(fields),
        },
    }


def run_agent(name, system, tools, task):
    messages = [{"role": "user", "content": task}]
    while True:
        response = client.messages.create(
            model=MODEL, max_tokens=600,
            output_config={"effort": "low"},
            system=system, tools=tools, messages=messages)
        if response.stop_reason != "tool_use":
            break
        messages.append({"role": "assistant",
                         "content": response.content})
        results = []
        for block in response.content:
            if block.type == "tool_use":
                output = FUNCTIONS[block.name](**block.input)
                print(f"  [{name}] {block.name}({block.input})")
                results.append({"type": "tool_result",
                                "tool_use_id": block.id,
                                "content": json.dumps(output)})
        messages.append({"role": "user", "content": results})
    for block in response.content:
        if block.type == "text":
            return block.text


ORDERS = {
    "ORD-87654321": {"item": "lamp", "status": "delivered",
                     "delivered_days_ago": 45, "price": 60},
}


def lookup_order(order_id):
    return ORDERS.get(order_id, {"error": "order not found"})


FUNCTIONS["lookup_order"] = lookup_order
ORDER_TOOLS = [
    make_tool("lookup_order", "Look up an order by its ID.",
              {"order_id": {"type": "string"}}),
]


def order_agent(order_id):
    return run_agent(
        "order",
        "You check orders. Report the item, status and "
        "delivery age in one sentence.",
        ORDER_TOOLS,
        f"Check order {order_id}.")


CHARGES = {
    "ORD-87654321": [{"amount": 60, "date": "day 1"},
                     {"amount": 60, "date": "day 1"}],
}


def list_charges(order_id):
    return CHARGES.get(order_id, [])


def refund_charge(order_id, amount):
    return {"refunded": amount, "order_id": order_id}


FUNCTIONS["list_charges"] = list_charges
FUNCTIONS["refund_charge"] = refund_charge
BILLING_TOOLS = [
    make_tool("list_charges", "List the charges of an order.",
              {"order_id": {"type": "string"}}),
    make_tool("refund_charge", "Refund one charge of an order.",
              {"order_id": {"type": "string"},
               "amount": {"type": "number"}}),
]


def billing_agent(order_id, facts):
    return run_agent(
        "billing",
        "You fix billing problems. Refund only a duplicate "
        "charge. Report what you did in one sentence.",
        BILLING_TOOLS,
        f"Order facts: {facts}\nCheck the charges of "
        f"{order_id} and fix any duplicate.")


FUNCTIONS["order_agent"] = order_agent
FUNCTIONS["billing_agent"] = billing_agent
COORDINATOR_TOOLS = [
    make_tool("order_agent", "Delegate: check an order.",
              {"order_id": {"type": "string"}}),
    make_tool("billing_agent", "Delegate: check and fix the "
              "charges of an order. Pass the order facts you "
              "already have.",
              {"order_id": {"type": "string"},
               "facts": {"type": "string"}}),
]

answer = run_agent(
    "coordinator",
    "You are ShopAssist's coordinator. You have no data tools. "
    "Delegate to the order agent and the billing agent, pass "
    "each one what it needs, and answer the customer in two "
    "sentences.",
    COORDINATOR_TOOLS,
    "I was charged twice for order ORD-87654321. Please fix it.")
print("ShopAssist:", answer)
