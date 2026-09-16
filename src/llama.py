import json
import urllib.request


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"


def generate_response(prompt: str) -> str:
    """Generate a response using a locally hosted Llama model."""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request) as response:
        result = json.loads(response.read().decode("utf-8"))

    return result["response"]


if __name__ == "__main__":
    ticket = (
        "I was charged twice for my hotel reservation "
        "and nobody has responded for three days. Please fix this."
    )

    prompt = f"""
You are an enterprise customer support assistant.

Analyze the following customer ticket and provide:
1. The likely issue
2. The priority
3. A concise professional response to the customer

Customer ticket:
{ticket}
"""

    response = generate_response(prompt)

    print("Customer Ticket:")
    print(ticket)

    print("\nLlama Response:")
    print(response)