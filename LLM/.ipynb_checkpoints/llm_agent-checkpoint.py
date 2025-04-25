import boto3
import json

# Create a Bedrock Runtime client in the desired AWS Region
bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")

def generate_prompt(viewed_asin_list, asin_to_description, asin_a, asin_b):
    history_descriptions = []
    for i, asin in enumerate(viewed_asin_list, 1):
        desc = asin_to_description.get(asin, "No description available.")
        history_descriptions.append(f"{i}. {desc}")

    history_text = "\n".join(history_descriptions)
    desc_a = asin_to_description.get(asin_a, "No description available.")
    desc_b = asin_to_description.get(asin_b, "No description available.")

    prompt = f"""You are a smart recommendation assistant.

Here is the customer's recent browsing history:
{history_text}

Now, we want to recommend **one** of the following two items:

{asin_a}: {desc_a}

{asin_b}: {desc_b}

Based on the customer's interest, which item would you recommend? Respond in the format:

Recommendation: [{asin_a} or {asin_b}]
Reason: [your explanation]
"""
    return prompt

def call_claude(prompt):
    model_id = "anthropic.claude-instant-v1"  # Replace with your authorized model ID
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 150,
        "temperature": 0.3
    }

    response = bedrock_client.invoke_model(
        modelId=model_id,
        body=json.dumps(payload),
        contentType="application/json",
        accept="application/json"
    )

    response_body = json.loads(response["body"].read())
    return response_body["content"][0]["text"]

def call_llama(prompt):
    model_id = "meta.llama3-70b-instruct-v1:0"  # Replace with your authorized LLaMA model ID

    payload = {
        "prompt": prompt,
        "max_gen_len": 150,
        "temperature": 0.3,
        "top_p": 0.9,
        "stop_sequences": []
    }

    response = bedrock_client.invoke_model(
        modelId=model_id,
        body=json.dumps(payload),
        contentType="application/json",
        accept="application/json"
    )

    response_body = json.loads(response["body"].read())
    return response_body["generation"]
