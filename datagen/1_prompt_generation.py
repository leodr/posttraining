import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

with open("topics.json", "r") as f:
    topics = json.load(f)

SYSTEM_PROMPT = """You generate diverse user prompts for training a conversational AI assistant.

Given a topic and description, generate 10 different user prompts that someone might ask about this topic. Make them varied:
- Different complexity levels (simple to advanced)
- Different styles (casual, formal, curious, confused)
- Different formats (questions, requests, statements that invite response)

Output as a JSON array of 10 strings. No explanation, just the JSON array."""


def generate_prompts_for_topic(
    category: str, topic_name: str, topic_description: str
) -> list[str]:
    user_message = (
        f"Category: {category}\nTopic: {topic_name}\nDescription: {topic_description}"
    )

    response = client.chat.completions.create(
        model="moonshotai/kimi-k2-0905",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "prompts",
                "description": "An array of diverse user prompts for training a conversational AI assistant",
                "strict": True,
                "schema": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        },
        timeout=60,
    )

    content = response.choices[0].message.content
    parsed = json.loads(content)

    # Handle different possible JSON structures
    if isinstance(parsed, list):
        return parsed
    elif isinstance(parsed, dict):
        # Find the array in the dict
        for value in parsed.values():
            if isinstance(value, list):
                return value

    return []


def load_existing_prompts() -> list:
    try:
        with open("generated_prompts.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def save_prompts(all_prompts: list):
    with open("generated_prompts.json", "w") as f:
        json.dump(all_prompts, f, indent=2)


def main():
    all_prompts = load_existing_prompts()
    min_prompts_per_topic = 100

    # Build a lookup for existing topic data
    topic_lookup = {(item["category"], item["topic"]): item for item in all_prompts}

    for category_data in topics:
        category = category_data["category"]

        for topic in category_data["topics"]:
            topic_name = topic["name"]
            topic_description = topic["description"]
            key = (category, topic_name)

            existing_item = topic_lookup.get(key)
            current_count = len(existing_item["prompts"]) if existing_item else 0

            if current_count >= min_prompts_per_topic:
                print(f"Skipping {category} > {topic_name} ({current_count} prompts)")
                continue

            prompts_needed = min_prompts_per_topic - current_count
            # Each request returns ~10 prompts
            requests_needed = (prompts_needed + 9) // 10
            requests_needed = min(requests_needed, 10)

            print(
                f"Generating for {category} > {topic_name} "
                f"({current_count} prompts, need {prompts_needed} more, {requests_needed} requests)..."
            )

            topic_prompts = []

            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {
                    executor.submit(
                        generate_prompts_for_topic,
                        category,
                        topic_name,
                        topic_description,
                    ): i
                    for i in range(requests_needed)
                }

                for future in as_completed(futures):
                    i = futures[future]
                    try:
                        prompts = future.result()
                        topic_prompts.extend(prompts)
                        print(
                            f"  Request {i + 1}/{requests_needed}: got {len(prompts)} prompts"
                        )
                    except Exception as e:
                        print(f"  Request {i + 1}/{requests_needed}: error - {e}")

            if existing_item:
                existing_item["prompts"].extend(topic_prompts)
            else:
                new_item = {
                    "category": category,
                    "topic": topic_name,
                    "prompts": topic_prompts,
                }
                all_prompts.append(new_item)
                topic_lookup[key] = new_item

            save_prompts(all_prompts)

    total = sum(len(item["prompts"]) for item in all_prompts)
    print(f"\nDone. Generated {total} prompts total. Saved to generated_prompts.json")


if __name__ == "__main__":
    main()
