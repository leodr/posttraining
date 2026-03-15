import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

load_dotenv()

# Initialize async client for Kimi K2.5
client = AsyncOpenAI(
    api_key=os.getenv("MOONSHOT_API_KEY"), base_url="https://api.moonshot.ai/v1"
)

SYSTEM_PROMPT = "You are a helpful assistant."
MAX_CONCURRENT = 10
OUTPUT_DIR = Path("conversations")


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
async def generate_response(prompt: str) -> str:
    """Generate a response for a single prompt with retry logic."""
    response = await client.chat.completions.create(
        model="kimi-k2.5",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=1,
        max_tokens=32768,
        top_p=0.95,
    )
    return response.choices[0].message.content


async def process_prompt(
    prompt: str, semaphore: asyncio.Semaphore, pbar: tqdm
) -> dict | None:
    """Process a single prompt with concurrency control."""
    async with semaphore:
        try:
            response = await generate_response(prompt)
            pbar.update(1)
            return {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ]
            }
        except Exception as e:
            pbar.write(f"Failed after retries: {prompt[:50]}... - {e}")
            pbar.update(1)
            return None


def get_output_file(topic: str) -> Path:
    """Get the output file path for a topic."""
    safe_topic = "".join(
        c if c.isalnum() or c in (" ", "-", "_") else "_" for c in topic
    )
    safe_topic = safe_topic.replace(" ", "_").lower()
    return OUTPUT_DIR / f"{safe_topic}.jsonl"


async def process_topic(topic: str, prompts: list[str], pbar_main: tqdm) -> None:
    """Process all prompts for a single topic and save to JSONL file."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    output_file = get_output_file(topic)

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    # Create progress bar for this topic
    with tqdm(
        total=len(prompts),
        desc=f"  {topic}",
        leave=False,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    ) as pbar:

        tasks = [process_prompt(prompt, semaphore, pbar) for prompt in prompts]
        results = await asyncio.gather(*tasks)

    # Filter out failed requests and write to file
    successful_results = [r for r in results if r is not None]

    with open(output_file, "w", encoding="utf-8") as f:
        for result in successful_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    pbar_main.update(1)
    pbar_main.write(
        f"Saved {len(successful_results)}/{len(prompts)} prompts to {output_file}"
    )


async def main():
    """Main entry point."""
    # Load prompts from JSON file
    with open("generated_prompts.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # Group prompts by topic
    topics_data: dict[str, list[str]] = {}
    for item in data:
        topic = item["topic"]
        prompts = item["prompts"]
        if topic in topics_data:
            topics_data[topic].extend(prompts)
        else:
            topics_data[topic] = list(prompts)

    # Filter out already processed topics
    topics_to_process = {
        topic: prompts
        for topic, prompts in topics_data.items()
        if not get_output_file(topic).exists()
    }
    skipped_topics = len(topics_data) - len(topics_to_process)

    total_prompts = sum(len(p) for p in topics_data.values())
    prompts_to_process = sum(len(p) for p in topics_to_process.values())

    print(f"\n{'='*60}")
    print("Kimi K2.5 Data Generation")
    print(f"{'='*60}")
    print(
        f"Topics: {len(topics_data)} ({skipped_topics} already done, {len(topics_to_process)} remaining)"
    )
    print(f"Total prompts: {total_prompts} ({prompts_to_process} to process)")
    print(f"Max concurrent requests: {MAX_CONCURRENT}")
    print(f"Output directory: {OUTPUT_DIR.absolute()}")
    print(f"{'='*60}\n")

    if not topics_to_process:
        print("All topics already processed. Nothing to do.")
        return

    # Process each topic
    with tqdm(
        total=len(topics_to_process),
        desc="Topics completed",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} topics",
    ) as pbar_main:
        for topic, prompts in topics_to_process.items():
            await process_topic(topic, prompts, pbar_main)

    print(f"\n{'='*60}")
    print("Generation complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
