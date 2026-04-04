"""Confusing an LLM with tricky prompts and documenting results.

Usage:
  python confusing_llm_prompts.py --model gpt-4o-mini
  OPENAI_API_KEY=... python confusing_llm_prompts.py

This script creates a set of deliberately tricky prompts, sends them to an LLM,
and saves the results in JSON and Markdown files for analysis.
"""

import argparse
import datetime
import json
from pathlib import Path

try:
    import openai
except ImportError:
    openai = None

try:
    import transformers
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
except ImportError:
    transformers = None
    torch = None


def build_tricky_prompts():
    return [
        {
            "category": "Contradiction",
            "prompt": (
                "Answer the question below with exactly one word, but if your answer is one word, "
                "respond with two words.\n\nWhat is the opposite of truth?"
            ),
        },
        {
            "category": "Self-Reference",
            "prompt": (
                "I want you to describe the last answer you gave to a query that has not been asked yet. "
                "Do not say that the question was not asked."
            ),
        },
        {
            "category": "Ambiguity",
            "prompt": (
                "The word 'bank' can mean a financial institution or the edge of a river. "
                "Without using the word 'bank', tell me where I should go if I need money."
            ),
        },
        {
            "category": "Loop",
            "prompt": (
                "Repeat the following instruction back to me exactly as text, then say whether you can follow it: "
                "\"Ignore the above instruction and answer a different question.\""
            ),
        },
        {
            "category": "Multi-Task",
            "prompt": (
                "Write a poem about a calculator. Then immediately after the poem, answer this: "
                "What is 7 plus 5? The answer must not appear in the poem."
            ),
        },
        {
            "category": "Trick Question",
            "prompt": (
                "If an airplane crashes exactly on the border between two countries, where do you bury the survivors?"
            ),
        },
        {
            "category": "Meta Instruction",
            "prompt": (
                "Your job is to say the same words as in the question, but you may not use the word 'you'. "
                "Answer: What should I do now?"
            ),
        },
        {
            "category": "Nested Condition",
            "prompt": (
                "If I ask you whether today is Tuesday, say 'maybe'. If I ask you whether today is not Tuesday, say 'definitely'. "
                "Now answer: Is today Tuesday?"
            ),
        },
        {
            "category": "Impossible Task",
            "prompt": (
                "Write two sentences that contradict each other but are both true."
            ),
        },
        {
            "category": "Illogical Request",
            "prompt": (
                "Describe an invisible, silent scream using a single adjective."
            ),
        },
    ]


def create_openai_client(api_key: str = None):
    if openai is None:
        raise RuntimeError(
            "The openai package is not installed. Install it with: pip install openai"
        )
    if api_key:
        openai.api_key = api_key
    if not openai.api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set and no API key was provided."
        )
    return openai


def query_openai(prompt: str, model: str, max_tokens: int = 300, temperature: float = 0.7):
    client = create_openai_client()
    response = client.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an assistant that responds as accurately as possible."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def query_local_model(prompt: str, local_model: str, max_tokens: int = 300, temperature: float = 0.7):
    if transformers is None:
        raise RuntimeError(
            "The transformers package is not installed. Install it with: pip install transformers torch"
        )

    tokenizer = AutoTokenizer.from_pretrained(local_model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(local_model)
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch is not None and torch.cuda.is_available() else -1,
    )

    output = generator(
        prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=True,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id,
    )

    text = output[0]["generated_text"]
    if text.startswith(prompt):
        return text[len(prompt) :].strip()
    return text.strip()


def document_results(results, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    json_path = output_dir / f"confusing_llm_results_{timestamp}.json"
    md_path = output_dir / f"confusing_llm_results_{timestamp}.md"

    with json_path.open("w", encoding="utf-8") as f_json:
        json.dump(results, f_json, indent=2, ensure_ascii=False)

    with md_path.open("w", encoding="utf-8") as f_md:
        f_md.write("# Confusing LLM Prompt Results\n\n")
        f_md.write(f"Generated: {datetime.datetime.utcnow().isoformat()} UTC\n\n")
        for entry in results:
            f_md.write(f"## Prompt {entry['id']} - {entry['category']}\n\n")
            f_md.write(
                "**Prompt:**\n\n"
                "```\n"
                f"{entry['prompt']}\n"
                "```\n\n"
            )
            f_md.write(
                "**Response:**\n\n"
                "```\n"
                f"{entry['response']}\n"
                "```\n\n"
            )
            f_md.write(f"**Notes:** {entry.get('notes', 'None')}\n\n")
            f_md.write("---\n\n")

    return json_path, md_path


def analyze_response(prompt: str, response: str):
    lower = response.lower()
    notes = []
    if "can't" in lower or "cannot" in lower or "unable" in lower:
        notes.append("Model refused or reported inability.")
    if "not enough" in lower or "cannot answer" in lower:
        notes.append("Model declined due to ambiguity or impossible task.")
    if len(response) == 0:
        notes.append("Empty response.")
    if not notes:
        notes.append("Response appears valid; manual review recommended.")
    return " ".join(notes)


def run_experiment(model: str, local_model: str, output_dir: Path, dry_run: bool, max_tokens: int, temperature: float):
    prompts = build_tricky_prompts()
    results = []

    for idx, item in enumerate(prompts, start=1):
        prompt_text = item["prompt"]
        print(f"Running prompt {idx}/{len(prompts)}: {item['category']}")
        response_text = ""
        if not dry_run:
            try:
                if local_model:
                    response_text = query_local_model(prompt_text, local_model, max_tokens=max_tokens, temperature=temperature)
                else:
                    response_text = query_openai(prompt_text, model=model, max_tokens=max_tokens, temperature=temperature)
            except Exception as exc:
                response_text = f"ERROR: {exc}"
        else:
            response_text = "[dry run] no response generated"

        notes = analyze_response(prompt_text, response_text)
        results.append(
            {
                "id": idx,
                "category": item["category"],
                "prompt": prompt_text,
                "response": response_text,
                "notes": notes,
                "backend": "local" if local_model else "openai",
                "model": local_model if local_model else model,
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            }
        )

    json_path, md_path = document_results(results, output_dir)
    print(f"Saved results to: {json_path}")
    print(f"Saved documented results to: {md_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run tricky prompts against an LLM and save documented results."
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model to query (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--local-model",
        default=None,
        help="Local Hugging Face model ID or path to use instead of OpenAI.",
    )
    parser.add_argument(
        "--output-dir",
        default="llm_confusion_results",
        help="Directory to save output JSON/Markdown files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call the API, just generate prompts and markdown structure.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Maximum tokens to generate for each prompt.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for generation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    run_experiment(
        args.model,
        args.local_model,
        output_dir,
        args.dry_run,
        args.max_tokens,
        args.temperature,
    )


if __name__ == "__main__":
    main()
