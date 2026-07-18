"""
utils.py
--------
Shared constants and helpers used across zero-shot, few-shot, and
fine-tuning scripts: the tag taxonomy, prompt templates, response parsing,
and the local/OpenAI LLM backend switch.
"""

import os
import re

CATEGORIES = [
    "Billing",
    "Technical Issue",
    "Account Access",
    "Feature Request",
    "Bug Report",
    "General Inquiry",
    "Refund Request",
    "Shipping and Delivery",
]

# A couple of hand-picked labeled examples per category, used for few-shot prompting.
FEW_SHOT_EXAMPLES = [
    {"text": "I was charged twice for my subscription this month, can you fix this?", "label": "Billing"},
    {"text": "The app crashes every time I try to upload a photo.", "label": "Technical Issue"},
    {"text": "I can't log into my account even after resetting my password.", "label": "Account Access"},
    {"text": "It would be great if the app supported dark mode.", "label": "Feature Request"},
    {"text": "The export button does nothing when I click it on the dashboard.", "label": "Bug Report"},
    {"text": "What are your customer support hours?", "label": "General Inquiry"},
    {"text": "I want a refund for the order I cancelled last week.", "label": "Refund Request"},
    {"text": "My package was supposed to arrive three days ago and still hasn't shown up.", "label": "Shipping and Delivery"},
]


def build_zero_shot_prompt(ticket_text, categories=CATEGORIES):
    category_list = "\n".join(f"- {c}" for c in categories)
    return (
        "You are a support ticket classifier. Given the ticket below, choose the "
        "TOP 3 most likely categories from this list, ordered from most to least likely:\n"
        f"{category_list}\n\n"
        f'Ticket: "{ticket_text}"\n\n'
        "Respond with ONLY a comma-separated list of exactly 3 category names from the "
        "list above, most likely first. Do not explain your answer."
    )


def build_few_shot_prompt(ticket_text, examples=FEW_SHOT_EXAMPLES, categories=CATEGORIES):
    category_list = "\n".join(f"- {c}" for c in categories)
    example_block = "\n".join(f'Ticket: "{ex["text"]}"\nTop category: {ex["label"]}\n' for ex in examples)
    return (
        "You are a support ticket classifier. Categories:\n"
        f"{category_list}\n\n"
        "Here are some labeled examples:\n\n"
        f"{example_block}\n"
        f'Now classify this new ticket: "{ticket_text}"\n\n'
        "Respond with ONLY a comma-separated list of exactly 3 category names from the "
        "list above, ordered from most to least likely. Do not explain your answer."
    )


def parse_top3(response_text, categories=CATEGORIES):
    """Extracts up to 3 valid category names from a free-text LLM response,
    matched case-insensitively against the known category list."""
    found = []
    lowered_categories = {c.lower(): c for c in categories}

    # Split on commas/newlines/semicolons, then fuzzy-match each piece
    pieces = re.split(r"[,;\n]", response_text)
    for piece in pieces:
        piece = piece.strip().strip(".").lower()
        if not piece:
            continue
        if piece in lowered_categories:
            canonical = lowered_categories[piece]
            if canonical not in found:
                found.append(canonical)
            continue
        # fallback: substring match (handles cases like "1. Billing" or extra words)
        for cat_lower, canonical in lowered_categories.items():
            if cat_lower in piece and canonical not in found:
                found.append(canonical)
                break
        if len(found) == 3:
            break

    return found[:3]


def build_llm(llm_backend="local", local_model="google/flan-t5-base", openai_model="gpt-4o-mini"):
    """Returns a simple `generate(prompt) -> str` callable for either backend."""
    if llm_backend == "openai":
        from openai import OpenAI
        from dotenv import load_dotenv

        load_dotenv()
        if not os.environ.get("OPENAI_API_KEY"):
            raise EnvironmentError(
                "OPENAI_API_KEY not set. Add it to a .env file or your environment, "
                "or use --llm_backend local instead."
            )
        client = OpenAI()

        def generate(prompt):
            response = client.chat.completions.create(
                model=openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=50,
            )
            return response.choices[0].message.content

        return generate

    # Local backend: no API key required
    from transformers import pipeline

    text_gen_pipeline = pipeline("text2text-generation", model=local_model, max_new_tokens=50)

    def generate(prompt):
        result = text_gen_pipeline(prompt, do_sample=False)
        return result[0]["generated_text"]

    return generate
