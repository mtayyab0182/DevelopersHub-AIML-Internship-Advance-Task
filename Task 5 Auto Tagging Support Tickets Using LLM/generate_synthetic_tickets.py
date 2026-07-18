"""
generate_synthetic_tickets.py
------------------------------
Generates a synthetic free-text support ticket dataset labeled with one
true category per ticket, so the full pipeline can be tested without
needing to source/download a real dataset first.

Usage:
    python generate_synthetic_tickets.py --n_per_category 40 --output_csv ./data/tickets.csv
"""

import argparse
import os
import random
import pandas as pd

from utils import CATEGORIES

TEMPLATES = {
    "Billing": [
        "I was charged {amount} but my plan should only cost {amount2}.",
        "Why does my invoice show two charges for the same month?",
        "Can you explain the extra {amount} fee on my latest bill?",
        "My credit card was billed twice for {product}.",
        "I need an itemized receipt for my {product} purchase.",
    ],
    "Technical Issue": [
        "The {product} app keeps freezing when I open the settings page.",
        "I'm getting a blank screen every time I try to load {product}.",
        "The website times out whenever I try to check out.",
        "{product} won't sync across my devices anymore.",
        "The video keeps buffering no matter my internet speed.",
    ],
    "Account Access": [
        "I can't log into my account even after resetting my password.",
        "Two-factor authentication isn't sending me a code.",
        "My account got locked after too many login attempts.",
        "I changed my email and now I can't sign in at all.",
        "It says my account doesn't exist but I've used it for years.",
    ],
    "Feature Request": [
        "It would be great if {product} supported dark mode.",
        "Can you add a way to export data to CSV?",
        "Please consider adding multi-language support.",
        "A bulk-edit option would save me a lot of time.",
        "Would love to see integration with {product2}.",
    ],
    "Bug Report": [
        "The export button does nothing when I click it.",
        "Clicking 'save' deletes my draft instead of saving it.",
        "The search results show items that don't match my query.",
        "Notifications are duplicated every time I open the app.",
        "The date picker shows the wrong month by default.",
    ],
    "General Inquiry": [
        "What are your customer support hours?",
        "Do you offer a student discount?",
        "Is {product} available in my country?",
        "How do I contact your sales team?",
        "Where can I find your terms of service?",
    ],
    "Refund Request": [
        "I want a refund for the order I cancelled last week.",
        "I was charged after cancelling my subscription, please refund me.",
        "The product arrived damaged, I'd like my money back.",
        "Can I get a refund since I never used the {product} license?",
        "I accidentally purchased the wrong plan, please refund the difference.",
    ],
    "Shipping and Delivery": [
        "My package was supposed to arrive three days ago and still hasn't shown up.",
        "The tracking number for my order isn't working.",
        "I received someone else's order instead of mine.",
        "Can I change the delivery address after placing an order?",
        "My {product} arrived without the accessories listed on the box.",
    ],
}

FILLERS = {
    "{amount}": ["$29.99", "$14.50", "$99.00", "$5.00"],
    "{amount2}": ["$19.99", "$9.50", "$79.00", "$2.00"],
    "{product}": ["the Pro plan", "my subscription", "the mobile app", "the desktop client", "my order"],
    "{product2}": ["Slack", "Google Calendar", "Zapier", "Notion"],
}


def fill_template(template, rng):
    text = template
    for placeholder, options in FILLERS.items():
        if placeholder in text:
            text = text.replace(placeholder, rng.choice(options))
    return text


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_per_category", type=int, default=40)
    parser.add_argument("--output_csv", type=str, default="./data/tickets.csv")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    rows = []
    ticket_id = 0
    for category in CATEGORIES:
        templates = TEMPLATES[category]
        for _ in range(args.n_per_category):
            template = rng.choice(templates)
            text = fill_template(template, rng)
            rows.append({"id": ticket_id, "text": text, "true_category": category})
            ticket_id += 1

    df = pd.DataFrame(rows).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    df["id"] = range(len(df))

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(df)} tickets across {len(CATEGORIES)} categories to {args.output_csv}")
    print(df["true_category"].value_counts())


if __name__ == "__main__":
    main()
