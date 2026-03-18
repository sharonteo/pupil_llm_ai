from typing import Dict, Any
import anthropic
from pathlib import Path

# Load API key from environment variable
import os
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Claude model (Sonnet 4–6 depending on your account)
CLAUDE_MODEL = "claude-sonnet-4-6"

def format_metrics_for_prompt(models_dict: Dict[str, Any]) -> str:
    """Format model metrics into readable text for the LLM prompt."""
    lines = []
    for name, info in models_dict.items():
        m = info["metrics"]
        lines.append(
            f"- {name}: "
            f"accuracy={m['accuracy']:.3f}, precision={m['precision']:.3f}, "
            f"recall={m['recall']:.3f}, f1={m['f1']:.3f}, roc_auc={m['roc_auc']:.3f}, "
            f"sensitivity={m['sensitivity']:.3f}, specificity={m['specificity']:.3f}"
        )
    return "\n".join(lines)


def generate_fda_style_summary(
    dataset_description: str,
    models_dict: Dict[str, Any],
    max_tokens: int = 900
) -> str:
    """Generate a structured FDA-style narrative using Claude Sonnet."""

    if not ANTHROPIC_API_KEY:
        return (
            "⚠️ Anthropic API key not found.\n"
            "Set ANTHROPIC_API_KEY in your environment to enable FDA narrative generation."
        )

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    metrics_text = format_metrics_for_prompt(models_dict)

    prompt = f"""
You are drafting a concise, regulator-facing summary for an FDA submission
for a pupillometry-based clinical decision support tool.

Dataset:
{dataset_description}

Model performance (binary outcome: GCS severe vs non-severe):
{metrics_text}

Write a structured narrative with the following sections:

1. Study Overview  
2. Dataset Characteristics  
3. Modeling Approach  
4. Performance Summary  
   - Interpret sensitivity, specificity, and ROC AUC  
5. Clinical Risk Considerations and Limitations  
6. Conclusion and Next Steps  

Use clear, formal language suitable for FDA reviewers. Avoid marketing language.
Keep the tone objective, evidence-based, and concise.
"""

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}]
    )

    # Anthropic returns content as blocks
    text_blocks = []
    for block in response.content:
        if block.type == "text":
            text_blocks.append(block.text)

    return "\n".join(text_blocks).strip()