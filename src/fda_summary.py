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
    max_tokens: int = 2000   # <-- increased from 900
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
You are generating an FDA-style system narrative summary for a machine learning–
based clinical decision support (CDS) tool. The summary must be formal, structured,
and written in a regulatory-appropriate tone. Do NOT provide clinical claims or
treatment recommendations. Focus on system behavior, dataset description, and
model performance.

Use the dataset description and model metrics below to produce a comprehensive,
well-structured narrative.

Dataset Description:
{dataset_description}

Model Performance (binary outcome: gcs_severe = GCS ≤ 8):
{metrics_text}

Write the summary using the following required sections:

1. Study Overview
   - Describe the purpose of the CDS tool.
   - State that it supports, not replaces, clinical judgment.
   - Define the prediction target (gcs_severe).

2. Dataset Characteristics
   - State that the dataset is synthetic and contains 5,000 observations.
   - Describe the simulated multi-site nature (4 sites).
   - List all available features: demographics, diagnosis, pupillometry parameters.
   - Explain that findings are preliminary and require validation on real clinical data.

3. Modeling Approach
   - Describe the three models evaluated (Logistic Regression, Random Forest, XGBoost).
   - Summarize preprocessing steps (one-hot encoding, scaling, stratified split).
   - List evaluation metrics used.
   - Note that no hyperparameter tuning or class-imbalance handling was performed.

4. Performance Summary
   - Present the model performance metrics in a table.
   - AFTER the table, write a full narrative interpretation in paragraph form.
   - The narrative MUST interpret:
       * accuracy
       * precision
       * recall (sensitivity)
       * specificity
       * F1 score
       * ROC AUC
   - The narrative MUST explain:
       * why sensitivity is modest
       * how class imbalance affects recall
       * the tradeoff between sensitivity and specificity
       * limitations of synthetic data
   - The narrative MUST be at least 6–8 sentences.

5. Limitations and Future Work
   - Emphasize synthetic data limitations.
   - Mention need for real-world validation.
   - Note opportunities for improving sensitivity (class weighting, threshold tuning, etc.).
   - Clarify that the tool does not provide diagnostic or treatment recommendations.

Tone Requirements:
- Neutral, technical, and FDA-style.
- Avoid marketing language.
- Avoid clinical claims.
- Focus on system description, not clinical interpretation.
- Use clear section headers and concise paragraphs.
"""

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}]
    )

    # Debug: check if Claude hit the token limit
    print("stop_reason:", response.stop_reason)

    text_blocks = []
    for block in response.content:
        if block.type == "text":
            text_blocks.append(block.text)

    return "\n".join(text_blocks).strip()
