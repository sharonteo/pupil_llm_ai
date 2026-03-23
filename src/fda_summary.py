from typing import Dict, Any
import anthropic
import os

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_MODEL = "claude-sonnet-4-6"


def format_metrics_for_prompt(models_dict: Dict[str, Any]) -> str:
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


# ---------------------------------------------------------
# PART 1 — Generate Sections 1–3 and 5 (NO performance narrative)
# ---------------------------------------------------------
def generate_fda_core_summary(dataset_description: str, max_tokens: int = 2500) -> str:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    prompt = f"""
You are generating an FDA-style system narrative summary for a machine learning
clinical decision support (CDS) tool. Write ONLY Sections 1, 2, 3, and 5.
DO NOT write Section 4 here.

Required sections:

1. Study Overview
2. Dataset Characteristics
3. Modeling Approach
5. Limitations and Future Work

Tone:
- FDA-style
- Technical, neutral
- No clinical claims
- No treatment recommendations

Dataset Description:
{dataset_description}
"""

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}]
    )

    blocks = [b.text for b in response.content if b.type == "text"]
    return "\n".join(blocks).strip()


# ---------------------------------------------------------
# PART 2 — Generate ONLY the Performance Interpretation
# ---------------------------------------------------------
def generate_performance_interpretation(models_dict: Dict[str, Any], max_tokens: int = 1200) -> str:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    metrics_text = format_metrics_for_prompt(models_dict)

    prompt = f"""
Write ONLY Section 4: Performance Summary (Interpretation Only).

Use the model metrics below:
{metrics_text}

Write a section titled:
"4. Performance Summary"

Then include the table EXACTLY as provided by the user (the app will insert it).

After the table, write a subsection titled:
"Performance Interpretation"

Write 1–2 paragraphs (8–12 sentences) interpreting:
- accuracy
- precision
- recall (sensitivity)
- specificity
- F1 score
- ROC AUC
- class imbalance effects
- sensitivity–specificity tradeoffs
- synthetic data limitations

Tone:
- FDA-style
- Analytical
- No clinical claims
"""

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}]
    )

    blocks = [b.text for b in response.content if b.type == "text"]
    return "\n".join(blocks).strip()


# ---------------------------------------------------------
# PART 3 — Combine into final FDA summary
# ---------------------------------------------------------
def generate_fda_style_summary(dataset_description: str, models_dict: Dict[str, Any]) -> str:
    core = generate_fda_core_summary(dataset_description)
    perf = generate_performance_interpretation(models_dict)

    return core + "\n\n" + perf