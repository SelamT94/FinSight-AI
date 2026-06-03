# """Prompt templates with `{placeholder}` fields — filled by model wrappers."""

# SYSTEM_PROMPT_FINANCIAL_ANALYST = """You are a financial analyst assistant helping interpret structured bank transaction summaries.

# Rules:
# - Be precise and grounded in the numbers provided in the user message.
# - Reference actual figures from the data when you state facts.
# - If information is missing, say so briefly instead of inventing amounts or categories.
# - Avoid speculation beyond what the data supports."""

# ZERO_SHOT_SUMMARY_PROMPT = """Below is structured account activity for one period.

# Period: {period}
# Total spent (non-deposit outflows): ${total_spent:.2f}
# Total received (deposits): ${total_received:.2f}
# Net flow (received minus spent): ${net_flow:.2f}
# Transaction count: {transaction_count}
# Anomalies flagged: {anomaly_count}

# Category breakdown:
# {category_breakdown_text}

# Write a professional 3–4 sentence summary suitable for an account holder. Use only the figures above."""

# FEW_SHOT_SUMMARY_PROMPT = """You summarize monthly account activity in a concise, professional tone.

# Example 1 — Input:
# Period: 2024-03 | Total spent: $12,400 | Total received: $2,000 | Net flow: -$10,400 | Categories: Withdrawal $9,000 (70%), Bills & Purchases $3,000 (23%), Transfer $400 (3%)
# Example 1 — Summary:
# March shows heavy cash withdrawal activity alongside steady bill payments. Deposits partially offset outflows but net flow remains sharply negative. Withdrawals dominate the category mix and warrant attention if unexpected.

# Example 2 — Input:
# Period: 2024-04 | Total spent: $3,200 | Total received: $5,500 | Net flow: +$2,300 | Categories: Bills & Purchases $2,800 (88%), Deposit-related inflows $5,500
# Example 2 — Summary:
# April is deposit-led with bill payments as the main outflow category. Net flow is positive thanks to larger inflows. Spending is concentrated in bills rather than withdrawals.

# Now summarize the following real period in the same style (3–4 sentences).

# Period: {period}
# Total spent (non-deposit outflows): ${total_spent:.2f}
# Total received (deposits): ${total_received:.2f}
# Net flow: ${net_flow:.2f}
# Transaction count: {transaction_count}
# Anomalies flagged: {anomaly_count}

# Category breakdown:
# {category_breakdown_text}"""

# CHAIN_OF_THOUGHT_SUMMARY_PROMPT = """Think step by step. First identify the dominant spending category (or largest non-deposit category). Then note any unusual patterns (anomalies, concentration, missing diversity). Then compare income (deposits received) vs spending (non-deposit outflows) using the numbers. Finally write a coherent 3–4 sentence summary that reflects those steps — but present only the final summary paragraphs clearly (you may include brief bullets for the reasoning if helpful).

# Data:
# Period: {period}
# Total spent (non-deposit): ${total_spent:.2f}
# Total received (deposits): ${total_received:.2f}
# Net flow: ${net_flow:.2f}
# Transaction count: {transaction_count}
# Anomalies flagged: {anomaly_count}

# Category breakdown:
# {category_breakdown_text}"""

# INSIGHT_GENERATION_PROMPT = """You previously summarized this account period. Now propose actionable insights grounded strictly in the numbers.

# Monthly statement (structured):
# {statement_json}

# Earlier summary:
# {summary}

# Produce exactly 4–5 numbered insights (1. … 2. …). Each insight must:
# - Be one concrete sentence.
# - Reference at least one specific number, category, or count from the data above.
# - Be actionable or diagnostic (what to verify, adjust, or monitor), not generic platitudes."""

# BEHAVIORAL_ANALYSIS_PROMPT = """The following behavioral signals were detected by rules and statistics over the user's transactions (not by a language model):

# {patterns_text}

# Explain in plain English what these patterns likely imply for spending habits and financial health. Stay conservative: tie claims to the signals given. Use short paragraphs or bullets."""

# COMPARATIVE_EVALUATION_PROMPT = """You evaluate another model's summary against the source data.

# Original structured statement (JSON):
# {statement_json}

# Another model's summary to evaluate:
# {llama_summary}

# Respond with a single JSON object only (no markdown fences), with keys:
# - "accuracy_score": integer 1-10
# - "completeness_score": integer 1-10
# - "clarity_score": integer 1-10
# - "alternative_summary": string (your own 3-4 sentence summary)
# - "factual_errors": string (list factual mistakes or contradictions vs the JSON; empty string if none)

# Base scores strictly on consistency with the JSON figures and categories."""

SYSTEM_PROMPT_FINANCIAL_ANALYST = """You are a financial analyst assistant helping interpret structured bank transaction summaries.

Rules:
- Be precise and grounded in the numbers provided in the user message.
- Reference actual figures from the data when you state facts.
- If information is missing, say so briefly instead of inventing amounts or categories.
- Avoid speculation beyond what the data supports."""

#ZERO-SHOT
# Deliberately minimal: no examples, no reasoning scaffold.


# Models must rely purely on their pre-trained instruction-following ability.

# Expected output: varies widely across models 
# It can be verbose or terse.

ZERO_SHOT_SUMMARY_PROMPT = """Summarize the bank account activity below. 
Do not add advice, caveats, or anything not derivable from the numbers given.

Period: {period}
Total spent: ${total_spent:.2f}
Total received: ${total_received:.2f}
Net flow: ${net_flow:.2f}
Transactions: {transaction_count}
Anomalies flagged: {anomaly_count}

Category breakdown:
{category_breakdown_text}

Write the 3-sentence summary now:"""


# FEW-SHOT

# We give it 2 examples: one negative-flow and one positive-flow.

# The examples are carefully chosen to model the tone, length, and structure of the summary.
# Expected output: more consistent across models; closer to the demonstrated style.

FEW_SHOT_SUMMARY_PROMPT = """You summarize monthly bank account activity. Study the two examples carefully — match their tone, length, and structure exactly.

━━━ EXAMPLE 1 (Negative net flow, withdrawal-heavy) ━━━
Input:
  Period: 2024-01 | Spent: $12,400 | Received: $2,000 | Net: -$10,400 | Transactions: 47 | Anomalies: 3
  Categories: Cash Withdrawal $9,000 (73%), Bills $2,800 (23%), Transfer $600 (5%)

Output:
  January recorded $12,400 in outflows against $2,000 in deposits, leaving a net deficit of $10,400. Cash withdrawals at $9,000 (73% of spending) dominated the period, with three flagged anomalies warranting review. Bill payments were steady at $2,800, while the deposit base was insufficient to offset overall outflows.

━━━ EXAMPLE 2 (Positive net flow, bill-payment-heavy) ━━━
Input:
  Period: 2024-04 | Spent: $3,200 | Received: $5,500 | Net: +$2,300 | Transactions: 29 | Anomalies: 0
  Categories: Bills & Purchases $2,800 (88%), Miscellaneous $400 (12%)

Output:
  April closed with a positive net flow of $2,300, driven by $5,500 in deposits against $3,200 in outflows. Spending was concentrated in bills and purchases at $2,800, representing 88% of all expenditures. No anomalies were flagged, indicating a straightforward and predictable month.

━━━ NOW SUMMARIZE THIS PERIOD (same format — 3 sentences) ━━━
Period: {period} | Spent: ${total_spent:.2f} | Received: ${total_received:.2f} | Net: ${net_flow:.2f} | Transactions: {transaction_count} | Anomalies: {anomaly_count}
Categories:
{category_breakdown_text}

Output:"""


# CHAIN-OF-THOUGHT

# Forces visible multi-step reasoning before the final answer.

# Each step is numbered so the reasoning is auditable and presentable.
# Expected output: longest response, most structured, shows model "thinking."

CHAIN_OF_THOUGHT_SUMMARY_PROMPT = """Analyze the bank account data below using the numbered reasoning steps. Show your work for each step, then write the final summary.

Data:
  Period: {period}
  Total spent (outflows): ${total_spent:.2f}
  Total received (deposits): ${total_received:.2f}
  Net flow: ${net_flow:.2f}
  Transaction count: {transaction_count}
  Anomalies flagged: {anomaly_count}

  Category breakdown:
  {category_breakdown_text}

STEP 1 — DOMINANT CATEGORY: Which category has the highest spending? What percentage of total outflows does it represent?

STEP 2 — CASH FLOW HEALTH: Is the net flow positive or negative? By how much? What does this imply about income vs spending balance?

STEP 3 — ANOMALY ASSESSMENT: How many anomalies were flagged? Is this high or low relative to transaction count? What could it suggest?

STEP 4 — SPENDING CONCENTRATION: Is spending spread across many categories or concentrated in one or two? What does this pattern indicate?

STEP 5 — FINAL SUMMARY: Using only the facts established in Steps 1–4, write a coherent 3-sentence professional summary for the account holder.

Begin:"""



# INSIGHT GENERATION
INSIGHT_GENERATION_PROMPT = """You previously summarized this account period. Now propose actionable insights grounded strictly in the numbers.

Monthly statement (structured):
{statement_json}

Earlier summary:
{summary}

Produce exactly 4–5 numbered insights (1. … 2. …). Each insight must:
- Be one concrete sentence.
- Reference at least one specific number, category, or count from the data above.
- Be actionable or diagnostic (what to verify, adjust or monitor) not generic platitudes."""


















# BEHAVIORAL ANALYSIS
BEHAVIORAL_ANALYSIS_PROMPT = """The following behavioral signals were detected by rules and statistics over the user's transactions (not by a language model):

{patterns_text}

Explain in plain English what these patterns likely imply for spending habits and financial health. Stay conservative: tie claims to the signals given. Use short paragraphs or bullets."""


# COMPARATIVE EVALUATION
COMPARATIVE_EVALUATION_PROMPT = """You evaluate another model's summary against the source data.

Original structured statement (JSON):
{statement_json}

Another model's summary to evaluate:
{llama_summary}

Respond with a single JSON object only (no markdown fences), with keys:
- "accuracy_score": integer 1-10
- "completeness_score": integer 1-10
- "clarity_score": integer 1-10
- "alternative_summary": string (your own 3-4 sentence summary)
- "factual_errors": string (list factual mistakes or contradictions vs the JSON; empty string if none)

Base scores strictly on consistency with the JSON figures and categories."""