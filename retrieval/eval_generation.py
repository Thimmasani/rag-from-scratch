"""
Groundedness evaluation, Part 6 of the RAG-from-scratch series.

Part 5 ended with a real, observed problem: generation can cite a real
source while still making a claim that source doesn't actually support
(the "out of stock" fabrication). Citation attribution (lexical overlap)
tells you WHICH source contributed vocabulary; it says nothing about
whether the specific claim is TRUE according to that source. This module
tries to close that gap with an LLM-as-judge groundedness check, tests it
against the exact fabrication Part 5 found, and is honest about the new
tradeoff that check introduces.

Two judge designs, tested against the same small labeled set:

  1. NAIVE judge: "Is this claim supported by this source? SUPPORTED or
     UNSUPPORTED." -- a single holistic judgment call.

  2. QUOTE judge: "Quote the exact words in the source that support this
     claim. If none exist, say NO SUCH TEXT EXISTS." -- forces the model
     to point at specific evidence rather than make a holistic call,
     following the same "decompose into a narrower, more mechanical task"
     pattern that fixed abstention and citations in Part 5.

Finding, confirmed directly: the naive judge fails to catch the real
fabrication from Part 5 (says SUPPORTED when it should say UNSUPPORTED).
The quote judge catches that fabrication correctly -- but ALSO incorrectly
rejects a true claim that paraphrases its source rather than quoting it
verbatim. Neither judge is simply "the fix." Forcing specific evidence
trades false negatives (missed fabrications) for new false positives
(true, paraphrased claims wrongly flagged as unsupported).
"""

from transformers import AutoModelForCausalLM, AutoTokenizer


class GroundednessJudge:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def _generate(self, prompt: str, max_new_tokens: int) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()

    def naive_judge(self, claim: str, source: str) -> bool:
        """Single holistic judgment call. Returns True if judged SUPPORTED."""
        prompt = f"""Source: {source}

Claim: {claim}

Is the claim fully supported by the source? Answer with exactly one word: SUPPORTED or UNSUPPORTED."""
        response = self._generate(prompt, max_new_tokens=10)
        return response.upper().startswith("SUPPORTED")

    def quote_judge(self, claim: str, source: str) -> bool:
        """Forces the model to point at specific evidence rather than make
        a holistic call. Returns True if the model found supporting text
        (i.e. did NOT say 'NO SUCH TEXT EXISTS')."""
        prompt = f"""Source: {source}

Claim: {claim}

Quote the exact words from the source that support this claim. If no such support exists in the source, respond with exactly: NO SUCH TEXT EXISTS."""
        response = self._generate(prompt, max_new_tokens=60)
        return "NO SUCH TEXT EXISTS" not in response.upper()


# Labeled test set: (claim, source, is_actually_supported). The first two
# are Part 5's real pipeline output, verbatim -- not constructed for this
# part. The third is the paraphrase case that reveals the quote judge's
# own blind spot.
LABELED_CASES = [
    (
        "SKU-48213-B is an automotive component that needs to be replaced due to being on back order until the next week.",
        "replacement part SKU-48213-B is on back order until next week",
        True,
        "Part 5's correctly-grounded sentence",
    ),
    (
        "This suggests that the vehicle's engine oil and brake system require immediate attention or replacement as they have been out of stock for this period.",
        "my automobile requires fresh engine oil and brake service",
        False,
        "Part 5's fabricated 'out of stock' claim",
    ),
    (
        "The SKU is currently unavailable and will not ship for several days.",
        "replacement part SKU-48213-B is on back order until next week",
        True,
        "true claim, but PARAPHRASED rather than quoted verbatim",
    ),
]


def run_eval(judge: GroundednessJudge):
    for name, judge_fn in [("naive_judge", judge.naive_judge), ("quote_judge", judge.quote_judge)]:
        print(f"--- {name} ---")
        correct = 0
        for claim, source, expected, label in LABELED_CASES:
            predicted = judge_fn(claim, source)
            is_correct = predicted == expected
            correct += is_correct
            print(f"  [{'OK' if is_correct else 'WRONG'}] expected={'SUPPORTED' if expected else 'UNSUPPORTED'}  predicted={'SUPPORTED' if predicted else 'UNSUPPORTED'}  ({label})")
        print(f"  accuracy: {correct}/{len(LABELED_CASES)}")
        print()


if __name__ == "__main__":
    judge = GroundednessJudge()
    run_eval(judge)
