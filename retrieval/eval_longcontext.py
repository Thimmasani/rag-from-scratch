"""
RAG vs. long-context evaluation driver, Part 10 of the RAG-from-scratch
series.

Two experiments:
  1. RAG vs. stuffing cost table: prompt token count and generation
     wall-clock latency for (a) retrieving just the relevant chunk vs.
     (b) stuffing a large padded context, at a couple of context sizes.
  2. "Lost in the middle": fixed total context length, needle fact
     placed at start/middle/end, checking via Generator.answer() whether
     the model's answer is actually grounded in the needle at each
     position.

Context sizes (500 / 2,000 / 4,000 target tokens) were chosen after
benchmarking actual CPU generation time for this model at each size
beforehand (roughly 1.7s / 7s / 19s for a 20-token completion) -- large
enough to meaningfully exercise a long context, small enough to stay
practical to run repeatedly on CPU.
"""

import time

from transformers import AutoTokenizer

from chunking_corpus import FACTS
from generation import Generator
from longcontext import build_context

QUERY = "When must a return be initiated to get a full refund?"
NEEDLE_FACT = FACTS[1]  # "Returns must be initiated within 45 days of delivery for a full refund."


def cost_table(generator: Generator, tokenizer: AutoTokenizer, target_sizes: list[int]):
    print("=== 1. RAG vs. stuffing: token cost and latency ===\n")
    for target in target_sizes:
        # (a) RAG: just the one relevant chunk.
        t0 = time.perf_counter()
        rag_result = generator.answer(QUERY, [NEEDLE_FACT])
        t1 = time.perf_counter()
        rag_prompt_tokens = len(tokenizer.encode(NEEDLE_FACT)) + 40  # + prompt scaffolding, approx

        # (b) Stuffing: the needle padded out to `target` tokens of filler.
        stuffed_context = build_context(NEEDLE_FACT, "middle", target, tokenizer)
        actual_tokens = len(tokenizer.encode(stuffed_context))
        t2 = time.perf_counter()
        stuff_result = generator.answer(QUERY, [stuffed_context])
        t3 = time.perf_counter()

        print(f"target context size: {target} tokens (actual stuffed: {actual_tokens})")
        print(f"  RAG:      ~{rag_prompt_tokens} prompt tokens, {t1 - t0:.2f}s generation, answer: {rag_result['answer']!r}")
        print(f"  Stuffing: ~{actual_tokens} prompt tokens, {t3 - t2:.2f}s generation, answer: {stuff_result['answer']!r}")
        print(f"  Latency ratio (stuffing / RAG): {(t3 - t2) / (t1 - t0):.1f}x")
        print()


def lost_in_the_middle(generator: Generator, tokenizer: AutoTokenizer, target_tokens: int = 2000):
    print(f"=== 2. Lost in the middle (fixed {target_tokens}-token context) ===\n")
    for position in ["start", "middle", "end"]:
        context = build_context(NEEDLE_FACT, position, target_tokens, tokenizer)
        result = generator.answer(QUERY, [context])
        grounded_correctly = "45 days" in result["answer"] or "45" in result["answer"]
        print(f"position={position:<7}  grounded={result['grounded']}  answer={result['answer']!r}")
        print(f"  Contains the actual answer ('45 days')? {grounded_correctly}")
        print()


if __name__ == "__main__":
    generator = Generator()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    cost_table(generator, tokenizer, target_sizes=[500, 2000, 4000])
    lost_in_the_middle(generator, tokenizer, target_tokens=2000)
