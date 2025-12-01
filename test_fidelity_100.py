#!/usr/bin/env python3
"""
Comprehensive 100-question fidelity test against the user's PA.
Tests GREEN (>=0.75), YELLOW (0.70-0.75), ORANGE (0.60-0.70), RED (<0.60) thresholds.
"""
import os
import sys
import numpy as np
import toml

sys.path.insert(0, '/Users/brunnerjf/Desktop/telos_privacy')
os.chdir('/Users/brunnerjf/Desktop/telos_privacy')

secrets_path = '/Users/brunnerjf/Desktop/telos_privacy/.streamlit/secrets.toml'
secrets = toml.load(secrets_path)

from mistralai import Mistral

class MistralEmbeddingProvider:
    def __init__(self):
        self.client = Mistral(api_key=secrets.get("MISTRAL_API_KEY"))
        self.model = "mistral-embed"
        self.dimension = 1024

    def encode(self, text: str) -> list:
        response = self.client.embeddings.create(model=self.model, inputs=[text])
        return response.data[0].embedding

# PA Configuration
PA_PURPOSE = "Explore and test TELOS Observatory capabilities, understand AI governance through purposeful conversation"
PA_SCOPE = "Focus on: AI governance, TELOS features, meaningful dialogue. Avoid: completely unrelated tangents"
PA_TEXT = f"Purpose: {PA_PURPOSE}. Scope: {PA_SCOPE}."

# Thresholds
FIDELITY_GREEN = 0.75
FIDELITY_YELLOW = 0.70
FIDELITY_ORANGE = 0.60

# 100 Test Questions - categorized by expected zone
TEST_QUESTIONS = {
    "expected_green": [
        # Direct TELOS/AI governance questions (should be GREEN)
        "How does TELOS detect conversational drift?",
        "Explain the TELOS Observatory's governance features",
        "What is AI governance and how does TELOS implement it?",
        "How does the fidelity calculation work in TELOS?",
        "What are primacy attractors in the TELOS framework?",
        "Tell me about the TELOS dual attractor system",
        "How does TELOS measure purpose alignment?",
        "What intervention strategies does TELOS use?",
        "Explain basin membership in TELOS mathematics",
        "How does TELOS handle edge cases in governance?",
        "What is the role of embeddings in TELOS?",
        "How does TELOS maintain conversational focus?",
        "Explain the TELOS primacy state calculation",
        "What makes TELOS different from other AI governance approaches?",
        "How does the TELOS Observatory visualize alignment?",
        "Can you demonstrate TELOS drift detection?",
        "What are the key metrics TELOS tracks?",
        "How does TELOS balance user freedom with governance?",
        "Explain the mathematical foundation of TELOS",
        "What is purposeful conversation in TELOS terms?",
        "How does TELOS use cosine similarity?",
        "What triggers a TELOS intervention?",
        "How does the Steward feature work in TELOS?",
        "Explain TELOS fidelity zones and thresholds",
        "What is the TELOSCOPE feature?",
    ],
    "expected_yellow": [
        # Related but slightly tangential (should be YELLOW)
        "I want to discuss my AI alignment research",
        "Can we explore ethical AI development?",
        "What makes dialogue meaningful?",
        "How do conversational AI systems maintain context?",
        "Let's talk about AI safety principles",
        "What are best practices for human-AI interaction?",
        "How do you measure conversation quality?",
        "What is responsible AI development?",
        "How do AI assistants stay on topic?",
        "What defines purposeful communication?",
        "Tell me about AI alignment techniques",
        "How do chatbots handle off-topic requests?",
        "What is conversational AI governance?",
        "How do we ensure AI serves user goals?",
        "What makes AI assistance effective?",
        "How do AI systems detect user intent?",
        "What is semantic similarity in AI?",
        "How do embeddings represent meaning?",
        "What is the future of AI governance?",
        "How can AI be more aligned with human values?",
    ],
    "expected_orange": [
        # Tangentially related tech topics (should be ORANGE)
        "How do neural networks learn patterns?",
        "What is machine learning?",
        "Explain deep learning architectures",
        "How do transformers work?",
        "What is natural language processing?",
        "How do large language models generate text?",
        "What is the attention mechanism?",
        "How does GPT work?",
        "What are vector databases?",
        "How do recommendation systems work?",
        "What is reinforcement learning?",
        "How do chatbots understand questions?",
        "What is sentiment analysis?",
        "How do search engines rank results?",
        "What is data science?",
        "How do AI models handle bias?",
        "What is prompt engineering?",
        "How do you fine-tune language models?",
        "What is the difference between AI and ML?",
        "How do computers process language?",
    ],
    "expected_red": [
        # Completely unrelated topics (should be RED)
        "What's a good recipe for chocolate cake?",
        "Tell me about the history of ancient Rome",
        "How do I fix my car's transmission?",
        "What are the rules of basketball?",
        "How do you grow tomatoes?",
        "What's the capital of Mongolia?",
        "How do airplanes fly?",
        "What causes earthquakes?",
        "How do I learn to play guitar?",
        "What's the best way to lose weight?",
        "Tell me about World War 2",
        "How do vaccines work?",
        "What is photosynthesis?",
        "How do I train my dog?",
        "What's the weather like on Mars?",
        "How do I bake bread?",
        "What are the rules of chess?",
        "How do submarines work?",
        "What is the stock market?",
        "How do I change a tire?",
        "What's the best vacation destination?",
        "How do birds migrate?",
        "What causes the northern lights?",
        "How do I start a garden?",
        "What is cryptocurrency?",
        "How do I improve my golf swing?",
        "What's the plot of Romeo and Juliet?",
        "How do refrigerators work?",
        "What are the planets in our solar system?",
        "How do I make pizza dough?",
        "What is the speed of light?",
        "How do I declutter my house?",
        "What are the symptoms of the flu?",
        "How do volcanoes form?",
        "What's the best way to study for exams?",
    ],
}

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_zone(score):
    if score >= FIDELITY_GREEN:
        return "GREEN"
    elif score >= FIDELITY_YELLOW:
        return "YELLOW"
    elif score >= FIDELITY_ORANGE:
        return "ORANGE"
    else:
        return "RED"

def main():
    print("=" * 80)
    print("TELOS 100-Question Fidelity Calibration Test")
    print("=" * 80)
    print(f"\nThresholds: GREEN >= {FIDELITY_GREEN}, YELLOW >= {FIDELITY_YELLOW}, ORANGE >= {FIDELITY_ORANGE}, RED < {FIDELITY_ORANGE}")
    print(f"\nPA: {PA_TEXT[:80]}...")

    provider = MistralEmbeddingProvider()
    pa_embedding = np.array(provider.encode(PA_TEXT))

    all_results = []
    zone_accuracy = {"GREEN": {"correct": 0, "total": 0}, "YELLOW": {"correct": 0, "total": 0},
                     "ORANGE": {"correct": 0, "total": 0}, "RED": {"correct": 0, "total": 0}}

    for expected_zone, questions in TEST_QUESTIONS.items():
        expected = expected_zone.replace("expected_", "").upper()
        print(f"\n{'=' * 80}")
        print(f"Testing {expected} zone ({len(questions)} questions)")
        print("=" * 80)

        for q in questions:
            q_embedding = np.array(provider.encode(q))
            score = cosine_similarity(q_embedding, pa_embedding)
            actual_zone = get_zone(score)
            correct = actual_zone == expected

            zone_accuracy[expected]["total"] += 1
            if correct:
                zone_accuracy[expected]["correct"] += 1

            marker = "✓" if correct else "✗"
            print(f"{score:.3f} {actual_zone:6} {marker} | {q[:55]}{'...' if len(q) > 55 else ''}")

            all_results.append({
                "question": q,
                "score": score,
                "expected": expected,
                "actual": actual_zone,
                "correct": correct
            })

    # Summary
    scores = [r["score"] for r in all_results]
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total questions: {len(all_results)}")
    print(f"Score range: {min(scores):.3f} - {max(scores):.3f}")
    print(f"Mean: {np.mean(scores):.3f}, Std: {np.std(scores):.3f}")

    print("\nZone Accuracy:")
    total_correct = 0
    total_questions = 0
    for zone, stats in zone_accuracy.items():
        if stats["total"] > 0:
            acc = stats["correct"] / stats["total"] * 100
            print(f"  {zone}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")
            total_correct += stats["correct"]
            total_questions += stats["total"]

    overall_acc = total_correct / total_questions * 100
    print(f"\nOverall Accuracy: {total_correct}/{total_questions} ({overall_acc:.1f}%)")

    # Zone distribution
    print("\nActual Zone Distribution:")
    zone_counts = {"GREEN": 0, "YELLOW": 0, "ORANGE": 0, "RED": 0}
    for r in all_results:
        zone_counts[r["actual"]] += 1
    for zone, count in zone_counts.items():
        print(f"  {zone}: {count}")

    # Misclassifications
    misclassified = [r for r in all_results if not r["correct"]]
    if misclassified:
        print(f"\nMisclassified ({len(misclassified)}):")
        for r in misclassified[:10]:  # Show first 10
            print(f"  {r['score']:.3f} Expected {r['expected']}, Got {r['actual']}: {r['question'][:50]}...")

if __name__ == "__main__":
    main()
