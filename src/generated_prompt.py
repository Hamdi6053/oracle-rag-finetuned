def prompt_template(context_text: str, num_records: int = 5) -> str:
    """
    PDF içeriğinden soru-cevap çiftleri üretmek için prompt template
    """
    return f"""You are an expert at creating high-quality question-answer pairs for instruction tuning from technical documents.

Given the following context from a technical document, generate {num_records} diverse, high-quality question-answer pairs that would be useful for fine-tuning a language model.

Context:
{context_text}

Requirements for question-answer pairs:
1. Questions should be clear, specific, and answerable from the given context
2. Answers should be comprehensive, accurate, and self-contained
3. Vary the question types: factual, explanatory, procedural, analytical
4. Focus on key concepts, processes, definitions, and practical applications
5. Ensure answers provide sufficient detail without being too verbose
6. Questions should cover different aspects of the content (what, how, why, when, where)

Question types to include:
- Definition questions: "What is...?"
- Process questions: "How do you...?"
- Purpose questions: "Why is... important?"
- Feature questions: "What are the key features of...?"
- Comparison questions: "What's the difference between...?"
- Implementation questions: "How do you implement...?"

Generate exactly {num_records} question-answer pairs in the following JSON format:

```json
[
  {{
    "question": "Your question here?",
    "answer": "Your detailed answer here."
  }},
  {{
    "question": "Another question here?",
    "answer": "Another detailed answer here."
  }}
]
```

Each answer should be detailed enough to be useful for training but concise enough to be clear.

IMPORTANT: Return ONLY the JSON array, nothing else. Do not include any explanations, comments, or additional text."""
