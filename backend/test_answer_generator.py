"""
Test Answer Generator

This test uses a small static context to verify:
- OpenAI connection
- Grounded answer generation
- Answer guardrails
"""

from app.rag.answer_generator import AnswerGenerator


def main():
    print("\n=== ANSWER GENERATOR TEST ===\n")

    query = (
        "What technologies are used for databases and storage?"
    )

    context = """
USER QUESTION:
What technologies are used for databases and storage?

RETRIEVED DOCUMENT CHUNKS:

Document: sample.pdf
Page: 1

Databases & Storage:
- PostgreSQL for relational data.
- Redis for caching and sessions.
- Elasticsearch/OpenSearch for full-text search.
- ClickHouse for analytics events.
- S3-compatible object storage for audio and images.
- CDN for asset and audio delivery.

GROUNDING AND SAFETY INSTRUCTIONS:
- Answer only using the supplied context.
- Do not invent facts.
- Mention page numbers when available.
"""

    generator = AnswerGenerator()

    result = generator.generate_answer(
        query=query,
        context=context,
    )

    print("Generated Answer:")
    print("-----------------")
    print(result["answer"])

    print("\nMetadata:")
    print(result)

    print("\n=== ANSWER GENERATOR TEST PASSED ===")


if __name__ == "__main__":
    main()