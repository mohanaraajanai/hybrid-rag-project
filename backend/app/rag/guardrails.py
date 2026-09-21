import re
from typing import Any


class GuardrailResult:
    """
    Represents the result of a guardrail validation.
    """

    def __init__(
        self,
        allowed: bool,
        message: str,
        cleaned_query: str = "",
    ):
        self.allowed = allowed
        self.message = message
        self.cleaned_query = cleaned_query


class RAGGuardrails:
    """
    Input and output guardrails for the RAG application.
    """

    MAX_QUERY_LENGTH = 500
    MAX_CONTEXT_LENGTH = 50000
    MAX_ANSWER_LENGTH = 5000

    BLOCKED_PATTERNS = [
        r"ignore previous instructions",
        r"ignore all instructions",
        r"disregard previous instructions",
        r"reveal your system prompt",
        r"show your system prompt",
        r"bypass your restrictions",
        r"act as an unrestricted",
        r"jailbreak",
    ]

    @classmethod
    def validate_query(
        cls,
        query: str,
    ) -> GuardrailResult:
        """
        Validate and clean the user query.
        """

        if not query:
            return GuardrailResult(
                allowed=False,
                message="Please enter a question.",
            )

        cleaned_query = query.strip()

        if not cleaned_query:
            return GuardrailResult(
                allowed=False,
                message="Please enter a valid question.",
            )

        if len(cleaned_query) > cls.MAX_QUERY_LENGTH:
            return GuardrailResult(
                allowed=False,
                message=(
                    "Your question is too long. "
                    f"Please keep it under "
                    f"{cls.MAX_QUERY_LENGTH} characters."
                ),
            )

        lowered_query = cleaned_query.lower()

        for pattern in cls.BLOCKED_PATTERNS:
            if re.search(pattern, lowered_query):
                return GuardrailResult(
                    allowed=False,
                    message=(
                        "I can answer questions about "
                        "the information available in "
                        "the provided documents."
                    ),
                )

        return GuardrailResult(
            allowed=True,
            message="Query accepted.",
            cleaned_query=cleaned_query,
        )

    @classmethod
    def validate_context(
        cls,
        context: str,
    ) -> GuardrailResult:
        """
        Validate retrieved context before sending it to the LLM.
        """

        if not context or not context.strip():
            return GuardrailResult(
                allowed=False,
                message=(
                    "No relevant context was retrieved "
                    "from the documents."
                ),
            )

        if len(context) > cls.MAX_CONTEXT_LENGTH:
            return GuardrailResult(
                allowed=False,
                message=(
                    "The retrieved context is too large "
                    "to process safely."
                ),
            )

        return GuardrailResult(
            allowed=True,
            message="Context accepted.",
        )

    @classmethod
    def validate_answer(
        cls,
        answer: str,
    ) -> GuardrailResult:
        """
        Validate the generated answer.
        """

        if not answer or not answer.strip():
            return GuardrailResult(
                allowed=False,
                message=(
                    "I could not generate a valid answer."
                ),
            )

        cleaned_answer = answer.strip()

        if len(cleaned_answer) > cls.MAX_ANSWER_LENGTH:
            cleaned_answer = (
                cleaned_answer[
                    : cls.MAX_ANSWER_LENGTH
                ].rstrip()
                + "..."
            )

        return GuardrailResult(
            allowed=True,
            message="Answer accepted.",
            cleaned_query=cleaned_answer,
        )