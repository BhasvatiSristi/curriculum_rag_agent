"""
generator.py

Generate answers using the Groq LLM.
"""

import logging
from typing import Dict, List

from groq import Groq

from backend.core.settings import (
    GROQ_API_KEY,
    LLM_MODEL,
)

logger = logging.getLogger(__name__)

client = Groq(api_key=GROQ_API_KEY)

NO_ANSWER = (
    "I could not find supporting information in the curriculum documents."
)

SYSTEM_PROMPT = """
You are an AI assistant for the IIITDM Kancheepuram Curriculum Assistant.

You answer questions ONLY using the retrieved curriculum context.

Rules:
1. Use only the provided context.
2. Never make up information.
3. If the answer is not found in the context, respond exactly with:
   "I could not find supporting information in the curriculum documents."
4. Keep answers clear and concise.
5. When listing courses always include:
   • Course Code
   • Course Name
   • Credits
6. Format answers using bullet points whenever appropriate.
"""


def build_prompt(
    question: str,
    context_chunks: List[Dict],
) -> str:
    """
    Build the prompt using the retrieved context.
    """

    context = "\n\n".join(
        [
            (
                f"Source: {chunk['source']} | "
                f"Page: {chunk['page']}\n"
                f"{chunk['text']}"
            )
            for chunk in context_chunks
        ]
    )

    return f"""
Context:

{context}

----------------------------------------

Question:
{question}

Answer:
"""


def generate_answer(
    question: str,
    context_chunks: List[Dict],
) -> str:
    """
    Generate an answer using the Groq LLM.

    Args:
        question: User question.
        context_chunks: Retrieved chunks from the retriever.

    Returns:
        Generated answer.
    """

    if not context_chunks:
        return NO_ANSWER

    prompt = build_prompt(
        question=question,
        context_chunks=context_chunks,
    )

    try:

        response = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0.1,
            max_tokens=1024,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        answer = response.choices[0].message.content.strip()

        if not answer:
            return NO_ANSWER

        return answer

    except Exception as e:

        logger.exception("Failed to generate response.")

        return f"Generation Error: {e}"