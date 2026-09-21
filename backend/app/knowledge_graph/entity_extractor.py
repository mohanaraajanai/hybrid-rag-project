import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


class EntityExtractor:
    """
    Extract entities and relationships from document text using an LLM.
    """

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("OPENAI_API_KEY is missing in the .env file.")

        self.client = OpenAI(api_key=api_key)

        self.model = os.getenv(
            "OPENAI_MODEL",
            "gpt-4o-mini",
        )

    def extract(self, text: str) -> dict[str, Any]:
        """
        Extract entities and relationships from a text chunk.
        """

        if not text or not text.strip():
            return {
                "entities": [],
                "relationships": [],
            }

        system_prompt = """
You are an information extraction system.

Extract entities and relationships from the supplied document text.

Return ONLY valid JSON using this exact structure:

{
  "entities": [
    {
      "name": "Entity name",
      "type": "Technology|Framework|Database|Organization|Product|Concept|Other"
    }
  ],
  "relationships": [
    {
      "source": "Source entity name",
      "relationship": "USES|PART_OF|PROVIDES|DEPENDS_ON|RELATED_TO|OTHER",
      "target": "Target entity name"
    }
  ]
}

Rules:
1. Extract only entities explicitly mentioned in the text.
2. Do not invent information.
3. Avoid duplicate entities.
4. Keep entity names concise.
5. Relationships must connect entities found in the entities list.
6. If nothing is found, return empty arrays.
"""

        user_prompt = f"""
Extract entities and relationships from this document text:

--- BEGIN TEXT ---
{text}
--- END TEXT ---
"""

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
        )

        content = response.choices[0].message.content

        if not content:
            return {
                "entities": [],
                "relationships": [],
            }

        try:
            extracted_data = json.loads(content)
        except json.JSONDecodeError:
            return {
                "entities": [],
                "relationships": [],
            }

        return self._validate_result(extracted_data)

    def _validate_result(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Validate and normalize the LLM response.
        """

        valid_entities = []
        valid_relationships = []

        entities = data.get("entities", [])
        relationships = data.get("relationships", [])

        entity_names = set()

        for entity in entities:
            if not isinstance(entity, dict):
                continue

            name = str(entity.get("name", "")).strip()
            entity_type = str(entity.get("type", "Other")).strip()

            if not name:
                continue

            normalized_name = name.lower()

            if normalized_name in entity_names:
                continue

            entity_names.add(normalized_name)

            valid_entities.append(
                {
                    "name": name,
                    "type": entity_type or "Other",
                }
            )

        for relationship in relationships:
            if not isinstance(relationship, dict):
                continue

            source = str(relationship.get("source", "")).strip()
            relation_type = str(
                relationship.get("relationship", "RELATED_TO")
            ).strip()
            target = str(relationship.get("target", "")).strip()

            if not source or not target:
                continue

            if (
                source.lower() not in entity_names
                or target.lower() not in entity_names
            ):
                continue

            valid_relationships.append(
                {
                    "source": source,
                    "relationship": relation_type or "RELATED_TO",
                    "target": target,
                }
            )

        return {
            "entities": valid_entities,
            "relationships": valid_relationships,
        }