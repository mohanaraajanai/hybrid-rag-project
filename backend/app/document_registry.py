"""
Document Registry

Responsibilities:
- Generate a unique SHA-256 hash for uploaded files.
- Store document processing information.
- Detect duplicate uploads.
- Persist registry data in JSON format.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Project root:
# hybrid-rag-project/
# ├── backend/
# │   └── app/
# └── data/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

REGISTRY_DIRECTORY = PROJECT_ROOT / "data" / "registry"
REGISTRY_FILE = REGISTRY_DIRECTORY / "documents_registry.json"

REGISTRY_DIRECTORY.mkdir(parents=True, exist_ok=True)


def calculate_file_hash(file_content: bytes) -> str:
    """
    Calculate a SHA-256 hash for file content.

    The same file content always produces the same hash.
    Even a small file modification produces a different hash.
    """

    return hashlib.sha256(file_content).hexdigest()


def _read_registry() -> dict[str, Any]:
    """
    Read the registry JSON file.

    Returns an empty registry if the file does not exist
    or contains invalid JSON.
    """

    if not REGISTRY_FILE.exists():
        return {}

    try:
        content = REGISTRY_FILE.read_text(encoding="utf-8")

        if not content.strip():
            return {}

        registry = json.loads(content)

        if not isinstance(registry, dict):
            return {}

        return registry

    except json.JSONDecodeError:
        return {}


def _write_registry(registry: dict[str, Any]) -> None:
    """
    Write registry data to disk.
    """

    temporary_file = REGISTRY_FILE.with_suffix(".tmp")

    temporary_file.write_text(
        json.dumps(
            registry,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Replace the old registry atomically where supported.
    temporary_file.replace(REGISTRY_FILE)


def get_document_by_hash(file_hash: str) -> dict[str, Any] | None:
    """
    Find a previously registered document using its file hash.

    Returns:
        Document metadata if found.
        None if the file has not been registered.
    """

    registry = _read_registry()

    return registry.get(file_hash)


def is_document_registered(file_hash: str) -> bool:
    """
    Check whether a file hash already exists in the registry.
    """

    return get_document_by_hash(file_hash) is not None


def register_document(
    file_hash: str,
    document_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Register a newly processed document.

    If the hash already exists, the existing record is returned
    and no duplicate record is created.
    """

    registry = _read_registry()

    existing_document = registry.get(file_hash)

    if existing_document is not None:
        return existing_document

    registry_record = {
        **document_data,
        "file_hash": file_hash,
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }

    registry[file_hash] = registry_record

    _write_registry(registry)

    return registry_record


def update_document(
    file_hash: str,
    updates: dict[str, Any],
) -> dict[str, Any] | None:
    """
    Update an existing document registry record.

    Returns:
        Updated record if found.
        None if the hash does not exist.
    """

    registry = _read_registry()

    if file_hash not in registry:
        return None

    registry[file_hash].update(updates)
    registry[file_hash]["updated_at"] = datetime.now(
        timezone.utc
    ).isoformat()

    _write_registry(registry)

    return registry[file_hash]


def list_registered_documents() -> list[dict[str, Any]]:
    """
    Return all registered documents as a list.
    """

    registry = _read_registry()

    return list(registry.values())


def remove_document(file_hash: str) -> bool:
    """
    Remove a document from the registry.

    Note:
    This removes only the registry entry.
    It does not delete the PDF, FAISS index, or Neo4j data.
    """

    registry = _read_registry()

    if file_hash not in registry:
        return False

    del registry[file_hash]

    _write_registry(registry)

    return True