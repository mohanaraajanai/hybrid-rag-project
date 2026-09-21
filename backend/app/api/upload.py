from fastapi import APIRouter, File, HTTPException, UploadFile

from app.ingestion.pipeline import process_uploaded_pdf
from app.ingestion.document_processing_service import (
    DocumentProcessingService,
)


router = APIRouter(
    prefix="/api/upload",
    tags=["Document Upload"],
)


@router.post("/pdf")
async def upload_pdf(
    file: UploadFile = File(...),
):
    """
    Upload and process a PDF document through the complete
    Hybrid RAG ingestion pipeline.

    Workflow:
        1. PDF validation
        2. File hash calculation
        3. Duplicate detection
        4. PDF extraction
        5. Registry persistence
        6. Text chunking
        7. Local embedding generation
        8. FAISS vector storage
        9. Neo4j document graph storage
        10. Entity and relationship extraction
        11. Neo4j entity graph storage
    """

    processing_service = None

    try:
        # ---------------------------------------------------------
        # Step 1: Existing upload and extraction pipeline
        # ---------------------------------------------------------
        upload_result = await process_uploaded_pdf(file)

        # ---------------------------------------------------------
        # Step 2: Handle duplicate documents
        # ---------------------------------------------------------
        if upload_result.get("duplicate", False):
            return {
                "status": "duplicate",
                "message": upload_result["message"],
                "document": upload_result["document"],
                "extraction_summary": upload_result[
                    "extraction_summary"
                ],
                "processed_file": upload_result[
                    "processed_file"
                ],
                "duplicate": True,
                "processing": {
                    "status": "skipped",
                    "message": (
                        "Document already exists. "
                        "Downstream processing was skipped."
                    ),
                },
            }

        # ---------------------------------------------------------
        # Step 3: Run complete Hybrid RAG processing
        # ---------------------------------------------------------
        processing_service = DocumentProcessingService(
            vector_index_name="documents"
        )

        processing_service.initialize_graph_constraints()

        processing_result = (
            processing_service.process_document(
                processed_file=upload_result[
                    "processed_file"
                ]
            )
        )

        # ---------------------------------------------------------
        # Step 4: Return combined result
        # ---------------------------------------------------------
        return {
            "status": "success",
            "message": (
                "PDF uploaded and processed successfully "
                "through the complete Hybrid RAG pipeline."
            ),
            "document": upload_result["document"],
            "extraction_summary": upload_result[
                "extraction_summary"
            ],
            "processed_file": upload_result[
                "processed_file"
            ],
            "duplicate": False,
            "processing": processing_result,
        }

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"PDF processing failed: {exc}",
        ) from exc

    finally:
        if processing_service is not None:
            processing_service.close()