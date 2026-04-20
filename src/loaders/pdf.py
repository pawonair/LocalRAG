"""
PDF Loader Module
Handles PDF document loading and extraction.
"""

import tempfile
from pathlib import Path
from typing import List

from .base import BaseLoader, LoaderResult, Document


class PDFLoader(BaseLoader):
    """Loader for PDF documents using PDFPlumber."""

    SUPPORTED_EXTENSIONS = [".pdf"]

    def load(self, file_path: str) -> LoaderResult:
        """Load a PDF document."""
        try:
            from langchain_community.document_loaders import PDFPlumberLoader

            loader = PDFPlumberLoader(file_path)
            docs = loader.load()

            documents = []
            for i, doc in enumerate(docs):
                documents.append(Document(
                    page_content=doc.page_content,
                    metadata={
                        **self._create_metadata(file_path),
                        "page": i + 1,
                        "total_pages": len(docs),
                    }
                ))

            return LoaderResult(
                success=True,
                documents=documents,
                file_type="pdf",
                metadata={"total_pages": len(docs)}
            )

        except Exception as e:
            return LoaderResult(
                success=False,
                error=f"Failed to load PDF: {str(e)}",
                file_type="pdf"
            )

    def load_from_bytes(self, content: bytes, filename: str) -> LoaderResult:
        """Load a PDF from bytes content."""
        try:
            # Write to temporary file
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            # Load from temporary file
            result = self.load(tmp_path)

            # Update metadata with original filename
            for doc in result.documents:
                doc.metadata["source"] = filename
                doc.metadata["filename"] = filename

            # Clean up
            Path(tmp_path).unlink(missing_ok=True)

            return result

        except Exception as e:
            return LoaderResult(
                success=False,
                error=f"Failed to load PDF from bytes: {str(e)}",
                file_type="pdf"
            )
