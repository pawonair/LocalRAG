"""
Batch Processing CLI
Command-line interface for batch document processing.
"""

import argparse
import sys
import os
import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loaders.router import DocumentRouter
from rag.vectorstore import VectorStoreManager
from rag.pipeline import AdvancedRAGPipeline, RAGConfig
from llm.ollama import OllamaClient
from db.database import get_database

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain.schema import Document as LangchainDocument


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    input_dir: str
    collection_id: str = "default"
    recursive: bool = False
    file_types: List[str] = field(default_factory=lambda: [
        ".pdf", ".txt", ".md", ".docx", ".pptx", ".xlsx", ".csv", ".json"
    ])
    max_workers: int = 4
    verbose: bool = False
    dry_run: bool = False
    output_file: Optional[str] = None


@dataclass
class ProcessingResult:
    """Result of processing a single file."""
    filepath: str
    success: bool
    chunks: int = 0
    error: Optional[str] = None
    processing_time: float = 0.0


class BatchProcessor:
    """
    Batch document processor for LocalRAG.
    Processes multiple documents from a directory.
    """

    def __init__(self, config: BatchConfig):
        """
        Initialize batch processor.

        Args:
            config: Batch processing configuration
        """
        self.config = config
        self.doc_router = DocumentRouter()
        self.vector_manager = VectorStoreManager()
        self.db = get_database()
        self.ollama_client = OllamaClient()
        self.pipeline = AdvancedRAGPipeline(
            vector_store_manager=self.vector_manager,
            llm_func=lambda p: self.ollama_client.generate(p),
            config=RAGConfig()
        )
        self.embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.splitter = SemanticChunker(self.embedder)
        self.results: List[ProcessingResult] = []

    def discover_files(self) -> List[Path]:
        """
        Discover files to process in the input directory.

        Returns:
            List of file paths to process
        """
        input_path = Path(self.config.input_dir)

        if not input_path.exists():
            raise ValueError(f"Input directory does not exist: {input_path}")

        if not input_path.is_dir():
            raise ValueError(f"Input path is not a directory: {input_path}")

        files = []
        pattern = "**/*" if self.config.recursive else "*"

        for file_path in input_path.glob(pattern):
            if file_path.is_file():
                if file_path.suffix.lower() in self.config.file_types:
                    files.append(file_path)

        return sorted(files)

    def process_file(self, filepath: Path) -> ProcessingResult:
        """
        Process a single file.

        Args:
            filepath: Path to the file to process

        Returns:
            Processing result
        """
        start_time = time.time()

        try:
            if self.config.dry_run:
                return ProcessingResult(
                    filepath=str(filepath),
                    success=True,
                    chunks=0,
                    processing_time=time.time() - start_time
                )

            # Load document
            result = self.doc_router.load(str(filepath))

            if not result.success:
                return ProcessingResult(
                    filepath=str(filepath),
                    success=False,
                    error=result.error,
                    processing_time=time.time() - start_time
                )

            # Convert to LangChain documents
            langchain_docs = [
                LangchainDocument(page_content=d.page_content, metadata=d.metadata)
                for d in result.documents
            ]

            # Split into chunks
            documents = self.splitter.split_documents(langchain_docs)

            # Add to vector store
            self.vector_manager.add_documents(self.config.collection_id, documents)

            # Add to BM25 index
            self.pipeline.add_to_bm25_index(self.config.collection_id, documents)

            # Store in database
            with open(filepath, "rb") as f:
                content = f.read()

            self.db.add_document(
                filename=filepath.name,
                file_type=result.file_type,
                content=content,
                collection_id=self.config.collection_id,
                chunk_count=len(documents),
                metadata=result.metadata
            )

            return ProcessingResult(
                filepath=str(filepath),
                success=True,
                chunks=len(documents),
                processing_time=time.time() - start_time
            )

        except Exception as e:
            return ProcessingResult(
                filepath=str(filepath),
                success=False,
                error=str(e),
                processing_time=time.time() - start_time
            )

    def process_all(self) -> List[ProcessingResult]:
        """
        Process all discovered files.

        Returns:
            List of processing results
        """
        files = self.discover_files()

        if not files:
            print("No files found to process.")
            return []

        print(f"Found {len(files)} files to process")

        if self.config.verbose:
            for f in files:
                print(f"  - {f}")

        self.results = []

        if self.config.max_workers > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {
                    executor.submit(self.process_file, f): f
                    for f in files
                }

                for future in as_completed(futures):
                    result = future.result()
                    self.results.append(result)
                    self._print_result(result)
        else:
            # Sequential processing
            for filepath in files:
                result = self.process_file(filepath)
                self.results.append(result)
                self._print_result(result)

        return self.results

    def _print_result(self, result: ProcessingResult) -> None:
        """Print a single processing result."""
        if result.success:
            status = "✓" if not self.config.dry_run else "○"
            print(f"{status} {result.filepath} ({result.chunks} chunks, {result.processing_time:.2f}s)")
        else:
            print(f"✗ {result.filepath}: {result.error}")

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of batch processing.

        Returns:
            Summary dictionary
        """
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]

        total_chunks = sum(r.chunks for r in successful)
        total_time = sum(r.processing_time for r in self.results)

        return {
            "total_files": len(self.results),
            "successful": len(successful),
            "failed": len(failed),
            "total_chunks": total_chunks,
            "total_time": total_time,
            "collection_id": self.config.collection_id,
            "dry_run": self.config.dry_run,
            "failures": [
                {"file": r.filepath, "error": r.error}
                for r in failed
            ]
        }

    def save_report(self, output_path: Optional[str] = None) -> str:
        """
        Save processing report to file.

        Args:
            output_path: Output file path (defaults to config.output_file)

        Returns:
            Path to saved report
        """
        path = output_path or self.config.output_file

        if not path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"batch_report_{timestamp}.json"

        summary = self.get_summary()
        summary["results"] = [
            {
                "filepath": r.filepath,
                "success": r.success,
                "chunks": r.chunks,
                "error": r.error,
                "processing_time": r.processing_time
            }
            for r in self.results
        ]
        summary["timestamp"] = datetime.now().isoformat()

        with open(path, "w") as f:
            json.dump(summary, f, indent=2)

        return path


def print_summary(summary: Dict[str, Any]) -> None:
    """Print batch processing summary."""
    print("\n" + "=" * 50)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 50)

    if summary["dry_run"]:
        print("MODE: Dry run (no files were actually processed)")

    print(f"Collection: {summary['collection_id']}")
    print(f"Total files: {summary['total_files']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"Total chunks: {summary['total_chunks']}")
    print(f"Total time: {summary['total_time']:.2f}s")

    if summary["failures"]:
        print("\nFailed files:")
        for failure in summary["failures"]:
            print(f"  - {failure['file']}: {failure['error']}")


def main():
    """Main entry point for batch processing CLI."""
    parser = argparse.ArgumentParser(
        description="LocalRAG Batch Document Processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/documents
  %(prog)s /path/to/documents --collection my-docs --recursive
  %(prog)s /path/to/documents --dry-run --verbose
  %(prog)s /path/to/documents --workers 8 --output report.json
        """
    )

    parser.add_argument(
        "input_dir",
        help="Directory containing documents to process"
    )

    parser.add_argument(
        "-c", "--collection",
        default="default",
        help="Collection ID to store documents in (default: default)"
    )

    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Recursively process subdirectories"
    )

    parser.add_argument(
        "-t", "--types",
        nargs="+",
        default=[".pdf", ".txt", ".md", ".docx", ".pptx", ".xlsx", ".csv", ".json"],
        help="File types to process (default: pdf, txt, md, docx, pptx, xlsx, csv, json)"
    )

    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )

    parser.add_argument(
        "-n", "--dry-run",
        action="store_true",
        help="List files without processing"
    )

    parser.add_argument(
        "-o", "--output",
        help="Output file for processing report (JSON)"
    )

    args = parser.parse_args()

    # Create config
    config = BatchConfig(
        input_dir=args.input_dir,
        collection_id=args.collection,
        recursive=args.recursive,
        file_types=args.types,
        max_workers=args.workers,
        verbose=args.verbose,
        dry_run=args.dry_run,
        output_file=args.output
    )

    try:
        # Run batch processing
        processor = BatchProcessor(config)
        processor.process_all()

        # Print summary
        summary = processor.get_summary()
        print_summary(summary)

        # Save report if requested
        if config.output_file:
            report_path = processor.save_report()
            print(f"\nReport saved to: {report_path}")

        # Exit with error code if any failures
        if summary["failed"] > 0:
            sys.exit(1)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
