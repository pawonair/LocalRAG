"""
Text File Loaders Module
Handles plain text, markdown, JSON, CSV, and XML files.
"""

import json
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any
from io import StringIO

from .base import BaseLoader, LoaderResult, Document


class TextLoader(BaseLoader):
    """Loader for plain text files."""

    SUPPORTED_EXTENSIONS = [".txt"]

    def load(self, file_path: str) -> LoaderResult:
        """Load a plain text file."""
        try:
            path = Path(file_path)
            content = path.read_text(encoding="utf-8")

            documents = [Document(
                page_content=content,
                metadata=self._create_metadata(file_path, char_count=len(content))
            )]

            return LoaderResult(
                success=True,
                documents=documents,
                file_type="text",
                metadata={"char_count": len(content)}
            )

        except UnicodeDecodeError:
            # Try with different encoding
            try:
                content = Path(file_path).read_text(encoding="latin-1")
                documents = [Document(
                    page_content=content,
                    metadata=self._create_metadata(file_path, encoding="latin-1")
                )]
                return LoaderResult(success=True, documents=documents, file_type="text")
            except Exception as e:
                return LoaderResult(
                    success=False,
                    error=f"Failed to decode text file: {str(e)}",
                    file_type="text"
                )
        except Exception as e:
            return LoaderResult(
                success=False,
                error=f"Failed to load text file: {str(e)}",
                file_type="text"
            )

    def load_from_bytes(self, content: bytes, filename: str) -> LoaderResult:
        """Load text from bytes."""
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("latin-1")

        documents = [Document(
            page_content=text,
            metadata={
                "source": filename,
                "filename": filename,
                "extension": Path(filename).suffix.lower(),
            }
        )]

        return LoaderResult(success=True, documents=documents, file_type="text")


class MarkdownLoader(BaseLoader):
    """Loader for Markdown files with structure preservation."""

    SUPPORTED_EXTENSIONS = [".md", ".markdown"]

    def load(self, file_path: str) -> LoaderResult:
        """Load a Markdown file."""
        try:
            path = Path(file_path)
            content = path.read_text(encoding="utf-8")

            # Extract sections based on headers
            sections = self._extract_sections(content)

            documents = []
            for i, section in enumerate(sections):
                documents.append(Document(
                    page_content=section["content"],
                    metadata={
                        **self._create_metadata(file_path),
                        "section": i + 1,
                        "header": section.get("header", ""),
                        "level": section.get("level", 0),
                    }
                ))

            return LoaderResult(
                success=True,
                documents=documents,
                file_type="markdown",
                metadata={"section_count": len(sections)}
            )

        except Exception as e:
            return LoaderResult(
                success=False,
                error=f"Failed to load Markdown file: {str(e)}",
                file_type="markdown"
            )

    def load_from_bytes(self, content: bytes, filename: str) -> LoaderResult:
        """Load Markdown from bytes."""
        try:
            text = content.decode("utf-8")
            sections = self._extract_sections(text)

            documents = []
            for i, section in enumerate(sections):
                documents.append(Document(
                    page_content=section["content"],
                    metadata={
                        "source": filename,
                        "filename": filename,
                        "extension": Path(filename).suffix.lower(),
                        "section": i + 1,
                        "header": section.get("header", ""),
                    }
                ))

            return LoaderResult(success=True, documents=documents, file_type="markdown")

        except Exception as e:
            return LoaderResult(
                success=False,
                error=f"Failed to load Markdown from bytes: {str(e)}",
                file_type="markdown"
            )

    def _extract_sections(self, content: str) -> List[Dict[str, Any]]:
        """Extract sections from Markdown based on headers."""
        lines = content.split("\n")
        sections = []
        current_section = {"header": "", "level": 0, "content": ""}

        for line in lines:
            # Check for headers
            if line.startswith("#"):
                # Save previous section if it has content
                if current_section["content"].strip():
                    sections.append(current_section)

                # Count header level
                level = 0
                for char in line:
                    if char == "#":
                        level += 1
                    else:
                        break

                header = line[level:].strip()
                current_section = {
                    "header": header,
                    "level": level,
                    "content": line + "\n"
                }
            else:
                current_section["content"] += line + "\n"

        # Add last section
        if current_section["content"].strip():
            sections.append(current_section)

        # If no sections found, return whole content as one section
        if not sections:
            sections = [{"header": "", "level": 0, "content": content}]

        return sections


class JSONLoader(BaseLoader):
    """Loader for JSON files with structure preservation."""

    SUPPORTED_EXTENSIONS = [".json"]

    def load(self, file_path: str) -> LoaderResult:
        """Load a JSON file."""
        try:
            path = Path(file_path)
            content = path.read_text(encoding="utf-8")
            data = json.loads(content)

            # Convert JSON to readable text
            documents = self._json_to_documents(data, file_path)

            return LoaderResult(
                success=True,
                documents=documents,
                file_type="json",
                metadata={"structure": type(data).__name__}
            )

        except json.JSONDecodeError as e:
            return LoaderResult(
                success=False,
                error=f"Invalid JSON format: {str(e)}",
                file_type="json"
            )
        except Exception as e:
            return LoaderResult(
                success=False,
                error=f"Failed to load JSON file: {str(e)}",
                file_type="json"
            )

    def load_from_bytes(self, content: bytes, filename: str) -> LoaderResult:
        """Load JSON from bytes."""
        try:
            text = content.decode("utf-8")
            data = json.loads(text)
            documents = self._json_to_documents(data, filename)

            return LoaderResult(success=True, documents=documents, file_type="json")

        except Exception as e:
            return LoaderResult(
                success=False,
                error=f"Failed to load JSON from bytes: {str(e)}",
                file_type="json"
            )

    def _json_to_documents(self, data: Any, source: str) -> List[Document]:
        """Convert JSON data to documents."""
        documents = []

        if isinstance(data, list):
            # Array of items - create document for each
            for i, item in enumerate(data):
                content = json.dumps(item, indent=2, ensure_ascii=False)
                documents.append(Document(
                    page_content=content,
                    metadata={
                        "source": source,
                        "filename": Path(source).name,
                        "extension": ".json",
                        "index": i,
                        "type": type(item).__name__,
                    }
                ))
        elif isinstance(data, dict):
            # Object - create document for each top-level key or whole object
            if len(data) > 10:
                # Many keys - split by key
                for key, value in data.items():
                    content = f"{key}:\n{json.dumps(value, indent=2, ensure_ascii=False)}"
                    documents.append(Document(
                        page_content=content,
                        metadata={
                            "source": source,
                            "filename": Path(source).name,
                            "extension": ".json",
                            "key": key,
                        }
                    ))
            else:
                # Few keys - keep as one document
                content = json.dumps(data, indent=2, ensure_ascii=False)
                documents.append(Document(
                    page_content=content,
                    metadata={
                        "source": source,
                        "filename": Path(source).name,
                        "extension": ".json",
                    }
                ))
        else:
            # Primitive value
            documents.append(Document(
                page_content=str(data),
                metadata={
                    "source": source,
                    "filename": Path(source).name,
                    "extension": ".json",
                }
            ))

        return documents


class CSVLoader(BaseLoader):
    """Loader for CSV files with table awareness."""

    SUPPORTED_EXTENSIONS = [".csv"]

    def load(self, file_path: str) -> LoaderResult:
        """Load a CSV file."""
        try:
            path = Path(file_path)
            content = path.read_text(encoding="utf-8")

            return self._parse_csv(content, file_path)

        except Exception as e:
            return LoaderResult(
                success=False,
                error=f"Failed to load CSV file: {str(e)}",
                file_type="csv"
            )

    def load_from_bytes(self, content: bytes, filename: str) -> LoaderResult:
        """Load CSV from bytes."""
        try:
            text = content.decode("utf-8")
            return self._parse_csv(text, filename)

        except Exception as e:
            return LoaderResult(
                success=False,
                error=f"Failed to load CSV from bytes: {str(e)}",
                file_type="csv"
            )

    def _parse_csv(self, content: str, source: str) -> LoaderResult:
        """Parse CSV content into documents."""
        documents = []

        reader = csv.DictReader(StringIO(content))
        headers = reader.fieldnames or []

        rows = list(reader)

        # Create documents - group rows if too many
        ROWS_PER_DOC = 50

        for i in range(0, len(rows), ROWS_PER_DOC):
            chunk_rows = rows[i:i + ROWS_PER_DOC]

            # Format as readable text
            lines = []
            for row in chunk_rows:
                row_text = " | ".join(f"{k}: {v}" for k, v in row.items())
                lines.append(row_text)

            content_text = f"Headers: {', '.join(headers)}\n\n" + "\n".join(lines)

            documents.append(Document(
                page_content=content_text,
                metadata={
                    "source": source,
                    "filename": Path(source).name,
                    "extension": ".csv",
                    "headers": headers,
                    "row_start": i,
                    "row_end": min(i + ROWS_PER_DOC, len(rows)),
                    "total_rows": len(rows),
                }
            ))

        return LoaderResult(
            success=True,
            documents=documents,
            file_type="csv",
            metadata={"headers": headers, "row_count": len(rows)}
        )


class XMLLoader(BaseLoader):
    """Loader for XML files with structure extraction."""

    SUPPORTED_EXTENSIONS = [".xml"]

    def load(self, file_path: str) -> LoaderResult:
        """Load an XML file."""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            documents = self._element_to_documents(root, file_path)

            return LoaderResult(
                success=True,
                documents=documents,
                file_type="xml",
                metadata={"root_tag": root.tag}
            )

        except ET.ParseError as e:
            return LoaderResult(
                success=False,
                error=f"Invalid XML format: {str(e)}",
                file_type="xml"
            )
        except Exception as e:
            return LoaderResult(
                success=False,
                error=f"Failed to load XML file: {str(e)}",
                file_type="xml"
            )

    def load_from_bytes(self, content: bytes, filename: str) -> LoaderResult:
        """Load XML from bytes."""
        try:
            root = ET.fromstring(content)
            documents = self._element_to_documents(root, filename)

            return LoaderResult(success=True, documents=documents, file_type="xml")

        except Exception as e:
            return LoaderResult(
                success=False,
                error=f"Failed to load XML from bytes: {str(e)}",
                file_type="xml"
            )

    def _element_to_documents(self, root: ET.Element, source: str) -> List[Document]:
        """Convert XML elements to documents."""
        documents = []

        def element_to_text(elem: ET.Element, depth: int = 0) -> str:
            """Convert element and children to readable text."""
            indent = "  " * depth
            lines = []

            # Element tag and attributes
            attrs = " ".join(f'{k}="{v}"' for k, v in elem.attrib.items())
            tag_line = f"{indent}<{elem.tag}" + (f" {attrs}" if attrs else "") + ">"
            lines.append(tag_line)

            # Text content
            if elem.text and elem.text.strip():
                lines.append(f"{indent}  {elem.text.strip()}")

            # Children
            for child in elem:
                lines.append(element_to_text(child, depth + 1))

            lines.append(f"{indent}</{elem.tag}>")

            return "\n".join(lines)

        # If root has many direct children, create doc per child
        if len(root) > 5:
            for i, child in enumerate(root):
                content = element_to_text(child)
                documents.append(Document(
                    page_content=content,
                    metadata={
                        "source": source,
                        "filename": Path(source).name,
                        "extension": ".xml",
                        "tag": child.tag,
                        "index": i,
                    }
                ))
        else:
            # Keep as single document
            content = element_to_text(root)
            documents.append(Document(
                page_content=content,
                metadata={
                    "source": source,
                    "filename": Path(source).name,
                    "extension": ".xml",
                    "root_tag": root.tag,
                }
            ))

        return documents
