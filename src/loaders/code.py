"""
Code File Loader Module
Handles source code files with syntax awareness.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import re

from .base import BaseLoader, LoaderResult, Document


# Language configurations
LANGUAGE_CONFIG = {
    ".py": {
        "name": "Python",
        "comment_single": "#",
        "comment_multi_start": '"""',
        "comment_multi_end": '"""',
        "function_pattern": r"^(?:async\s+)?def\s+(\w+)",
        "class_pattern": r"^class\s+(\w+)",
    },
    ".js": {
        "name": "JavaScript",
        "comment_single": "//",
        "comment_multi_start": "/*",
        "comment_multi_end": "*/",
        "function_pattern": r"(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:function|\([^)]*\)\s*=>))",
        "class_pattern": r"^class\s+(\w+)",
    },
    ".ts": {
        "name": "TypeScript",
        "comment_single": "//",
        "comment_multi_start": "/*",
        "comment_multi_end": "*/",
        "function_pattern": r"(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:function|\([^)]*\)\s*=>))",
        "class_pattern": r"^class\s+(\w+)",
    },
    ".jsx": {
        "name": "JavaScript (JSX)",
        "comment_single": "//",
        "comment_multi_start": "/*",
        "comment_multi_end": "*/",
        "function_pattern": r"(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:function|\([^)]*\)\s*=>))",
        "class_pattern": r"^class\s+(\w+)",
    },
    ".tsx": {
        "name": "TypeScript (TSX)",
        "comment_single": "//",
        "comment_multi_start": "/*",
        "comment_multi_end": "*/",
        "function_pattern": r"(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:function|\([^)]*\)\s*=>))",
        "class_pattern": r"^class\s+(\w+)",
    },
    ".java": {
        "name": "Java",
        "comment_single": "//",
        "comment_multi_start": "/*",
        "comment_multi_end": "*/",
        "function_pattern": r"(?:public|private|protected)?\s*(?:static)?\s*\w+\s+(\w+)\s*\(",
        "class_pattern": r"(?:public|private)?\s*class\s+(\w+)",
    },
    ".cpp": {
        "name": "C++",
        "comment_single": "//",
        "comment_multi_start": "/*",
        "comment_multi_end": "*/",
        "function_pattern": r"\w+\s+(\w+)\s*\([^)]*\)\s*{",
        "class_pattern": r"class\s+(\w+)",
    },
    ".c": {
        "name": "C",
        "comment_single": "//",
        "comment_multi_start": "/*",
        "comment_multi_end": "*/",
        "function_pattern": r"\w+\s+(\w+)\s*\([^)]*\)\s*{",
        "class_pattern": None,
    },
    ".go": {
        "name": "Go",
        "comment_single": "//",
        "comment_multi_start": "/*",
        "comment_multi_end": "*/",
        "function_pattern": r"func\s+(?:\([^)]+\)\s+)?(\w+)",
        "class_pattern": r"type\s+(\w+)\s+struct",
    },
    ".rs": {
        "name": "Rust",
        "comment_single": "//",
        "comment_multi_start": "/*",
        "comment_multi_end": "*/",
        "function_pattern": r"(?:pub\s+)?fn\s+(\w+)",
        "class_pattern": r"(?:pub\s+)?struct\s+(\w+)",
    },
    ".rb": {
        "name": "Ruby",
        "comment_single": "#",
        "comment_multi_start": "=begin",
        "comment_multi_end": "=end",
        "function_pattern": r"def\s+(\w+)",
        "class_pattern": r"class\s+(\w+)",
    },
    ".php": {
        "name": "PHP",
        "comment_single": "//",
        "comment_multi_start": "/*",
        "comment_multi_end": "*/",
        "function_pattern": r"function\s+(\w+)",
        "class_pattern": r"class\s+(\w+)",
    },
    ".swift": {
        "name": "Swift",
        "comment_single": "//",
        "comment_multi_start": "/*",
        "comment_multi_end": "*/",
        "function_pattern": r"func\s+(\w+)",
        "class_pattern": r"class\s+(\w+)",
    },
    ".kt": {
        "name": "Kotlin",
        "comment_single": "//",
        "comment_multi_start": "/*",
        "comment_multi_end": "*/",
        "function_pattern": r"fun\s+(\w+)",
        "class_pattern": r"class\s+(\w+)",
    },
    ".scala": {
        "name": "Scala",
        "comment_single": "//",
        "comment_multi_start": "/*",
        "comment_multi_end": "*/",
        "function_pattern": r"def\s+(\w+)",
        "class_pattern": r"class\s+(\w+)",
    },
    ".sql": {
        "name": "SQL",
        "comment_single": "--",
        "comment_multi_start": "/*",
        "comment_multi_end": "*/",
        "function_pattern": r"(?:CREATE\s+)?(?:FUNCTION|PROCEDURE)\s+(\w+)",
        "class_pattern": r"CREATE\s+TABLE\s+(\w+)",
    },
    ".sh": {
        "name": "Shell",
        "comment_single": "#",
        "comment_multi_start": None,
        "comment_multi_end": None,
        "function_pattern": r"(\w+)\s*\(\)\s*{",
        "class_pattern": None,
    },
    ".bash": {
        "name": "Bash",
        "comment_single": "#",
        "comment_multi_start": None,
        "comment_multi_end": None,
        "function_pattern": r"(\w+)\s*\(\)\s*{",
        "class_pattern": None,
    },
    ".yaml": {
        "name": "YAML",
        "comment_single": "#",
        "comment_multi_start": None,
        "comment_multi_end": None,
        "function_pattern": None,
        "class_pattern": None,
    },
    ".yml": {
        "name": "YAML",
        "comment_single": "#",
        "comment_multi_start": None,
        "comment_multi_end": None,
        "function_pattern": None,
        "class_pattern": None,
    },
    ".toml": {
        "name": "TOML",
        "comment_single": "#",
        "comment_multi_start": None,
        "comment_multi_end": None,
        "function_pattern": None,
        "class_pattern": r"\[(\w+)\]",
    },
    ".html": {
        "name": "HTML",
        "comment_single": None,
        "comment_multi_start": "<!--",
        "comment_multi_end": "-->",
        "function_pattern": None,
        "class_pattern": None,
    },
    ".css": {
        "name": "CSS",
        "comment_single": None,
        "comment_multi_start": "/*",
        "comment_multi_end": "*/",
        "function_pattern": None,
        "class_pattern": r"\.(\w+)\s*{",
    },
}

# Default config for unknown extensions
DEFAULT_CONFIG = {
    "name": "Unknown",
    "comment_single": "#",
    "comment_multi_start": None,
    "comment_multi_end": None,
    "function_pattern": None,
    "class_pattern": None,
}


class CodeLoader(BaseLoader):
    """Loader for source code files with syntax awareness."""

    SUPPORTED_EXTENSIONS = list(LANGUAGE_CONFIG.keys()) + [
        ".h", ".hpp", ".zsh", ".ini", ".cfg", ".conf", ".htm"
    ]

    def load(self, file_path: str) -> LoaderResult:
        """Load a code file."""
        try:
            path = Path(file_path)
            content = path.read_text(encoding="utf-8")
            ext = path.suffix.lower()

            config = LANGUAGE_CONFIG.get(ext, DEFAULT_CONFIG)
            language = config["name"]

            # Extract code structure
            structure = self._extract_structure(content, config)

            # Create documents
            documents = self._create_documents(content, structure, file_path, language)

            return LoaderResult(
                success=True,
                documents=documents,
                file_type="code",
                metadata={
                    "language": language,
                    "line_count": len(content.splitlines()),
                    "functions": structure.get("functions", []),
                    "classes": structure.get("classes", []),
                }
            )

        except UnicodeDecodeError:
            try:
                content = Path(file_path).read_text(encoding="latin-1")
                documents = [Document(
                    page_content=content,
                    metadata=self._create_metadata(file_path, encoding="latin-1")
                )]
                return LoaderResult(success=True, documents=documents, file_type="code")
            except Exception as e:
                return LoaderResult(
                    success=False,
                    error=f"Failed to decode code file: {str(e)}",
                    file_type="code"
                )
        except Exception as e:
            return LoaderResult(
                success=False,
                error=f"Failed to load code file: {str(e)}",
                file_type="code"
            )

    def load_from_bytes(self, content: bytes, filename: str) -> LoaderResult:
        """Load code from bytes."""
        try:
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError:
                text = content.decode("latin-1")

            ext = Path(filename).suffix.lower()
            config = LANGUAGE_CONFIG.get(ext, DEFAULT_CONFIG)
            language = config["name"]

            structure = self._extract_structure(text, config)
            documents = self._create_documents(text, structure, filename, language)

            return LoaderResult(
                success=True,
                documents=documents,
                file_type="code",
                metadata={
                    "language": language,
                    "line_count": len(text.splitlines()),
                }
            )

        except Exception as e:
            return LoaderResult(
                success=False,
                error=f"Failed to load code from bytes: {str(e)}",
                file_type="code"
            )

    def _extract_structure(self, content: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract code structure (functions, classes)."""
        structure = {
            "functions": [],
            "classes": [],
        }

        lines = content.splitlines()

        # Extract functions
        if config.get("function_pattern"):
            pattern = re.compile(config["function_pattern"])
            for i, line in enumerate(lines):
                match = pattern.search(line)
                if match:
                    # Get first non-None group
                    name = next((g for g in match.groups() if g), None)
                    if name:
                        structure["functions"].append({
                            "name": name,
                            "line": i + 1,
                        })

        # Extract classes
        if config.get("class_pattern"):
            pattern = re.compile(config["class_pattern"])
            for i, line in enumerate(lines):
                match = pattern.search(line)
                if match:
                    name = match.group(1)
                    structure["classes"].append({
                        "name": name,
                        "line": i + 1,
                    })

        return structure

    def _create_documents(
        self,
        content: str,
        structure: Dict[str, Any],
        source: str,
        language: str
    ) -> List[Document]:
        """Create documents from code content."""
        documents = []
        lines = content.splitlines()
        total_lines = len(lines)

        # For small files, keep as single document
        if total_lines <= 100:
            documents.append(Document(
                page_content=f"```{language.lower()}\n{content}\n```",
                metadata={
                    "source": source,
                    "filename": Path(source).name,
                    "extension": Path(source).suffix.lower(),
                    "language": language,
                    "line_count": total_lines,
                    "functions": [f["name"] for f in structure.get("functions", [])],
                    "classes": [c["name"] for c in structure.get("classes", [])],
                }
            ))
            return documents

        # For larger files, try to split by functions/classes
        # Get all split points (function and class definitions)
        split_points = []

        for func in structure.get("functions", []):
            split_points.append((func["line"], "function", func["name"]))

        for cls in structure.get("classes", []):
            split_points.append((cls["line"], "class", cls["name"]))

        # Sort by line number
        split_points.sort(key=lambda x: x[0])

        if not split_points:
            # No split points found, split by line count
            LINES_PER_CHUNK = 100
            for i in range(0, total_lines, LINES_PER_CHUNK):
                chunk_lines = lines[i:i + LINES_PER_CHUNK]
                chunk_content = "\n".join(chunk_lines)

                documents.append(Document(
                    page_content=f"```{language.lower()}\n{chunk_content}\n```",
                    metadata={
                        "source": source,
                        "filename": Path(source).name,
                        "extension": Path(source).suffix.lower(),
                        "language": language,
                        "line_start": i + 1,
                        "line_end": min(i + LINES_PER_CHUNK, total_lines),
                    }
                ))
        else:
            # Split by definitions
            # First, add header section (before first split point)
            if split_points[0][0] > 1:
                header_lines = lines[:split_points[0][0] - 1]
                if header_lines:
                    header_content = "\n".join(header_lines)
                    documents.append(Document(
                        page_content=f"```{language.lower()}\n{header_content}\n```",
                        metadata={
                            "source": source,
                            "filename": Path(source).name,
                            "extension": Path(source).suffix.lower(),
                            "language": language,
                            "section": "header/imports",
                            "line_start": 1,
                            "line_end": split_points[0][0] - 1,
                        }
                    ))

            # Add each section
            for i, (line_num, def_type, name) in enumerate(split_points):
                # Determine end line
                if i + 1 < len(split_points):
                    end_line = split_points[i + 1][0] - 1
                else:
                    end_line = total_lines

                section_lines = lines[line_num - 1:end_line]
                section_content = "\n".join(section_lines)

                documents.append(Document(
                    page_content=f"```{language.lower()}\n{section_content}\n```",
                    metadata={
                        "source": source,
                        "filename": Path(source).name,
                        "extension": Path(source).suffix.lower(),
                        "language": language,
                        "definition_type": def_type,
                        "definition_name": name,
                        "line_start": line_num,
                        "line_end": end_line,
                    }
                ))

        return documents
