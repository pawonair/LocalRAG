"""
Web Content Loader Module
Handles URL ingestion and web page scraping.
"""

import re
from typing import List, Optional
from urllib.parse import urlparse, urljoin

from .base import BaseLoader, LoaderResult, Document


class WebLoader(BaseLoader):
    """Loader for web content from URLs."""

    SUPPORTED_EXTENSIONS = []  # URLs don't have extensions in the traditional sense

    def __init__(self, timeout: int = 30):
        """
        Initialize web loader.

        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout

    def load(self, url: str) -> LoaderResult:
        """
        Load content from a URL.

        Args:
            url: The URL to fetch

        Returns:
            LoaderResult with extracted content
        """
        try:
            import requests
            from bs4 import BeautifulSoup

            # Validate URL
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return LoaderResult(
                    success=False,
                    error=f"Invalid URL: {url}",
                    file_type="web"
                )

            # Fetch content
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; LocalRAG/1.0; +https://github.com/localrag)"
            }

            response = requests.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()

            # Parse HTML
            soup = BeautifulSoup(response.text, "html.parser")

            # Remove script and style elements
            for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
                element.decompose()

            # Extract title
            title = soup.title.string if soup.title else parsed.netloc

            # Extract main content
            # Try to find main content area
            main_content = (
                soup.find("main") or
                soup.find("article") or
                soup.find("div", {"class": re.compile(r"content|main|article", re.I)}) or
                soup.find("div", {"id": re.compile(r"content|main|article", re.I)}) or
                soup.body
            )

            if not main_content:
                main_content = soup

            # Extract text
            text = self._extract_text(main_content)

            # Extract metadata
            meta_description = ""
            meta_tag = soup.find("meta", {"name": "description"})
            if meta_tag:
                meta_description = meta_tag.get("content", "")

            documents = [Document(
                page_content=text,
                metadata={
                    "source": url,
                    "title": title,
                    "description": meta_description,
                    "domain": parsed.netloc,
                    "type": "webpage",
                }
            )]

            return LoaderResult(
                success=True,
                documents=documents,
                file_type="web",
                metadata={
                    "title": title,
                    "url": url,
                    "domain": parsed.netloc,
                }
            )

        except ImportError as e:
            missing = "requests" if "requests" in str(e) else "beautifulsoup4"
            return LoaderResult(
                success=False,
                error=f"{missing} is required for web loading. Install with: pip install {missing}",
                file_type="web"
            )
        except requests.exceptions.Timeout:
            return LoaderResult(
                success=False,
                error=f"Request timed out after {self.timeout} seconds",
                file_type="web"
            )
        except requests.exceptions.RequestException as e:
            return LoaderResult(
                success=False,
                error=f"Failed to fetch URL: {str(e)}",
                file_type="web"
            )
        except Exception as e:
            return LoaderResult(
                success=False,
                error=f"Failed to load web content: {str(e)}",
                file_type="web"
            )

    def load_from_bytes(self, content: bytes, filename: str) -> LoaderResult:
        """
        Load web content from bytes (HTML content).

        Args:
            content: HTML content as bytes
            filename: URL or filename for metadata

        Returns:
            LoaderResult with extracted content
        """
        try:
            from bs4 import BeautifulSoup

            html = content.decode("utf-8")
            soup = BeautifulSoup(html, "html.parser")

            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
                element.decompose()

            title = soup.title.string if soup.title else filename

            main_content = (
                soup.find("main") or
                soup.find("article") or
                soup.body or
                soup
            )

            text = self._extract_text(main_content)

            documents = [Document(
                page_content=text,
                metadata={
                    "source": filename,
                    "title": title,
                    "type": "html",
                }
            )]

            return LoaderResult(
                success=True,
                documents=documents,
                file_type="web"
            )

        except Exception as e:
            return LoaderResult(
                success=False,
                error=f"Failed to parse HTML content: {str(e)}",
                file_type="web"
            )

    def _extract_text(self, element) -> str:
        """
        Extract clean text from a BeautifulSoup element.

        Args:
            element: BeautifulSoup element

        Returns:
            Cleaned text content
        """
        # Get text with some structure preservation
        texts = []

        for elem in element.descendants:
            if elem.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                level = int(elem.name[1])
                prefix = "#" * level
                texts.append(f"\n{prefix} {elem.get_text(strip=True)}\n")
            elif elem.name == "p":
                text = elem.get_text(strip=True)
                if text:
                    texts.append(f"\n{text}\n")
            elif elem.name == "li":
                text = elem.get_text(strip=True)
                if text:
                    texts.append(f"• {text}")
            elif elem.name == "br":
                texts.append("\n")
            elif elem.name in ["code", "pre"]:
                text = elem.get_text(strip=True)
                if text:
                    texts.append(f"\n```\n{text}\n```\n")

        # If no structured text found, fall back to simple extraction
        if not texts:
            return element.get_text(separator="\n", strip=True)

        # Clean up result
        result = "".join(texts)
        # Remove excessive newlines
        result = re.sub(r"\n{3,}", "\n\n", result)

        return result.strip()

    def is_valid_url(self, url: str) -> bool:
        """Check if a string is a valid URL."""
        try:
            parsed = urlparse(url)
            return bool(parsed.scheme and parsed.netloc)
        except Exception:
            return False
