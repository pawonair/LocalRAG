"""
Chat Export
Export conversations to various formats.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from datetime import datetime
from pathlib import Path
import json
import io


class ExportFormat(Enum):
    """Supported export formats."""
    MARKDOWN = "markdown"
    PDF = "pdf"
    JSON = "json"
    HTML = "html"
    TXT = "txt"


@dataclass
class ExportedMessage:
    """Message prepared for export."""
    role: str
    content: str
    timestamp: Optional[datetime] = None
    thinking: Optional[str] = None
    sources: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExportOptions:
    """Options for export."""
    include_thinking: bool = True
    include_sources: bool = True
    include_timestamps: bool = True
    include_metadata: bool = False
    title: str = "LocalRAG Chat Export"
    author: str = ""
    date: Optional[datetime] = None


class ChatExporter:
    """
    Export chat conversations to various formats.
    """

    def __init__(self, options: Optional[ExportOptions] = None):
        """
        Initialize exporter.

        Args:
            options: Export options
        """
        self.options = options or ExportOptions()

    def export(
        self,
        messages: List[ExportedMessage],
        format: ExportFormat,
        output_path: Optional[Path] = None
    ) -> Union[str, bytes]:
        """
        Export messages to specified format.

        Args:
            messages: Messages to export
            format: Export format
            output_path: Optional file path to save

        Returns:
            Exported content (string or bytes)
        """
        exporters = {
            ExportFormat.MARKDOWN: self._to_markdown,
            ExportFormat.PDF: self._to_pdf,
            ExportFormat.JSON: self._to_json,
            ExportFormat.HTML: self._to_html,
            ExportFormat.TXT: self._to_txt,
        }

        exporter = exporters.get(format)
        if not exporter:
            raise ValueError(f"Unsupported format: {format}")

        content = exporter(messages)

        if output_path:
            mode = "wb" if isinstance(content, bytes) else "w"
            with open(output_path, mode) as f:
                f.write(content)

        return content

    def _to_markdown(self, messages: List[ExportedMessage]) -> str:
        """Export to Markdown format."""
        lines = []

        # Header
        lines.append(f"# {self.options.title}")
        lines.append("")

        if self.options.date:
            lines.append(f"**Date:** {self.options.date.strftime('%Y-%m-%d %H:%M')}")
        else:
            lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")

        if self.options.author:
            lines.append(f"**Author:** {self.options.author}")

        lines.append(f"**Messages:** {len(messages)}")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Messages
        for i, msg in enumerate(messages, 1):
            # Role header
            role_icon = "👤" if msg.role == "user" else "🤖"
            role_name = "User" if msg.role == "user" else "Assistant"

            lines.append(f"### {role_icon} {role_name}")

            # Timestamp
            if self.options.include_timestamps and msg.timestamp:
                lines.append(f"*{msg.timestamp.strftime('%H:%M:%S')}*")
                lines.append("")

            # Thinking section
            if self.options.include_thinking and msg.thinking:
                lines.append("<details>")
                lines.append("<summary>💭 Reasoning Process</summary>")
                lines.append("")
                lines.append(msg.thinking)
                lines.append("")
                lines.append("</details>")
                lines.append("")

            # Content
            lines.append(msg.content)
            lines.append("")

            # Sources
            if self.options.include_sources and msg.sources:
                lines.append("**Sources:**")
                for j, source in enumerate(msg.sources, 1):
                    source_name = source.get("source", f"Source {j}")
                    lines.append(f"- [{j}] {source_name}")
                lines.append("")

            lines.append("---")
            lines.append("")

        # Footer
        lines.append("")
        lines.append("*Exported from LocalRAG*")

        return "\n".join(lines)

    def _to_pdf(self, messages: List[ExportedMessage]) -> bytes:
        """Export to PDF format."""
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            )

            buffer = io.BytesIO()
            doc = SimpleDocTemplate(
                buffer,
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )

            styles = getSampleStyleSheet()

            # Custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30
            )

            user_style = ParagraphStyle(
                'UserMessage',
                parent=styles['Normal'],
                fontSize=11,
                leftIndent=20,
                spaceAfter=10,
                backColor=colors.HexColor('#E3F2FD')
            )

            assistant_style = ParagraphStyle(
                'AssistantMessage',
                parent=styles['Normal'],
                fontSize=11,
                leftIndent=20,
                spaceAfter=10,
                backColor=colors.HexColor('#F5F5F5')
            )

            thinking_style = ParagraphStyle(
                'ThinkingMessage',
                parent=styles['Normal'],
                fontSize=9,
                leftIndent=40,
                textColor=colors.gray,
                spaceAfter=5
            )

            story = []

            # Title
            story.append(Paragraph(self.options.title, title_style))

            # Metadata
            date_str = (self.options.date or datetime.now()).strftime('%Y-%m-%d %H:%M')
            story.append(Paragraph(f"<b>Date:</b> {date_str}", styles['Normal']))
            story.append(Paragraph(f"<b>Messages:</b> {len(messages)}", styles['Normal']))
            story.append(Spacer(1, 20))

            # Messages
            for msg in messages:
                role_name = "User" if msg.role == "user" else "Assistant"
                style = user_style if msg.role == "user" else assistant_style

                # Role header
                story.append(Paragraph(f"<b>{role_name}:</b>", styles['Heading3']))

                # Thinking
                if self.options.include_thinking and msg.thinking:
                    thinking_text = msg.thinking[:500] + "..." if len(msg.thinking) > 500 else msg.thinking
                    story.append(Paragraph(f"<i>Thinking: {thinking_text}</i>", thinking_style))

                # Content - escape HTML
                content = msg.content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                content = content.replace('\n', '<br/>')
                story.append(Paragraph(content, style))

                # Sources
                if self.options.include_sources and msg.sources:
                    sources_text = ", ".join([
                        s.get("source", f"Source {i}")
                        for i, s in enumerate(msg.sources, 1)
                    ])
                    story.append(Paragraph(f"<i>Sources: {sources_text}</i>", styles['Normal']))

                story.append(Spacer(1, 15))

            # Footer
            story.append(Spacer(1, 30))
            story.append(Paragraph("<i>Exported from LocalRAG</i>", styles['Normal']))

            doc.build(story)
            return buffer.getvalue()

        except ImportError:
            # Fallback: return markdown as bytes
            markdown = self._to_markdown(messages)
            return markdown.encode('utf-8')

    def _to_json(self, messages: List[ExportedMessage]) -> str:
        """Export to JSON format."""
        data = {
            "title": self.options.title,
            "exported_at": datetime.now().isoformat(),
            "author": self.options.author,
            "message_count": len(messages),
            "messages": []
        }

        for msg in messages:
            msg_data = {
                "role": msg.role,
                "content": msg.content,
            }

            if self.options.include_timestamps and msg.timestamp:
                msg_data["timestamp"] = msg.timestamp.isoformat()

            if self.options.include_thinking and msg.thinking:
                msg_data["thinking"] = msg.thinking

            if self.options.include_sources and msg.sources:
                msg_data["sources"] = msg.sources

            if self.options.include_metadata and msg.metadata:
                msg_data["metadata"] = msg.metadata

            data["messages"].append(msg_data)

        return json.dumps(data, indent=2, ensure_ascii=False)

    def _to_html(self, messages: List[ExportedMessage]) -> str:
        """Export to HTML format."""
        lines = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            f"<title>{self.options.title}</title>",
            "<style>",
            "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }",
            ".message { margin: 15px 0; padding: 15px; border-radius: 10px; }",
            ".user { background: #E3F2FD; }",
            ".assistant { background: #F5F5F5; }",
            ".role { font-weight: bold; margin-bottom: 10px; }",
            ".thinking { color: #666; font-style: italic; font-size: 0.9em; margin: 10px 0; padding: 10px; background: #FFF9C4; border-radius: 5px; }",
            ".sources { font-size: 0.85em; color: #555; margin-top: 10px; }",
            ".timestamp { font-size: 0.8em; color: #999; }",
            "details { margin: 10px 0; }",
            "summary { cursor: pointer; color: #666; }",
            "</style>",
            "</head>",
            "<body>",
            f"<h1>{self.options.title}</h1>",
        ]

        date_str = (self.options.date or datetime.now()).strftime('%Y-%m-%d %H:%M')
        lines.append(f"<p><strong>Date:</strong> {date_str}</p>")
        lines.append(f"<p><strong>Messages:</strong> {len(messages)}</p>")
        lines.append("<hr>")

        for msg in messages:
            role_class = "user" if msg.role == "user" else "assistant"
            role_icon = "👤" if msg.role == "user" else "🤖"
            role_name = "User" if msg.role == "user" else "Assistant"

            lines.append(f'<div class="message {role_class}">')
            lines.append(f'<div class="role">{role_icon} {role_name}</div>')

            if self.options.include_timestamps and msg.timestamp:
                lines.append(f'<div class="timestamp">{msg.timestamp.strftime("%H:%M:%S")}</div>')

            if self.options.include_thinking and msg.thinking:
                lines.append("<details>")
                lines.append("<summary>💭 Reasoning Process</summary>")
                lines.append(f'<div class="thinking">{msg.thinking}</div>')
                lines.append("</details>")

            # Escape HTML and preserve newlines
            content = msg.content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            content = content.replace('\n', '<br>')
            lines.append(f"<div>{content}</div>")

            if self.options.include_sources and msg.sources:
                sources_html = ", ".join([
                    s.get("source", f"Source {i}")
                    for i, s in enumerate(msg.sources, 1)
                ])
                lines.append(f'<div class="sources">📚 Sources: {sources_html}</div>')

            lines.append("</div>")

        lines.extend([
            "<hr>",
            "<p><em>Exported from LocalRAG</em></p>",
            "</body>",
            "</html>"
        ])

        return "\n".join(lines)

    def _to_txt(self, messages: List[ExportedMessage]) -> str:
        """Export to plain text format."""
        lines = []

        lines.append(f"{self.options.title}")
        lines.append("=" * len(self.options.title))
        lines.append("")

        date_str = (self.options.date or datetime.now()).strftime('%Y-%m-%d %H:%M')
        lines.append(f"Date: {date_str}")
        lines.append(f"Messages: {len(messages)}")
        lines.append("")
        lines.append("-" * 50)
        lines.append("")

        for msg in messages:
            role_name = "USER" if msg.role == "user" else "ASSISTANT"

            lines.append(f"[{role_name}]")

            if self.options.include_timestamps and msg.timestamp:
                lines.append(f"Time: {msg.timestamp.strftime('%H:%M:%S')}")

            lines.append("")
            lines.append(msg.content)
            lines.append("")

            if self.options.include_sources and msg.sources:
                lines.append("Sources:")
                for i, source in enumerate(msg.sources, 1):
                    lines.append(f"  [{i}] {source.get('source', 'Unknown')}")
                lines.append("")

            lines.append("-" * 50)
            lines.append("")

        lines.append("Exported from LocalRAG")

        return "\n".join(lines)


def export_to_markdown(
    messages: List[ExportedMessage],
    title: str = "Chat Export",
    include_thinking: bool = True
) -> str:
    """Quick export to Markdown."""
    options = ExportOptions(
        title=title,
        include_thinking=include_thinking
    )
    exporter = ChatExporter(options)
    return exporter.export(messages, ExportFormat.MARKDOWN)


def export_to_pdf(
    messages: List[ExportedMessage],
    title: str = "Chat Export",
    output_path: Optional[Path] = None
) -> bytes:
    """Quick export to PDF."""
    options = ExportOptions(title=title)
    exporter = ChatExporter(options)
    return exporter.export(messages, ExportFormat.PDF, output_path)


def export_to_json(
    messages: List[ExportedMessage],
    include_metadata: bool = False
) -> str:
    """Quick export to JSON."""
    options = ExportOptions(include_metadata=include_metadata)
    exporter = ChatExporter(options)
    return exporter.export(messages, ExportFormat.JSON)
