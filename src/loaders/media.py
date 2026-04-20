"""
Media Loaders Module
Handles image, audio, and video file processing.
"""

import tempfile
import base64
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import io

from .base import BaseLoader, LoaderResult, Document


class ImageLoader(BaseLoader):
    """
    Loader for image files.
    Supports OCR text extraction and vision model descriptions.
    """

    SUPPORTED_EXTENSIONS = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff", ".tif"]

    def __init__(self, use_ocr: bool = True, use_vision: bool = True, vision_model: str = "llava:7b"):
        """
        Initialize image loader.

        Args:
            use_ocr: Whether to extract text using OCR
            use_vision: Whether to use vision model for descriptions
            vision_model: Ollama vision model to use
        """
        self.use_ocr = use_ocr
        self.use_vision = use_vision
        self.vision_model = vision_model

    def load(self, file_path: str) -> LoaderResult:
        """Load an image file."""
        try:
            from PIL import Image

            path = Path(file_path)
            img = Image.open(file_path)

            documents = []
            extracted_text = []

            # Get basic image info
            width, height = img.size
            mode = img.mode
            format_type = img.format or path.suffix.upper().replace(".", "")

            metadata = {
                **self._create_metadata(file_path),
                "width": width,
                "height": height,
                "mode": mode,
                "format": format_type,
            }

            # OCR extraction
            if self.use_ocr:
                ocr_text = self._extract_ocr(img)
                if ocr_text:
                    extracted_text.append(f"[OCR Text]\n{ocr_text}")
                    metadata["has_ocr"] = True

            # Vision model description
            if self.use_vision:
                with open(file_path, "rb") as f:
                    image_bytes = f.read()
                description = self._get_vision_description(image_bytes)
                if description:
                    extracted_text.append(f"[Image Description]\n{description}")
                    metadata["has_vision_description"] = True

            # Combine all extracted content
            if extracted_text:
                content = f"Image: {path.name}\nDimensions: {width}x{height}\n\n"
                content += "\n\n".join(extracted_text)

                documents.append(Document(
                    page_content=content,
                    metadata=metadata
                ))
            else:
                # At minimum, store image metadata
                documents.append(Document(
                    page_content=f"Image: {path.name}\nDimensions: {width}x{height}\nFormat: {format_type}",
                    metadata=metadata
                ))

            return LoaderResult(
                success=True,
                documents=documents,
                file_type="image",
                metadata=metadata
            )

        except ImportError as e:
            return LoaderResult(
                success=False,
                error="Pillow is required for image processing. Install with: pip install Pillow",
                file_type="image"
            )
        except Exception as e:
            return LoaderResult(
                success=False,
                error=f"Failed to load image: {str(e)}",
                file_type="image"
            )

    def load_from_bytes(self, content: bytes, filename: str) -> LoaderResult:
        """Load image from bytes."""
        try:
            with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            result = self.load(tmp_path)

            # Update metadata with original filename
            for doc in result.documents:
                doc.metadata["source"] = filename
                doc.metadata["filename"] = filename

            Path(tmp_path).unlink(missing_ok=True)
            return result

        except Exception as e:
            return LoaderResult(
                success=False,
                error=f"Failed to load image from bytes: {str(e)}",
                file_type="image"
            )

    def _extract_ocr(self, img) -> Optional[str]:
        """Extract text from image using OCR."""
        try:
            import pytesseract

            # Convert to RGB if necessary
            if img.mode != "RGB":
                img = img.convert("RGB")

            text = pytesseract.image_to_string(img)
            return text.strip() if text.strip() else None

        except ImportError:
            # pytesseract not installed
            return None
        except Exception:
            return None

    def _get_vision_description(self, image_bytes: bytes) -> Optional[str]:
        """Get image description using vision model."""
        try:
            import ollama

            # Encode image to base64
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")

            response = ollama.chat(
                model=self.vision_model,
                messages=[{
                    "role": "user",
                    "content": "Describe this image in detail. Include any text visible in the image, objects, people, colors, and overall context.",
                    "images": [image_b64]
                }]
            )

            return response["message"]["content"]

        except Exception:
            # Vision model not available or error
            return None


class AudioLoader(BaseLoader):
    """
    Loader for audio files.
    Uses Whisper for transcription.
    """

    SUPPORTED_EXTENSIONS = [".mp3", ".wav", ".m4a", ".flac", ".ogg", ".wma", ".aac"]

    def __init__(self, model_size: str = "base"):
        """
        Initialize audio loader.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
        """
        self.model_size = model_size
        self._model = None

    def _get_model(self):
        """Lazy load Whisper model."""
        if self._model is None:
            import whisper
            self._model = whisper.load_model(self.model_size)
        return self._model

    def load(self, file_path: str) -> LoaderResult:
        """Load and transcribe an audio file."""
        try:
            import whisper

            path = Path(file_path)

            # Get audio duration
            duration = self._get_duration(file_path)

            # Transcribe
            model = self._get_model()

            with open(file_path, "rb") as f:
                result = model.transcribe(file_path)

            transcription = result["text"].strip()

            if not transcription:
                return LoaderResult(
                    success=False,
                    error="No speech detected in audio file",
                    file_type="audio"
                )

            # Create document with segments if available
            documents = []

            # Main transcription document
            metadata = {
                **self._create_metadata(file_path),
                "duration_seconds": duration,
                "language": result.get("language", "unknown"),
            }

            documents.append(Document(
                page_content=f"Audio Transcription: {path.name}\nDuration: {self._format_duration(duration)}\n\n{transcription}",
                metadata=metadata
            ))

            # If long audio, also create segment documents
            segments = result.get("segments", [])
            if len(segments) > 5:
                SEGMENTS_PER_DOC = 10
                for i in range(0, len(segments), SEGMENTS_PER_DOC):
                    chunk_segments = segments[i:i + SEGMENTS_PER_DOC]
                    chunk_text = " ".join(seg["text"].strip() for seg in chunk_segments)

                    start_time = chunk_segments[0]["start"]
                    end_time = chunk_segments[-1]["end"]

                    documents.append(Document(
                        page_content=chunk_text,
                        metadata={
                            **metadata,
                            "segment_start": start_time,
                            "segment_end": end_time,
                            "timestamp": f"{self._format_duration(start_time)} - {self._format_duration(end_time)}",
                        }
                    ))

            return LoaderResult(
                success=True,
                documents=documents,
                file_type="audio",
                metadata={
                    "duration_seconds": duration,
                    "language": result.get("language"),
                    "segment_count": len(segments),
                }
            )

        except ImportError:
            return LoaderResult(
                success=False,
                error="openai-whisper is required for audio transcription. Install with: pip install openai-whisper",
                file_type="audio"
            )
        except Exception as e:
            return LoaderResult(
                success=False,
                error=f"Failed to transcribe audio: {str(e)}",
                file_type="audio"
            )

    def load_from_bytes(self, content: bytes, filename: str) -> LoaderResult:
        """Load audio from bytes."""
        try:
            suffix = Path(filename).suffix or ".mp3"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            result = self.load(tmp_path)

            for doc in result.documents:
                doc.metadata["source"] = filename
                doc.metadata["filename"] = filename

            Path(tmp_path).unlink(missing_ok=True)
            return result

        except Exception as e:
            return LoaderResult(
                success=False,
                error=f"Failed to load audio from bytes: {str(e)}",
                file_type="audio"
            )

    def _get_duration(self, file_path: str) -> float:
        """Get audio duration in seconds."""
        try:
            import subprocess
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", file_path],
                capture_output=True, text=True
            )
            return float(result.stdout.strip())
        except Exception:
            return 0.0

    def _format_duration(self, seconds: float) -> str:
        """Format duration as HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"


class VideoLoader(BaseLoader):
    """
    Loader for video files.
    Extracts frames and transcribes audio.
    """

    SUPPORTED_EXTENSIONS = [".mp4", ".mov", ".avi", ".mkv", ".webm", ".wmv", ".flv"]

    def __init__(
        self,
        extract_frames: bool = True,
        transcribe_audio: bool = True,
        frame_interval: int = 30,
        max_frames: int = 10,
        vision_model: str = "llava:7b",
        whisper_model: str = "base"
    ):
        """
        Initialize video loader.

        Args:
            extract_frames: Whether to extract and analyze frames
            transcribe_audio: Whether to transcribe audio track
            frame_interval: Extract frame every N seconds
            max_frames: Maximum number of frames to extract
            vision_model: Ollama vision model for frame analysis
            whisper_model: Whisper model size for transcription
        """
        self.extract_frames = extract_frames
        self.transcribe_audio = transcribe_audio
        self.frame_interval = frame_interval
        self.max_frames = max_frames
        self.vision_model = vision_model
        self.whisper_model = whisper_model

    def load(self, file_path: str) -> LoaderResult:
        """Load and process a video file."""
        try:
            import cv2

            path = Path(file_path)

            # Open video
            cap = cv2.VideoCapture(file_path)

            if not cap.isOpened():
                return LoaderResult(
                    success=False,
                    error="Could not open video file",
                    file_type="video"
                )

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            metadata = {
                **self._create_metadata(file_path),
                "duration_seconds": duration,
                "fps": fps,
                "frame_count": frame_count,
                "width": width,
                "height": height,
            }

            documents = []
            content_parts = [
                f"Video: {path.name}",
                f"Duration: {self._format_duration(duration)}",
                f"Resolution: {width}x{height}",
                f"FPS: {fps:.2f}",
                ""
            ]

            # Extract and analyze frames
            if self.extract_frames:
                frame_descriptions = self._extract_and_analyze_frames(cap, duration)
                if frame_descriptions:
                    content_parts.append("[Frame Analysis]")
                    for timestamp, description in frame_descriptions:
                        content_parts.append(f"At {timestamp}: {description}")
                    content_parts.append("")

            cap.release()

            # Transcribe audio
            if self.transcribe_audio:
                transcription = self._transcribe_audio(file_path)
                if transcription:
                    content_parts.append("[Audio Transcription]")
                    content_parts.append(transcription)

            # Create main document
            documents.append(Document(
                page_content="\n".join(content_parts),
                metadata=metadata
            ))

            return LoaderResult(
                success=True,
                documents=documents,
                file_type="video",
                metadata=metadata
            )

        except ImportError:
            return LoaderResult(
                success=False,
                error="opencv-python is required for video processing. Install with: pip install opencv-python",
                file_type="video"
            )
        except Exception as e:
            return LoaderResult(
                success=False,
                error=f"Failed to process video: {str(e)}",
                file_type="video"
            )

    def load_from_bytes(self, content: bytes, filename: str) -> LoaderResult:
        """Load video from bytes."""
        try:
            suffix = Path(filename).suffix or ".mp4"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            result = self.load(tmp_path)

            for doc in result.documents:
                doc.metadata["source"] = filename
                doc.metadata["filename"] = filename

            Path(tmp_path).unlink(missing_ok=True)
            return result

        except Exception as e:
            return LoaderResult(
                success=False,
                error=f"Failed to load video from bytes: {str(e)}",
                file_type="video"
            )

    def _extract_and_analyze_frames(self, cap, duration: float) -> List[tuple]:
        """Extract frames and get descriptions."""
        try:
            import cv2
            import ollama

            descriptions = []
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Calculate frame positions
            interval_frames = int(self.frame_interval * fps)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            frame_positions = []
            pos = 0
            while pos < total_frames and len(frame_positions) < self.max_frames:
                frame_positions.append(pos)
                pos += interval_frames

            for frame_pos in frame_positions:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                ret, frame = cap.read()

                if not ret:
                    continue

                # Convert frame to bytes
                _, buffer = cv2.imencode(".jpg", frame)
                image_b64 = base64.b64encode(buffer).decode("utf-8")

                # Get description from vision model
                try:
                    response = ollama.chat(
                        model=self.vision_model,
                        messages=[{
                            "role": "user",
                            "content": "Briefly describe what's happening in this video frame in 1-2 sentences.",
                            "images": [image_b64]
                        }]
                    )

                    timestamp = self._format_duration(frame_pos / fps)
                    description = response["message"]["content"].strip()
                    descriptions.append((timestamp, description))

                except Exception:
                    continue

            return descriptions

        except Exception:
            return []

    def _transcribe_audio(self, video_path: str) -> Optional[str]:
        """Extract and transcribe audio from video."""
        try:
            import subprocess
            import whisper

            # Extract audio to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                audio_path = tmp.name

            # Use ffmpeg to extract audio
            subprocess.run([
                "ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1", "-y", audio_path
            ], capture_output=True, check=True)

            # Transcribe
            model = whisper.load_model(self.whisper_model)
            result = model.transcribe(audio_path)

            # Clean up
            Path(audio_path).unlink(missing_ok=True)

            return result["text"].strip()

        except Exception:
            return None

    def _format_duration(self, seconds: float) -> str:
        """Format duration as HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"


# Convenience function to get all media loaders
def get_media_loaders() -> Dict[str, BaseLoader]:
    """Get dictionary of all media loaders."""
    return {
        "image": ImageLoader(),
        "audio": AudioLoader(),
        "video": VideoLoader(),
    }
