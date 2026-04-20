"""
Ollama Client
Wrapper for Ollama API with configuration and status management.
"""

import ollama

from dataclasses import dataclass, field
from typing import Optional, Generator, List, Dict, Any, Callable, Union
from enum import Enum

class ConnectionStatus(Enum):
    """Ollama connection status."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"


@dataclass
class OllamaConfig:
    """Configuration for Ollama LLM."""
    model: str = "deepseek-r1:latest"
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 40
    num_ctx: int = 4096  # Context window size
    num_predict: int = -1  # Max tokens (-1 = unlimited)
    repeat_penalty: float = 1.1
    stop: List[str] = field(default_factory=list)

    # Ollama server settings
    host: str = "http://localhost:11434"
    timeout: float = 120.0

    def to_options(self) -> Dict[str, Any]:
        """Convert to Ollama options dict."""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
            "repeat_penalty": self.repeat_penalty,
        }


@dataclass
class ModelStatus:
    """Status information for a model."""
    name: str
    is_loaded: bool = False
    is_available: bool = False
    size: str = ""
    family: str = ""
    parameter_size: str = ""
    quantization: str = ""
    modified_at: str = ""
    digest: str = ""


@dataclass
class GenerationStats:
    """Statistics from generation."""
    total_duration: float = 0.0
    load_duration: float = 0.0
    prompt_eval_count: int = 0
    prompt_eval_duration: float = 0.0
    eval_count: int = 0
    eval_duration: float = 0.0

    @property
    def tokens_per_second(self) -> float:
        """Calculate tokens per second."""
        if self.eval_duration > 0:
            return self.eval_count / (self.eval_duration / 1e9)
        return 0.0


class OllamaClient:
    """
    Client for interacting with Ollama API.
    Provides configuration, streaming, and status management.
    """

    def __init__(self, config: Optional[OllamaConfig] = None):
        """
        Initialize Ollama client.
        Args: (config: Ollama configuration)
        """
        self.config = config or OllamaConfig()
        self._client = ollama
        self._last_stats: Optional[GenerationStats] = None

    def check_connection(self) -> ConnectionStatus:
        """
        Check connection to Ollama server.
        Returns: Connection status
        """
        try:
            self._client.list()
            return ConnectionStatus.CONNECTED
        except Exception:
            return ConnectionStatus.DISCONNECTED

    def is_connected(self) -> bool:
        """Check if connected to Ollama."""
        return self.check_connection() == ConnectionStatus.CONNECTED

    def list_models(self) -> List[ModelStatus]:
        """
        List all available models.
        Returns: List of model status objects
        """
        try:
            response = self._client.list()
            models = []

            for model in response.get("models", []):
                name = model.get("model", "")
                details = model.get("details", {})

                models.append(ModelStatus(
                    name=name,
                    is_available=True,
                    is_loaded=False,  # Will be updated by get_running_models
                    size=self._format_size(model.get("size", 0)),
                    family=details.get("family", ""),
                    parameter_size=details.get("parameter_size", ""),
                    quantization=details.get("quantization_level", ""),
                    modified_at=model.get("modified_at", ""),
                    digest=model.get("digest", "")[:12] if model.get("digest") else ""
                ))

            return models

        except Exception:
            return []

    def get_running_models(self) -> List[str]:
        """
        Get list of currently loaded/running models.
        Returns:List of model names
        """
        try:
            response = self._client.ps()
            return [m.get("model", "") for m in response.get("models", [])]
        except Exception:
            return []

    def get_model_info(self, model_name: str) -> Optional[ModelStatus]:
        """
        Get detailed information about a specific model.
        Args: (model_name: Name of the model)
        Returns:Model status or None if not found
        """
        try:
            response = self._client.show(model_name)

            details = response.get("details", {})
            modelinfo = response.get("modelinfo", {})

            # Get size from modelinfo
            size = 0
            for key, value in modelinfo.items():
                if "size" in key.lower() and isinstance(value, (int, float)):
                    size = value
                    break

            running = self.get_running_models()

            return ModelStatus(
                name=model_name,
                is_available=True,
                is_loaded=model_name in running,
                size=self._format_size(size),
                family=details.get("family", ""),
                parameter_size=details.get("parameter_size", ""),
                quantization=details.get("quantization_level", ""),
            )

        except Exception:
            return None

    def model_exists(self, model_name: str) -> bool:
        """Check if a model exists locally."""
        models = self.list_models()
        return any(m.name == model_name or m.name.startswith(model_name + ":") for m in models)

    def pull_model(
        self,
        model_name: str,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> bool:
        """
        Pull a model from Ollama registry.
        Args:
            model_name: Name of the model to pull
            progress_callback: Optional callback for progress updates (status, percent)
        Returns:True if successful
        """
        try:
            stream = self._client.pull(model_name, stream=True)

            for chunk in stream:
                status = chunk.get("status", "")
                total = chunk.get("total", 0)
                completed = chunk.get("completed", 0)

                if progress_callback and total > 0:
                    percent = (completed / total) * 100
                    progress_callback(status, percent)
                elif progress_callback:
                    progress_callback(status, 0)

            return True

        except Exception as e:
            if progress_callback:
                progress_callback(f"Error: {e}", 0)
            return False

    def delete_model(self, model_name: str) -> bool:
        """
        Delete a model.
        Args: (model_name: Name of the model to delete)
        Returns: True if successful
        """
        try:
            self._client.delete(model_name)
            return True
        except Exception:
            return False

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate a response (non-streaming).

        Args:
            prompt: User prompt
            model: Model to use (defaults to config)
            system: System prompt
            **kwargs: Override config options

        Returns: Generated text
        """
        model_name = model or self.config.model
        options = self.config.to_options()
        options.update(kwargs)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.chat(
                model=model_name,
                messages=messages,
                options=options
            )

            # Store stats
            self._last_stats = self._parse_stats(response)

            return response.get("message", {}).get("content", "")

        except Exception as e:
            return f"Error: {e}"

    def generate_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        Generate a streaming response.

        Args:
            prompt: User prompt
            model: Model to use (defaults to config)
            system: System prompt
            **kwargs: Override config options

        Yields: Response chunks
        """
        model_name = model or self.config.model
        options = self.config.to_options()
        options.update(kwargs)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            stream = self._client.chat(
                model=model_name,
                messages=messages,
                options=options,
                stream=True
            )

            for chunk in stream:
                if "message" in chunk and "content" in chunk["message"]:
                    yield chunk["message"]["content"]

                # Store final stats
                if chunk.get("done", False):
                    self._last_stats = self._parse_stats(chunk)

        except Exception as e:
            yield f"Error: Could not connect to Ollama. Make sure it's running. ({e})"

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        stream: bool = True,
        **kwargs
    ) -> Union[Generator[str, None, None], str]:
        """
        Chat with message history.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use
            stream: Whether to stream response
            **kwargs: Override config options

        Returns: Generated response (generator if streaming, string if not)
        """
        model_name = model or self.config.model
        options = self.config.to_options()
        options.update(kwargs)

        try:
            if stream:
                response_stream = self._client.chat(
                    model=model_name,
                    messages=messages,
                    options=options,
                    stream=True
                )

                def stream_generator():
                    for chunk in response_stream:
                        if "message" in chunk and "content" in chunk["message"]:
                            yield chunk["message"]["content"]
                        if chunk.get("done", False):
                            self._last_stats = self._parse_stats(chunk)

                return stream_generator()
            else:
                response = self._client.chat(
                    model=model_name,
                    messages=messages,
                    options=options
                )
                self._last_stats = self._parse_stats(response)
                return response.get("message", {}).get("content", "")

        except Exception as e:
            if stream:
                def error_generator():
                    yield f"Error: {e}"
                return error_generator()
            return f"Error: {e}"

    def embed(
        self,
        text: str,
        model: str = "nomic-embed-text"
    ) -> List[float]:
        """
        Generate embeddings for text.

        Args:
            text: Text to embed
            model: Embedding model

        Returns: Embedding vector
        """
        try:
            response = self._client.embeddings(model=model, prompt=text)
            return response.get("embedding", [])
        except Exception:
            return []

    def get_last_stats(self) -> Optional[GenerationStats]:
        """Get statistics from last generation."""
        return self._last_stats

    def update_config(self, **kwargs) -> None:
        """
        Update configuration.

        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

    def set_model(self, model_name: str) -> None:
        """Set the current model."""
        self.config.model = model_name

    def _parse_stats(self, response: Dict[str, Any]) -> GenerationStats:
        """Parse generation statistics from response."""
        return GenerationStats(
            total_duration=response.get("total_duration", 0) / 1e9,
            load_duration=response.get("load_duration", 0) / 1e9,
            prompt_eval_count=response.get("prompt_eval_count", 0),
            prompt_eval_duration=response.get("prompt_eval_duration", 0),
            eval_count=response.get("eval_count", 0),
            eval_duration=response.get("eval_duration", 0),
        )

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format size in human-readable format."""
        if size_bytes == 0:
            return "Unknown"

        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024

        return f"{size_bytes:.1f} PB"
