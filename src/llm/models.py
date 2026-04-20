"""
Model Registry
Definitions and capabilities for supported models.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from enum import Enum, auto


class ModelCapability(Enum):
    """Capabilities that models can have."""
    CHAT = auto()           # Basic chat/conversation
    THINKING = auto()       # Chain-of-thought reasoning with <think> tags
    VISION = auto()         # Image understanding
    CODE = auto()           # Code generation/understanding
    EMBEDDING = auto()      # Text embeddings
    FUNCTION_CALLING = auto()  # Function/tool calling
    LONG_CONTEXT = auto()   # Extended context window (32k+)
    FAST = auto()           # Optimized for speed
    MULTILINGUAL = auto()   # Multiple language support


class ModelCategory(Enum):
    """Categories of models."""
    THINKING = "thinking"
    CHAT = "chat"
    VISION = "vision"
    CODE = "code"
    EMBEDDING = "embedding"


@dataclass
class ModelInfo:
    """Information about a model."""
    name: str
    display_name: str
    category: ModelCategory
    capabilities: Set[ModelCapability]
    parameter_size: str
    context_length: int
    description: str
    recommended_for: List[str] = field(default_factory=list)
    variants: List[str] = field(default_factory=list)

    @property
    def has_thinking(self) -> bool:
        """Check if model supports thinking/reasoning."""
        return ModelCapability.THINKING in self.capabilities

    @property
    def has_vision(self) -> bool:
        """Check if model supports vision."""
        return ModelCapability.VISION in self.capabilities

    @property
    def is_embedding_model(self) -> bool:
        """Check if model is for embeddings."""
        return ModelCapability.EMBEDDING in self.capabilities


# Supported models registry
SUPPORTED_MODELS: Dict[str, ModelInfo] = {
    # Thinking/Reasoning Models
    "deepseek-r1:latest": ModelInfo(
        name="deepseek-r1:latest",
        display_name="Deepseek R1 8B",
        category=ModelCategory.THINKING,
        capabilities={ModelCapability.CHAT, ModelCapability.THINKING, ModelCapability.CODE},
        parameter_size="8.2B",
        context_length=131072,
        description="Premium reasoning model for most demanding tasks",
        recommended_for=["Research", "Complex analysis", "Expert-level tasks"],
        variants=["deepseek-r1:8b", "deepseek-r1:8b-qwen3"]
    ),
    "deepseek-r1:1.5b": ModelInfo(
        name="deepseek-r1:1.5b",
        display_name="Deepseek R1 1.5B",
        category=ModelCategory.THINKING,
        capabilities={ModelCapability.CHAT, ModelCapability.THINKING, ModelCapability.CODE},
        parameter_size="1.5B",
        context_length=4096,
        description="Compact reasoning model with chain-of-thought",
        recommended_for=["Quick reasoning", "Code analysis", "Low resource usage"],
        variants=["deepseek-r1:1.5b", "deepseek-r1:1.5b-qwen-distill"]
    ),
    "deepseek-r1:7b": ModelInfo(
        name="deepseek-r1:7b",
        display_name="Deepseek R1 7B",
        category=ModelCategory.THINKING,
        capabilities={ModelCapability.CHAT, ModelCapability.THINKING, ModelCapability.CODE},
        parameter_size="7B",
        context_length=4096,
        description="Balanced reasoning model with improved capabilities",
        recommended_for=["Complex reasoning", "Detailed analysis", "Code generation"],
        variants=["deepseek-r1:7b", "deepseek-r1:7b-qwen-distill"]
    ),
    "deepseek-r1:14b": ModelInfo(
        name="deepseek-r1:14b",
        display_name="Deepseek R1 14B",
        category=ModelCategory.THINKING,
        capabilities={ModelCapability.CHAT, ModelCapability.THINKING, ModelCapability.CODE, ModelCapability.LONG_CONTEXT},
        parameter_size="14B",
        context_length=8192,
        description="Large reasoning model for complex tasks",
        recommended_for=["Advanced reasoning", "Long documents", "Complex code"],
        variants=["deepseek-r1:14b", "deepseek-r1:14b-qwen-distill"]
    ),
    "deepseek-r1:32b": ModelInfo(
        name="deepseek-r1:32b",
        display_name="Deepseek R1 32B",
        category=ModelCategory.THINKING,
        capabilities={ModelCapability.CHAT, ModelCapability.THINKING, ModelCapability.CODE, ModelCapability.LONG_CONTEXT},
        parameter_size="32B",
        context_length=16384,
        description="Premium reasoning model for most demanding tasks",
        recommended_for=["Research", "Complex analysis", "Expert-level tasks"],
        variants=["deepseek-r1:32b"]
    ),

    # Chat Models
    "llama3.2:1b": ModelInfo(
        name="llama3.2:1b",
        display_name="Llama 3.2 1B",
        category=ModelCategory.CHAT,
        capabilities={ModelCapability.CHAT, ModelCapability.FAST},
        parameter_size="1B",
        context_length=4096,
        description="Ultra-fast, lightweight chat model",
        recommended_for=["Quick responses", "Simple tasks", "Low memory"],
        variants=["llama3.2:1b"]
    ),
    "llama3.2:3b": ModelInfo(
        name="llama3.2:3b",
        display_name="Llama 3.2 3B",
        category=ModelCategory.CHAT,
        capabilities={ModelCapability.CHAT, ModelCapability.FAST, ModelCapability.MULTILINGUAL},
        parameter_size="3B",
        context_length=4096,
        description="Fast and capable chat model",
        recommended_for=["General chat", "Quick tasks", "Multilingual"],
        variants=["llama3.2:3b"]
    ),
    "llama3.1:8b": ModelInfo(
        name="llama3.1:8b",
        display_name="Llama 3.1 8B",
        category=ModelCategory.CHAT,
        capabilities={ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.FUNCTION_CALLING, ModelCapability.LONG_CONTEXT},
        parameter_size="8B",
        context_length=128000,
        description="Powerful chat model with extended context",
        recommended_for=["Long conversations", "Document analysis", "Code"],
        variants=["llama3.1:8b", "llama3.1:8b-instruct"]
    ),
    "mistral:7b": ModelInfo(
        name="mistral:7b",
        display_name="Mistral 7B",
        category=ModelCategory.CHAT,
        capabilities={ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.FAST},
        parameter_size="7B",
        context_length=8192,
        description="Efficient and capable general-purpose model",
        recommended_for=["General tasks", "Coding", "Fast inference"],
        variants=["mistral:7b", "mistral:7b-instruct"]
    ),
    "qwen2.5:7b": ModelInfo(
        name="qwen2.5:7b",
        display_name="Qwen 2.5 7B",
        category=ModelCategory.CHAT,
        capabilities={ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.MULTILINGUAL, ModelCapability.LONG_CONTEXT},
        parameter_size="7B",
        context_length=32768,
        description="Multilingual model with long context",
        recommended_for=["Multilingual", "Long documents", "Coding"],
        variants=["qwen2.5:7b", "qwen2.5:7b-instruct"]
    ),
    "qwen2.5:14b": ModelInfo(
        name="qwen2.5:14b",
        display_name="Qwen 2.5 14B",
        category=ModelCategory.CHAT,
        capabilities={ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.MULTILINGUAL, ModelCapability.LONG_CONTEXT},
        parameter_size="14B",
        context_length=32768,
        description="Large multilingual model",
        recommended_for=["Complex multilingual", "Advanced coding", "Analysis"],
        variants=["qwen2.5:14b", "qwen2.5:14b-instruct"]
    ),
    "gemma2:9b": ModelInfo(
        name="gemma2:9b",
        display_name="Gemma 2 9B",
        category=ModelCategory.CHAT,
        capabilities={ModelCapability.CHAT, ModelCapability.CODE},
        parameter_size="9B",
        context_length=8192,
        description="Google's efficient chat model",
        recommended_for=["General chat", "Creative writing", "Code"],
        variants=["gemma2:9b", "gemma2:2b"]
    ),
    "phi3:medium": ModelInfo(
        name="phi3:medium",
        display_name="Phi-3 Medium",
        category=ModelCategory.CHAT,
        capabilities={ModelCapability.CHAT, ModelCapability.CODE, ModelCapability.FAST},
        parameter_size="14B",
        context_length=4096,
        description="Microsoft's efficient model",
        recommended_for=["Coding", "Reasoning", "Efficient inference"],
        variants=["phi3:mini", "phi3:medium"]
    ),

    # Vision Models
    "llava:7b": ModelInfo(
        name="llava:7b",
        display_name="LLaVA 7B",
        category=ModelCategory.VISION,
        capabilities={ModelCapability.CHAT, ModelCapability.VISION},
        parameter_size="7B",
        context_length=4096,
        description="Vision-language model for image understanding",
        recommended_for=["Image analysis", "Visual Q&A", "OCR assistance"],
        variants=["llava:7b", "llava:13b", "llava:34b"]
    ),
    "llava:13b": ModelInfo(
        name="llava:13b",
        display_name="LLaVA 13B",
        category=ModelCategory.VISION,
        capabilities={ModelCapability.CHAT, ModelCapability.VISION},
        parameter_size="13B",
        context_length=4096,
        description="Larger vision model for detailed analysis",
        recommended_for=["Detailed image analysis", "Complex visuals"],
        variants=["llava:13b"]
    ),
    "llava-llama3:8b": ModelInfo(
        name="llava-llama3:8b",
        display_name="LLaVA Llama3 8B",
        category=ModelCategory.VISION,
        capabilities={ModelCapability.CHAT, ModelCapability.VISION},
        parameter_size="8B",
        context_length=4096,
        description="LLaVA with Llama 3 backbone",
        recommended_for=["Image understanding", "Visual chat"],
        variants=["llava-llama3:8b"]
    ),
    "moondream:1.8b": ModelInfo(
        name="moondream:1.8b",
        display_name="Moondream 1.8B",
        category=ModelCategory.VISION,
        capabilities={ModelCapability.VISION, ModelCapability.FAST},
        parameter_size="1.8B",
        context_length=2048,
        description="Lightweight vision model",
        recommended_for=["Quick image analysis", "Low resources"],
        variants=["moondream:1.8b"]
    ),

    # Code Models
    "codellama:7b": ModelInfo(
        name="codellama:7b",
        display_name="Code Llama 7B",
        category=ModelCategory.CODE,
        capabilities={ModelCapability.CODE, ModelCapability.CHAT},
        parameter_size="7B",
        context_length=16384,
        description="Specialized code generation model",
        recommended_for=["Code generation", "Code completion", "Debugging"],
        variants=["codellama:7b", "codellama:13b", "codellama:34b"]
    ),
    "codellama:13b": ModelInfo(
        name="codellama:13b",
        display_name="Code Llama 13B",
        category=ModelCategory.CODE,
        capabilities={ModelCapability.CODE, ModelCapability.CHAT, ModelCapability.LONG_CONTEXT},
        parameter_size="13B",
        context_length=16384,
        description="Advanced code model",
        recommended_for=["Complex code", "Large codebases", "Refactoring"],
        variants=["codellama:13b"]
    ),
    "deepseek-coder:6.7b": ModelInfo(
        name="deepseek-coder:6.7b",
        display_name="Deepseek Coder 6.7B",
        category=ModelCategory.CODE,
        capabilities={ModelCapability.CODE, ModelCapability.CHAT},
        parameter_size="6.7B",
        context_length=16384,
        description="Specialized coding model",
        recommended_for=["Code generation", "Bug fixing", "Code review"],
        variants=["deepseek-coder:6.7b", "deepseek-coder:33b"]
    ),
    "starcoder2:7b": ModelInfo(
        name="starcoder2:7b",
        display_name="StarCoder2 7B",
        category=ModelCategory.CODE,
        capabilities={ModelCapability.CODE},
        parameter_size="7B",
        context_length=16384,
        description="Open-source code model",
        recommended_for=["Code completion", "Multiple languages"],
        variants=["starcoder2:3b", "starcoder2:7b", "starcoder2:15b"]
    ),

    # Embedding Models
    "nomic-embed-text": ModelInfo(
        name="nomic-embed-text",
        display_name="Nomic Embed Text",
        category=ModelCategory.EMBEDDING,
        capabilities={ModelCapability.EMBEDDING, ModelCapability.LONG_CONTEXT},
        parameter_size="137M",
        context_length=8192,
        description="High-quality text embeddings",
        recommended_for=["Semantic search", "RAG", "Clustering"],
        variants=["nomic-embed-text"]
    ),
    "mxbai-embed-large": ModelInfo(
        name="mxbai-embed-large",
        display_name="MixedBread Embed Large",
        category=ModelCategory.EMBEDDING,
        capabilities={ModelCapability.EMBEDDING},
        parameter_size="335M",
        context_length=512,
        description="Large embedding model for high accuracy",
        recommended_for=["High-precision search", "Similarity"],
        variants=["mxbai-embed-large"]
    ),
    "all-minilm": ModelInfo(
        name="all-minilm",
        display_name="All-MiniLM",
        category=ModelCategory.EMBEDDING,
        capabilities={ModelCapability.EMBEDDING, ModelCapability.FAST},
        parameter_size="33M",
        context_length=256,
        description="Fast, lightweight embeddings",
        recommended_for=["Quick embeddings", "Low resources"],
        variants=["all-minilm"]
    ),
    "snowflake-arctic-embed": ModelInfo(
        name="snowflake-arctic-embed",
        display_name="Snowflake Arctic Embed",
        category=ModelCategory.EMBEDDING,
        capabilities={ModelCapability.EMBEDDING, ModelCapability.LONG_CONTEXT},
        parameter_size="335M",
        context_length=8192,
        description="Enterprise-grade embeddings",
        recommended_for=["Enterprise search", "Long documents"],
        variants=["snowflake-arctic-embed:xs", "snowflake-arctic-embed:s", "snowflake-arctic-embed:m"]
    ),
}


class ModelRegistry:
    """
    Registry for managing model information.
    """

    def __init__(self):
        """Initialize model registry."""
        self._models = SUPPORTED_MODELS.copy()

    def get_model(self, name: str) -> Optional[ModelInfo]:
        """Get model info by name."""
        # Exact match
        if name in self._models:
            return self._models[name]

        # Try base name (without tag)
        base_name = name.split(":")[0]
        for model_name, info in self._models.items():
            if model_name.startswith(base_name + ":"):
                return info

        return None

    def list_models(self, category: Optional[ModelCategory] = None) -> List[ModelInfo]:
        """List models, optionally filtered by category."""
        models = list(self._models.values())

        if category:
            models = [m for m in models if m.category == category]

        return models

    def list_by_capability(self, capability: ModelCapability) -> List[ModelInfo]:
        """List models with a specific capability."""
        return [m for m in self._models.values() if capability in m.capabilities]

    def get_thinking_models(self) -> List[ModelInfo]:
        """Get models with thinking/reasoning capability."""
        return self.list_by_capability(ModelCapability.THINKING)

    def get_vision_models(self) -> List[ModelInfo]:
        """Get models with vision capability."""
        return self.list_by_capability(ModelCapability.VISION)

    def get_embedding_models(self) -> List[ModelInfo]:
        """Get embedding models."""
        return self.list_models(category=ModelCategory.EMBEDDING)

    def get_chat_models(self) -> List[ModelInfo]:
        """Get general chat models."""
        return self.list_models(category=ModelCategory.CHAT)

    def get_code_models(self) -> List[ModelInfo]:
        """Get code-specialized models."""
        return self.list_models(category=ModelCategory.CODE)

    def add_custom_model(self, model: ModelInfo) -> None:
        """Add a custom model to the registry."""
        self._models[model.name] = model

    def get_recommended_models(self) -> Dict[str, ModelInfo]:
        """Get recommended models for each category."""
        return {
            "thinking": self._models.get("deepseek-r1:1.5b"),
            "chat": self._models.get("llama3.2:3b"),
            "vision": self._models.get("llava:7b"),
            "code": self._models.get("codellama:7b"),
            "embedding": self._models.get("nomic-embed-text"),
        }


def get_model_info(name: str) -> Optional[ModelInfo]:
    """
    Get model information by name.

    Args:
        name: Model name

    Returns:
        ModelInfo or None
    """
    return ModelRegistry().get_model(name)
