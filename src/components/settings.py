"""
Settings Component
Model configuration and parameter tuning UI.
"""

import streamlit as st
from typing import Optional, Dict, Any, List, Callable

from ..llm.ollama import OllamaClient, OllamaConfig, ConnectionStatus
from ..llm.models import (
    ModelRegistry, ModelInfo, ModelCategory, ModelCapability,
    SUPPORTED_MODELS
)


def render_connection_status(client: OllamaClient) -> bool:
    """
    Render Ollama connection status.

    Returns:
        True if connected
    """
    status = client.check_connection()

    if status == ConnectionStatus.CONNECTED:
        st.success("🟢 Ollama Connected")
        return True
    else:
        st.error("🔴 Ollama Disconnected")
        st.caption("Make sure Ollama is running: `ollama serve`")
        return False


def render_model_selector(
    client: OllamaClient,
    registry: ModelRegistry,
    current_model: str,
    on_change: Optional[Callable[[str], None]] = None,
    category_filter: Optional[ModelCategory] = None,
    key_prefix: str = "model"
) -> str:
    """
    Render model selection dropdown with info.

    Args:
        client: OllamaClient instance
        registry: ModelRegistry instance
        current_model: Currently selected model
        on_change: Callback when model changes
        category_filter: Filter to specific category
        key_prefix: Unique key prefix for widgets

    Returns:
        Selected model name
    """
    # Get available models from Ollama
    available_models = client.list_models()
    available_names = [m.name for m in available_models]

    # Get models from registry
    if category_filter:
        registry_models = registry.list_models(category=category_filter)
    else:
        registry_models = list(SUPPORTED_MODELS.values())

    # Build options list - prioritize available models
    options = []
    model_info_map: Dict[str, Optional[ModelInfo]] = {}

    # Add available models first
    for model in available_models:
        options.append(model.name)
        model_info_map[model.name] = registry.get_model(model.name)

    # Add registry models that aren't available (for pulling)
    for info in registry_models:
        if info.name not in options:
            options.append(info.name)
            model_info_map[info.name] = info

    # Ensure current model is in list
    if current_model and current_model not in options:
        options.insert(0, current_model)
        model_info_map[current_model] = registry.get_model(current_model)

    # Default selection index
    try:
        default_index = options.index(current_model) if current_model in options else 0
    except ValueError:
        default_index = 0

    # Model selector
    selected = st.selectbox(
        "Model",
        options=options,
        index=default_index,
        format_func=lambda x: _format_model_option(x, model_info_map.get(x), x in available_names),
        key=f"{key_prefix}_selector"
    )

    # Show model info
    info = model_info_map.get(selected)
    if info:
        _render_model_info(info, selected in available_names)
    elif selected in available_names:
        st.caption(f"Custom model: {selected}")

    # Handle model not available
    if selected not in available_names:
        st.warning(f"Model not installed locally")
        if st.button(f"Pull {selected}", key=f"{key_prefix}_pull"):
            _pull_model_with_progress(client, selected)
            st.rerun()

    # Trigger callback if changed
    if on_change and selected != current_model:
        on_change(selected)

    return selected


def _format_model_option(name: str, info: Optional[ModelInfo], available: bool) -> str:
    """Format model name for dropdown."""
    status = "✓" if available else "↓"

    if info:
        return f"{status} {info.display_name} ({info.parameter_size})"
    else:
        return f"{status} {name}"


def _render_model_info(info: ModelInfo, available: bool):
    """Render model information card."""
    # Capabilities badges
    caps = []
    if ModelCapability.THINKING in info.capabilities:
        caps.append("🧠 Thinking")
    if ModelCapability.VISION in info.capabilities:
        caps.append("👁️ Vision")
    if ModelCapability.CODE in info.capabilities:
        caps.append("💻 Code")
    if ModelCapability.LONG_CONTEXT in info.capabilities:
        caps.append("📚 Long Context")
    if ModelCapability.FAST in info.capabilities:
        caps.append("⚡ Fast")

    if caps:
        st.caption(" • ".join(caps))

    st.caption(info.description)


def _pull_model_with_progress(client: OllamaClient, model_name: str):
    """Pull model with progress bar."""
    progress_bar = st.progress(0, text=f"Pulling {model_name}...")

    def update_progress(status: str, percent: float):
        progress_bar.progress(
            min(int(percent), 100),
            text=f"{status}: {percent:.1f}%"
        )

    success = client.pull_model(model_name, progress_callback=update_progress)

    if success:
        progress_bar.progress(100, text="Complete!")
        st.success(f"Successfully pulled {model_name}")
    else:
        st.error(f"Failed to pull {model_name}")


def render_parameter_tuning(
    config: OllamaConfig,
    on_change: Optional[Callable[[str, Any], None]] = None,
    key_prefix: str = "params"
) -> OllamaConfig:
    """
    Render parameter tuning controls.

    Args:
        config: Current configuration
        on_change: Callback when parameter changes
        key_prefix: Unique key prefix

    Returns:
        Updated configuration
    """
    st.subheader("Generation Parameters")

    col1, col2 = st.columns(2)

    with col1:
        # Temperature
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=config.temperature,
            step=0.1,
            help="Higher = more creative, Lower = more focused",
            key=f"{key_prefix}_temp"
        )

        if temperature != config.temperature:
            config.temperature = temperature
            if on_change:
                on_change("temperature", temperature)

        # Top-P
        top_p = st.slider(
            "Top-P (Nucleus Sampling)",
            min_value=0.0,
            max_value=1.0,
            value=config.top_p,
            step=0.05,
            help="Probability mass to consider",
            key=f"{key_prefix}_top_p"
        )

        if top_p != config.top_p:
            config.top_p = top_p
            if on_change:
                on_change("top_p", top_p)

    with col2:
        # Top-K
        top_k = st.slider(
            "Top-K",
            min_value=1,
            max_value=100,
            value=config.top_k,
            step=1,
            help="Number of tokens to consider",
            key=f"{key_prefix}_top_k"
        )

        if top_k != config.top_k:
            config.top_k = top_k
            if on_change:
                on_change("top_k", top_k)

        # Repeat Penalty
        repeat_penalty = st.slider(
            "Repeat Penalty",
            min_value=1.0,
            max_value=2.0,
            value=config.repeat_penalty,
            step=0.05,
            help="Penalize repeated tokens",
            key=f"{key_prefix}_repeat"
        )

        if repeat_penalty != config.repeat_penalty:
            config.repeat_penalty = repeat_penalty
            if on_change:
                on_change("repeat_penalty", repeat_penalty)

    # Context window
    context_options = [2048, 4096, 8192, 16384, 32768, 65536, 131072]
    current_ctx_idx = context_options.index(config.num_ctx) if config.num_ctx in context_options else 1

    num_ctx = st.select_slider(
        "Context Window",
        options=context_options,
        value=context_options[current_ctx_idx],
        format_func=lambda x: f"{x:,} tokens",
        help="Maximum context length",
        key=f"{key_prefix}_ctx"
    )

    if num_ctx != config.num_ctx:
        config.num_ctx = num_ctx
        if on_change:
            on_change("num_ctx", num_ctx)

    # Max tokens
    max_tokens = st.number_input(
        "Max Output Tokens",
        min_value=-1,
        max_value=8192,
        value=config.num_predict,
        help="-1 for unlimited",
        key=f"{key_prefix}_max_tokens"
    )

    if max_tokens != config.num_predict:
        config.num_predict = max_tokens
        if on_change:
            on_change("num_predict", max_tokens)

    return config


def render_model_status(client: OllamaClient):
    """Render current model status and stats."""
    st.subheader("Model Status")

    # Running models
    running = client.get_running_models()

    if running:
        st.markdown("**Loaded Models:**")
        for model in running:
            st.markdown(f"- 🟢 {model}")
    else:
        st.caption("No models currently loaded")

    # Last generation stats
    stats = client.get_last_stats()
    if stats and stats.eval_count > 0:
        st.divider()
        st.markdown("**Last Generation:**")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Tokens/sec", f"{stats.tokens_per_second:.1f}")

        with col2:
            st.metric("Tokens", stats.eval_count)

        with col3:
            st.metric("Duration", f"{stats.total_duration:.1f}s")


def render_available_models(client: OllamaClient, registry: ModelRegistry):
    """Render list of available models."""
    st.subheader("Available Models")

    available = client.list_models()

    if not available:
        st.info("No models installed. Use `ollama pull <model>` to install.")
        return

    for model in available:
        info = registry.get_model(model.name)

        with st.expander(f"📦 {model.name}", expanded=False):
            col1, col2 = st.columns([3, 1])

            with col1:
                if info:
                    st.markdown(f"**{info.display_name}**")
                    st.caption(info.description)

                    # Capabilities
                    caps = []
                    for cap in info.capabilities:
                        caps.append(cap.name.lower())
                    st.caption(f"Capabilities: {', '.join(caps)}")
                else:
                    st.markdown(f"**{model.name}**")

                st.caption(f"Size: {model.size} | Family: {model.family or 'Unknown'}")

            with col2:
                if st.button("🗑️ Delete", key=f"del_{model.name}"):
                    if client.delete_model(model.name):
                        st.success("Deleted")
                        st.rerun()
                    else:
                        st.error("Failed to delete")


def render_model_categories(registry: ModelRegistry) -> ModelCategory:
    """Render model category tabs."""
    tabs = st.tabs(["💭 Thinking", "💬 Chat", "👁️ Vision", "💻 Code", "📊 Embedding"])

    categories = [
        ModelCategory.THINKING,
        ModelCategory.CHAT,
        ModelCategory.VISION,
        ModelCategory.CODE,
        ModelCategory.EMBEDDING
    ]

    for tab, category in zip(tabs, categories):
        with tab:
            models = registry.list_models(category=category)

            for model in models:
                st.markdown(f"**{model.display_name}** ({model.parameter_size})")
                st.caption(model.description)

                if model.recommended_for:
                    st.caption(f"Best for: {', '.join(model.recommended_for)}")

                st.divider()

    return categories[0]  # Default to first category


def render_quick_settings(
    config: OllamaConfig,
    key_prefix: str = "quick"
) -> OllamaConfig:
    """Render compact quick settings."""
    col1, col2, col3 = st.columns(3)

    with col1:
        temp = st.slider(
            "🌡️ Temperature",
            0.0, 2.0, config.temperature, 0.1,
            key=f"{key_prefix}_temp"
        )
        config.temperature = temp

    with col2:
        ctx = st.selectbox(
            "📏 Context",
            [2048, 4096, 8192, 16384],
            index=[2048, 4096, 8192, 16384].index(config.num_ctx)
            if config.num_ctx in [2048, 4096, 8192, 16384] else 1,
            format_func=lambda x: f"{x//1024}K",
            key=f"{key_prefix}_ctx"
        )
        config.num_ctx = ctx

    with col3:
        preset = st.selectbox(
            "🎯 Preset",
            ["Balanced", "Creative", "Precise", "Fast"],
            key=f"{key_prefix}_preset"
        )

        if preset == "Creative":
            config.temperature = 1.2
            config.top_p = 0.95
        elif preset == "Precise":
            config.temperature = 0.3
            config.top_p = 0.5
        elif preset == "Fast":
            config.temperature = 0.7
            config.num_ctx = 2048

    return config
