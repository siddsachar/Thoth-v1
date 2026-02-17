import ollama
from langchain_ollama import ChatOllama

DEFAULT_MODEL = "qwen3:8b"

POPULAR_MODELS = [
    "llama3.1:8b", "llama3.1:70b",
    "llama3.2:1b", "llama3.2:3b",
    "llama3.3:70b",
    "qwen3:0.6b", "qwen3:1.7b", "qwen3:4b", "qwen3:8b", "qwen3:14b", "qwen3:30b",
    "qwen2.5:7b", "qwen2.5:14b", "qwen2.5:32b", "qwen2.5:72b",
    "gemma3:1b", "gemma3:4b", "gemma3:12b", "gemma3:27b",
    "gemma2:2b", "gemma2:9b", "gemma2:27b",
    "mistral:7b",
    "mixtral:8x7b",
    "phi4:14b", "phi4-mini:3.8b",
    "deepseek-r1:1.5b", "deepseek-r1:7b", "deepseek-r1:8b",
    "deepseek-r1:14b", "deepseek-r1:32b", "deepseek-r1:70b",
]

_current_model = DEFAULT_MODEL
_llm_instance = None


def get_llm() -> ChatOllama:
    """Return the current LLM instance, creating one if needed."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ChatOllama(model=_current_model)
    return _llm_instance


def set_model(model_name: str):
    """Switch the active model (call after ensuring it's downloaded)."""
    global _current_model, _llm_instance
    _current_model = model_name
    _llm_instance = ChatOllama(model=model_name)


def get_current_model() -> str:
    return _current_model


def list_local_models() -> list[str]:
    """Return names of models already downloaded in Ollama."""
    try:
        response = ollama.list()
        return sorted({m.model for m in response.models})
    except Exception:
        return []


def list_all_models() -> list[str]:
    """Return a combined, sorted list of local + popular models."""
    local = list_local_models()
    return sorted(set(local + POPULAR_MODELS))


def is_model_local(model_name: str) -> bool:
    """Check whether a model is already downloaded."""
    local = list_local_models()
    return any(
        model_name == m
        or f"{model_name}:latest" == m
        or model_name == m.split(":")[0]
        for m in local
    )


def pull_model(model_name: str):
    """Download a model from Ollama. Yields progress dicts when streamed."""
    return ollama.pull(model_name, stream=True)