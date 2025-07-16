from __future__ import annotations

"""model_adapter.py
A lightweight, extensible adapter framework that provides a unified interface
for interacting with various large language model (LLM) back-ends.

Usage example
-------------
>>> from model_adapter import get_adapter
>>> adapter = get_adapter("openai", api_key="sk-...", model="gpt-3.5-turbo")
>>> print(adapter.generate("Hello, how are you?"))

The framework currently ships with two concrete adapters:
1. ``openai`` – OpenAI ChatCompletion/GPT models (requires ``openai`` package)
2. ``huggingface`` – Local Hugging Face transformers (requires ``transformers``)

You can register your own adapters via the ``@register_adapter`` decorator.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Callable

__all__ = [
    "BaseAdapter",
    "register_adapter",
    "get_adapter",
]

# ---------------------------------------------------------------------------
# Core abstraction
# ---------------------------------------------------------------------------


class BaseAdapter(ABC):
    """Interface every concrete model adapter must implement."""

    def __init__(self, **kwargs: Any) -> None:
        self.config: Dict[str, Any] = kwargs

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:  # noqa: D401 – imperative mood
        """Generate a single string response for *prompt*.

        Concrete adapters decide how *prompt* is routed to the underlying model.
        Additional keyword arguments override adapter-specific generation params.
        """

    # Optional convenience for chat-style interfaces ---------------------------------
    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:  # noqa: D401
        """Adapters supporting ChatCompletion-style messages can override this."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement chat() – use generate()."
        )

    # -----------------------------------------------------------------------
    # Helper utilities each adapter *may* leverage
    # -----------------------------------------------------------------------
    @staticmethod
    def _ensure_dep(pkg: str, import_name: str | None = None) -> Any:  # noqa: D401
        """Import *pkg* lazily and raise informative error if missing."""
        try:
            return __import__(import_name or pkg)
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                f"Package '{pkg}' is required for this adapter. "
                f"Install via `pip install {pkg}`."
            ) from exc


# ---------------------------------------------------------------------------
# Registry machinery
# ---------------------------------------------------------------------------

_ADAPTER_REGISTRY: dict[str, Callable[..., BaseAdapter]] = {}


def register_adapter(name: str):  # noqa: D401 – decorator
    """Class decorator used to register *name* for a concrete adapter class."""

    def decorator(cls: type[BaseAdapter]):  # noqa: D401
        if name in _ADAPTER_REGISTRY:
            raise ValueError(f"Adapter '{name}' already registered.")
        if not issubclass(cls, BaseAdapter):
            raise TypeError("@register_adapter target must subclass BaseAdapter")
        _ADAPTER_REGISTRY[name] = cls  # type: ignore[assignment]
        return cls

    return decorator


def get_adapter(name: str, /, **kwargs: Any) -> BaseAdapter:  # noqa: D401
    """Factory returning an instance of the adapter registered as *name*."""
    try:
        cls = _ADAPTER_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"Adapter '{name}' not found. Available: {', '.join(_ADAPTER_REGISTRY)}"
        ) from exc
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Built-in adapters
# ---------------------------------------------------------------------------


@register_adapter("openai")
class OpenAIAdapter(BaseAdapter):
    """Adapter for OpenAI ChatCompletion API (GPT-3.5/4)."""

    def __init__(self, model: str = "gpt-3.5-turbo", api_key: str | None = None, **kwargs: Any):
        super().__init__(model=model, api_key=api_key, **kwargs)
        self.model = model
        self.api_key = api_key  # can also be set via env var
        # Lazy import to avoid mandatory dependency for non-OpenAI users
        self.openai = self._ensure_dep("openai")
        if self.api_key:
            self.openai.api_key = self.api_key

    # ---------------------------------------------------------------------
    # Core generation functions
    # ---------------------------------------------------------------------

    def generate(self, prompt: str, **kwargs: Any) -> str:  # noqa: D401
        """Single-prompt helper that wraps ChatCompletion behind the scenes."""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, **kwargs)

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 512),
        }
        params.update(kwargs)
        completion = self.openai.ChatCompletion.create(**params)  # type: ignore[attr-defined]
        return completion.choices[0].message.content.strip()


@register_adapter("huggingface")
class HuggingFaceAdapter(BaseAdapter):
    """Adapter for local Hugging Face transformer models."""

    def __init__(self, model: str = "gpt2", device: str | int | None = None, **kwargs: Any):
        super().__init__(model=model, device=device, **kwargs)
        # Lazy import to avoid heavy dependencies if not needed
        transformers = self._ensure_dep("transformers")
        from transformers import pipeline  # type: ignore

        self.generator = pipeline(
            "text-generation",
            model=model,
            device=device if isinstance(device, int) else -1,  # cpu=-1
            **kwargs,
        )

    def generate(self, prompt: str, **kwargs: Any) -> str:  # noqa: D401
        outputs = self.generator(prompt, **kwargs)
        # HF pipeline returns list[dict[str,str]] where "generated_text" key holds the data
        return outputs[0]["generated_text"].strip()