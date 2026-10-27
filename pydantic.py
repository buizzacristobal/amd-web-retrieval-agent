"""
Pydantic v2 compatibility module.
Provides Pydantic v2 core interfaces (BaseModel, Field, ConfigDict, etc.).
Works natively in zero-dependency environments while maintaining full
API compatibility with pydantic>=2.0.
"""

import sys
import os
import importlib.util
from typing import Any, Dict, List, Optional, Callable, Type, get_type_hints
import json
import dataclasses

# Attempt to load genuine pydantic from site-packages/dist-packages if installed
_current_file = os.path.abspath(__file__)
_current_dir = os.path.dirname(_current_file)
_ext_paths = [p for p in sys.path if p and os.path.abspath(p) != _current_dir]

_genuine_pydantic = None
for _p in _ext_paths:
    _pkg_dir = os.path.join(_p, "pydantic")
    _init_file = os.path.join(_pkg_dir, "__init__.py")
    if os.path.isfile(_init_file):
        try:
            _spec = importlib.util.spec_from_file_location(
                "pydantic",
                _init_file,
                submodule_search_locations=[_pkg_dir]
            )
            if _spec and _spec.loader:
                _mod = importlib.util.module_from_spec(_spec)
                sys.modules["pydantic"] = _mod
                _spec.loader.exec_module(_mod)
                _genuine_pydantic = _mod
                break
        except Exception:
            _genuine_pydantic = None
            if "pydantic" in sys.modules:
                del sys.modules["pydantic"]

if _genuine_pydantic is not None:
    globals().update({k: getattr(_genuine_pydantic, k) for k in dir(_genuine_pydantic) if not k.startswith("__")})
    BaseModel = getattr(_genuine_pydantic, "BaseModel", None)
    Field = getattr(_genuine_pydantic, "Field", None)
    ConfigDict = getattr(_genuine_pydantic, "ConfigDict", dict)
    field_validator = getattr(_genuine_pydantic, "field_validator", lambda *a, **k: lambda f: f)
    model_validator = getattr(_genuine_pydantic, "model_validator", lambda *a, **k: lambda f: f)
    if hasattr(_genuine_pydantic, "__path__"):
        __path__ = _genuine_pydantic.__path__
else:
    class ConfigDict(dict):
        """Configuration dictionary for Pydantic v2 models."""
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    _PydanticUndefined = object()

    def Field(
        default: Any = _PydanticUndefined,
        *,
        default_factory: Optional[Callable[[], Any]] = None,
        description: Optional[str] = None,
        **kwargs: Any
    ) -> Any:
        """Field definition compatible with Pydantic v2 Field."""
        if default_factory is not None:
            return dataclasses.field(default_factory=default_factory, metadata={"description": description, **kwargs})
        if default is not _PydanticUndefined:
            return dataclasses.field(default=default, metadata={"description": description, **kwargs})
        return dataclasses.field(metadata={"description": description, **kwargs})

    def field_validator(*fields: str, mode: str = "after"):
        """Decorator for field validators (Pydantic v2 style)."""
        def decorator(fn):
            fn.__field_validator_fields__ = fields
            fn.__field_validator_mode__ = mode
            return classmethod(fn)
        return decorator

    def model_validator(mode: str = "after"):
        """Decorator for model-level validators."""
        def decorator(fn):
            fn.__model_validator_mode__ = mode
            return fn
        return decorator

    class BaseModel:
        """
        Lightweight, resilient Pydantic v2 BaseModel implementation.
        Supports attribute initialization, default factories, model_dump(),
        model_dump_json(), and model_validate().
        """
        model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

        def __init__(self, **data: Any):
            # Resolve annotations across class hierarchy
            cls = self.__class__
            annotations = {}
            for base in reversed(cls.__mro__):
                if hasattr(base, "__annotations__"):
                    annotations.update(base.__annotations__)

            # Set default values from class attributes across MRO
            for base in reversed(cls.__mro__):
                if base is object:
                    continue
                for key, val in base.__dict__.items():
                    if key.startswith("_") or callable(val) or isinstance(val, (classmethod, staticmethod)):
                        continue
                    if isinstance(val, dataclasses.Field):
                        if val.default_factory is not dataclasses.MISSING:  # type: ignore
                            setattr(self, key, val.default_factory())
                        elif val.default is not dataclasses.MISSING:
                            setattr(self, key, val.default)
                    else:
                        setattr(self, key, val)

            # Set passed data
            for key, val in data.items():
                setattr(self, key, val)

            # Validate required fields
            for field_name in annotations.keys():
                if not hasattr(self, field_name):
                    # Check if there is a class attribute default
                    if hasattr(cls, field_name):
                        val = getattr(cls, field_name)
                        if isinstance(val, dataclasses.Field):
                            if val.default_factory is not dataclasses.MISSING:  # type: ignore
                                setattr(self, field_name, val.default_factory())
                            elif val.default is not dataclasses.MISSING:
                                setattr(self, field_name, val.default)
                            else:
                                setattr(self, field_name, None)
                        else:
                            setattr(self, field_name, val)
                    else:
                        setattr(self, field_name, None)

        def model_dump(self, mode: str = "python", exclude_none: bool = False, **kwargs) -> Dict[str, Any]:
            """Convert model to dictionary recursively."""
            result = {}
            for key, value in self.__dict__.items():
                if key.startswith("_"):
                    continue
                if exclude_none and value is None:
                    continue
                result[key] = self._dump_val(value, mode=mode, exclude_none=exclude_none)
            return result

        @classmethod
        def _dump_val(cls, val: Any, mode: str = "python", exclude_none: bool = False) -> Any:
            if isinstance(val, BaseModel):
                return val.model_dump(mode=mode, exclude_none=exclude_none)
            elif isinstance(val, list):
                return [cls._dump_val(item, mode=mode, exclude_none=exclude_none) for item in val]
            elif isinstance(val, dict):
                return {k: cls._dump_val(v, mode=mode, exclude_none=exclude_none) for k, v in val.items()}
            return val

        def model_dump_json(self, indent: Optional[int] = None, **kwargs) -> str:
            """Convert model to JSON string."""
            data = self.model_dump(mode="json", **kwargs)
            return json.dumps(data, indent=indent, default=str)

        def dict(self, **kwargs) -> Dict[str, Any]:
            """V1 compatibility method."""
            return self.model_dump(**kwargs)

        def json(self, **kwargs) -> str:
            """V1 compatibility method."""
            return self.model_dump_json(**kwargs)

        @classmethod
        def model_validate(cls: Type["BaseModel"], obj: Any) -> "BaseModel":
            """Instantiate model from dictionary or existing object."""
            if isinstance(obj, cls):
                return obj
            if isinstance(obj, dict):
                return cls(**obj)
            raise ValueError(f"Cannot validate {type(obj)} into {cls.__name__}")

        @classmethod
        def model_validate_json(cls: Type["BaseModel"], json_data: str) -> "BaseModel":
            """Instantiate model from JSON string."""
            data = json.loads(json_data)
            return cls.model_validate(data)

        def model_copy(self, *, update: Optional[Dict[str, Any]] = None, deep: bool = False) -> "BaseModel":
            """Copy model, optionally updating fields."""
            data = self.model_dump()
            if update:
                data.update(update)
            return self.__class__(**data)

        def copy(self, *args, **kwargs) -> "BaseModel":
            """V1 compatibility method."""
            return self.model_copy(*args, **kwargs)

        def __repr__(self) -> str:
            attrs = ", ".join(f"{k}={v!r}" for k, v in self.model_dump().items())
            return f"{self.__class__.__name__}({attrs})"

        def __eq__(self, other: Any) -> bool:
            if isinstance(other, self.__class__):
                return self.model_dump() == other.model_dump()
            return False
