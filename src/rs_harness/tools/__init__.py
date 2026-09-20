"""Basic local tools and their shared input constraint."""

from typing import Annotated

from pydantic import AfterValidator, Field


def _non_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("value must not be blank")
    return value


NonBlank = Annotated[str, Field(strict=True, min_length=1), AfterValidator(_non_blank)]
