"""Tool argument models and Gemini function declarations.

Schemas are built explicitly from pydantic models rather than parsed from
function docstrings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, get_args, get_origin

from google.genai import types
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Structured result returned by every tool implementation."""

    ok: bool
    message: str

    def __str__(self) -> str:
        return self.message


class ReadFileArgs(BaseModel):
    """Arguments for ``read_file``."""

    path: str = Field(..., description="Path to the file to read.")
    start_line: int | None = Field(
        None, description="First line to read (1-indexed, inclusive)."
    )
    end_line: int | None = Field(
        None, description="Last line to read (1-indexed, inclusive)."
    )


class WriteFileArgs(BaseModel):
    """Arguments for ``write_file``."""

    path: str = Field(..., description="Path to the file to write.")
    content: str = Field(..., description="Content to write to the file.")


class ListDirectoryArgs(BaseModel):
    """Arguments for ``list_directory``."""

    path: str = Field(
        ".", description="Path to the directory to list. Defaults to the working directory."
    )


class DeleteFileArgs(BaseModel):
    """Arguments for ``delete_file``."""

    path: str = Field(..., description="Path to the file or empty directory to delete.")


class RunCommandArgs(BaseModel):
    """Arguments for ``run_command``."""

    command: str = Field(..., description="The PowerShell command to execute.")
    working_dir: str | None = Field(
        None, description="Working directory for the command. Defaults to the workspace root."
    )
    timeout: int = Field(
        30,
        ge=1,
        description="Maximum time to wait for the command, in seconds (max 300).",
    )


def _is_optional(annotation: type) -> bool:
    origin = get_origin(annotation)
    if origin is not None:
        return type(None) in get_args(annotation)
    return False


def _schema_for_annotation(annotation: type) -> types.Schema:
    """Map a small set of Python types to Gemini schema types."""
    if _is_optional(annotation):
        # Strip the ``None`` union for simplicity; the field is not required.
        args = [a for a in get_args(annotation) if a is not type(None)]
        annotation = args[0] if args else str

    if annotation is int:
        return types.Schema(type=types.Type.INTEGER)
    if annotation is bool:
        return types.Schema(type=types.Type.BOOLEAN)
    return types.Schema(type=types.Type.STRING)


def _build_schema(model: type[BaseModel]) -> types.Schema:
    """Convert a pydantic argument model to an explicit ``types.Schema``."""
    properties: dict[str, types.Schema] = {}
    required: list[str] = []
    for name, field_info in model.model_fields.items():
        annotation = field_info.annotation if field_info.annotation is not None else str
        schema = _schema_for_annotation(annotation)
        if field_info.description:
            schema.description = field_info.description
        if field_info.default is not None and not field_info.is_required():
            schema.default = field_info.default
        properties[name] = schema
        if field_info.is_required():
            required.append(name)

    return types.Schema(
        type=types.Type.OBJECT,
        properties=properties,
        required=required,
        property_ordering=list(properties.keys()),
    )


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    args_schema: type[BaseModel]
    destructive: bool


ALL_TOOLS: list[ToolSpec] = [
    ToolSpec(
        name="read_file",
        description=(
            "Read the contents of a file. Optionally specify start_line and end_line "
            "(1-indexed, inclusive) to read a specific range of lines."
        ),
        args_schema=ReadFileArgs,
        destructive=False,
    ),
    ToolSpec(
        name="write_file",
        description=(
            "Write content to a file. Creates parent directories if they don't exist. "
            "Overwrites existing files."
        ),
        args_schema=WriteFileArgs,
        destructive=True,
    ),
    ToolSpec(
        name="list_directory",
        description="List the contents of a directory, showing file types and sizes.",
        args_schema=ListDirectoryArgs,
        destructive=False,
    ),
    ToolSpec(
        name="delete_file",
        description="Delete a file or an empty directory.",
        args_schema=DeleteFileArgs,
        destructive=True,
    ),
    ToolSpec(
        name="run_command",
        description=(
            "Execute a PowerShell command and return its output (stdout + stderr combined). "
            "Timeout is in seconds (default 30, max 300)."
        ),
        args_schema=RunCommandArgs,
        destructive=True,
    ),
]


def build_function_declarations() -> Sequence[types.Tool]:
    """Build explicit Gemini tools from function declarations.

    The Live SDK expects ``list[types.Tool]``; each tool wraps one function
    declaration.
    """
    return [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=spec.name,
                    description=spec.description,
                    parameters=_build_schema(spec.args_schema),
                )
            ]
        )
        for spec in ALL_TOOLS
    ]
