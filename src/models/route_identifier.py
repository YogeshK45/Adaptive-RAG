"""
Route identifier model.
"""

from typing import Literal
from pydantic import BaseModel, Field


class RouteIdentifier(BaseModel):
    """Model for routing queries to appropriate nodes."""

    route: Literal["index", "general", "search"] = Field(
        description="The selected route: 'index', 'general', or 'search'"
    )
