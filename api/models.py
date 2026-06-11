from pydantic import BaseModel
from typing import Optional, List, Any


class SolveRequest(BaseModel):
    op:   str
    var:  Optional[str] = None
    vars: Optional[List[str]] = None
    expr: Optional[Any] = None
    u:    Optional[Any] = None
    v:    Optional[Any] = None
    text_input: Optional[str] = None   # plain text from Member 4 website


class StepInfo(BaseModel):
    rule:        str
    description: str


class SolveResponse(BaseModel):
    status:     str                           # "solved" | "placeholder" | "unverified" | "error"
    rule:       Optional[str]          = None
    confidence: Optional[float]        = None
    verified:   Optional[bool]         = None
    steps:      Optional[List[StepInfo]] = None
    expr:       Optional[Any]          = None
    message:    Optional[str]          = None


class ValidateRequest(BaseModel):
    expr: Any


class ValidateResponse(BaseModel):
    valid:  bool
    reason: Optional[str] = None