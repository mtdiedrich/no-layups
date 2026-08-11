from pydantic import BaseModel


class UploadResponse(BaseModel):
    job_id: str


class KeyframesUpdate(BaseModel):
    """Section 10 POST /api/swings/{swing_id}/keyframes body."""

    address: int
    top: int
    impact: int
