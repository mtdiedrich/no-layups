class PipelineError(Exception):
    """Raised for every pipeline failure. code is one of the Section 7.8 error codes."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
