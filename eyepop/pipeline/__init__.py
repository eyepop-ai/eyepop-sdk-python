"""A client for the pipeline API [EXPERIMENTAL].

Imports nothing from the rest of the SDK but `eyepop.auth`:

    from eyepop.pipeline import EyePopClient, EyePopError
"""

from eyepop.pipeline.client import EyePopClient, EyePopError

__all__ = ["EyePopClient", "EyePopError"]
