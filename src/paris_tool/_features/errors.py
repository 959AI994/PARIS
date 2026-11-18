class ParisError(Exception):
    """Base class for PARIS errors."""


class ValidationError(ParisError):
    """Raised when an input artifact is invalid."""


class DependencyError(ParisError):
    """Raised when an optional dependency is required but unavailable."""
