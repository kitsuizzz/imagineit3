"""Exceptions shared by the GPU worker and its backends."""


class InferenceCancelled(Exception):
    """Raised when the app asks the worker to stop a job."""
