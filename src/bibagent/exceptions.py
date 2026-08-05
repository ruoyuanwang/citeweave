class BibAgentError(Exception):
    """Base exception."""


class ConfigurationError(BibAgentError):
    """Invalid project or runtime configuration."""


class AcquisitionError(BibAgentError):
    """A source could not be acquired completely."""


class CompletenessError(AcquisitionError):
    """The acquired record count does not satisfy the source contract."""


class ProcessingError(BibAgentError):
    """Metadata cleaning or structural materialization failed."""


class QualityGateError(BibAgentError):
    """A quality gate prevented downstream analysis."""


class EvidenceError(BibAgentError):
    """Generated text is not supported by its evidence packet."""
