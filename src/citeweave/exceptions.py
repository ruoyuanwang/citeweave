class CiteWeaveError(Exception):
    """Base exception."""


class ConfigurationError(CiteWeaveError):
    """Invalid project or runtime configuration."""


class AcquisitionError(CiteWeaveError):
    """A source could not be acquired completely."""


class CompletenessError(AcquisitionError):
    """The acquired record count does not satisfy the source contract."""


class ProcessingError(CiteWeaveError):
    """Metadata cleaning or structural materialization failed."""


class QualityGateError(CiteWeaveError):
    """A quality gate prevented downstream analysis."""


class EvidenceError(CiteWeaveError):
    """Generated text is not supported by its evidence packet."""
