"""signate_submitter package init."""

from .agent import AgentResult, TaskInput, build_success_result, main
from .data_manager import DataManager, DataMeta
from .pipeline import SignatePipeline, SignatePipelineResult

__all__ = [
    "main",
    "build_success_result",
    "AgentResult",
    "TaskInput",
    "DataManager",
    "DataMeta",
    "SignatePipeline",
    "SignatePipelineResult",
]
