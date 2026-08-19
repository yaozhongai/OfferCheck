"""Task-profile definitions for the reusable Nexa Agent core.

Profiles isolate scenario instructions and terminal schemas from the shared
agent loop.  A stage always selects the OfferCheck profile for backwards
compatibility; stage-less calls default to the generic research profile.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TaskProfile:
    """Scenario contract consumed by the shared ReAct loop."""

    profile_id: str
    base_prompt_file: str
    finalizer_tool: str
    requires_external_evidence: bool = True
    offercheck: bool = False


GENERIC_RESEARCH_PROFILE = TaskProfile(
    profile_id="generic_research",
    base_prompt_file="generic_research_system.txt",
    finalizer_tool="submit_answer",
)

OFFERCHECK_PROFILE = TaskProfile(
    profile_id="offercheck",
    base_prompt_file="react_system.txt",
    finalizer_tool="submit_verdict",
    offercheck=True,
)

_PROFILES = {
    GENERIC_RESEARCH_PROFILE.profile_id: GENERIC_RESEARCH_PROFILE,
    OFFERCHECK_PROFILE.profile_id: OFFERCHECK_PROFILE,
}


def resolve_task_profile(
    task_profile: Optional[str] = None,
    stage: Optional[str] = None,
) -> TaskProfile:
    """Resolve the scenario contract without silently mixing profiles.

    Existing application calls pass an OfferCheck stage and therefore retain
    their current behavior.  New stage-less engine calls are generic.
    """

    if task_profile:
        try:
            return _PROFILES[task_profile]
        except KeyError as exc:
            supported = ", ".join(sorted(_PROFILES))
            raise ValueError(
                f"Unknown task profile {task_profile!r}; supported: {supported}"
            ) from exc
    if stage:
        return OFFERCHECK_PROFILE
    return GENERIC_RESEARCH_PROFILE

