"""GaitDataSource: the abstraction that decouples the pipeline from any dataset.

Stage 1-4 code asks a GaitDataSource for trials and does not care whether they
came from Fukuchi 2017, a synthetic generator, or (one day) a Mirai insole
export. New sources are added by subclassing, not by editing consumers
(open/closed; dependency inversion).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from achilles.data.trial import GaitTrial


class GaitDataSource(ABC):
    """A provider of time-normalised gait trials."""

    @abstractmethod
    def iter_trials(self) -> Iterator[GaitTrial]:
        """Yield every available GaitTrial."""
        raise NotImplementedError

    def load_trials(self) -> list[GaitTrial]:
        """Materialise all trials into a list."""
        return list(self.iter_trials())

    def subjects(self) -> list[str]:
        """Sorted unique subject ids present in this source."""
        return sorted({t.subject_id for t in self.iter_trials()})
