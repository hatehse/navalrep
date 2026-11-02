from dataclasses import dataclass


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Transcript:
    words: tuple[Word, ...]
    segments: tuple[Segment, ...]
    language: str = "es"
    device: str | None = None
    compute_type: str | None = None


@dataclass(frozen=True)
class AudioPeak:
    time: float
    strength: float


@dataclass(frozen=True)
class Clip:
    start: float
    end: float
    score: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class MediaInfo:
    duration: float
    width: int
    height: int
    has_audio: bool
