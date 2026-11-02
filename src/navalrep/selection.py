"""Rank emotional moments and end clips exactly at detected shot boundaries.

All selection takes place in source time. Duration constraints are converted to
source seconds first, so the *sped-up output* still lasts 35–55 seconds.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
import math
import re
import unicodedata

from .models import AudioPeak, Clip, Transcript


@dataclass(frozen=True)
class _Event:
    time: float
    score: float
    reason: str


def _tokens(text: str) -> list[str]:
    """Match Spanish text independent of case, accents and punctuation."""
    plain = "".join(
        c for c in unicodedata.normalize("NFKD", text.casefold())
        if not unicodedata.combining(c)
    )
    return re.findall(r"\w+", plain, flags=re.UNICODE)


def _valid_time(value: float, duration: float) -> bool:
    return math.isfinite(value) and 0 <= value <= duration


def _text_events(transcript: Transcript, phrases: tuple[str, ...], duration: float) -> list[_Event]:
    events: list[_Event] = []
    # Word timestamps localize a trigger even inside a long Whisper segment.
    timed_text = transcript.words or transcript.segments
    lexical: list[tuple[str, float, float]] = []
    for item in timed_text:
        if not _valid_time(item.end, duration) or item.end < item.start:
            continue
        if "?" in item.text:
            events.append(_Event(item.end, 2.25, "question"))
        if "!" in item.text:
            events.append(_Event(item.end, 2.3, "exclamation"))
        lexical.extend((token, item.start, item.end) for token in _tokens(item.text))

    seen: set[tuple[float, str]] = set()
    for phrase in phrases:
        needle = _tokens(phrase)
        if not needle:
            continue
        count = len(needle)
        for i in range(len(lexical) - count + 1):
            window = lexical[i:i + count]
            if [token for token, _, _ in window] != needle:
                continue
            # Avoid assembling a phrase across unrelated lines minutes apart.
            if any(window[j + 1][1] - window[j][2] > 3 for j in range(count - 1)):
                continue
            end = window[-1][2]
            key = (end, " ".join(needle))
            if key not in seen:
                seen.add(key)
                events.append(_Event(end, 3.5, "trigger:" + key[1]))
    return events


def _overlap(a: Clip, b: Clip) -> float:
    intersection = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    return intersection / min(a.end - a.start, b.end - b.start)


def select_clips(
    boundaries: list[float],
    transcript: Transcript,
    peaks: list[AudioPeak],
    duration: float,
    config,
) -> list[Clip]:
    """Return chronological, nonredundant clips with an emotional final beat.

    Scene starts are preferred when legal. If none yields the required final
    duration, only the contextual *start* is trimmed inside a scene; the ending
    always remains a detected cut (or the exact end of the source file). No
    arbitrary fallback clips are generated when the evidence is insufficient.
    """
    if not math.isfinite(duration) or duration <= 0:
        return []
    if (
        not 0 < config.min_duration <= config.target_duration <= config.max_duration
        or not math.isfinite(config.speed) or config.speed <= 0
        or config.peak_lookback <= 0
        or config.max_silence_after_peak < 0
        or not 0 <= config.max_overlap <= 1
    ):
        raise ValueError("Invalid duration, speed, peak window or overlap settings")
    if config.max_clips <= 0:
        return []

    min_source = config.min_duration * config.speed
    max_source = config.max_duration * config.speed
    target_source = config.target_duration * config.speed
    if duration < min_source:
        return []
    cuts = sorted({0.0, duration, *(b for b in boundaries if _valid_time(b, duration))})
    events = _text_events(transcript, tuple(config.trigger_phrases), duration)
    audio_threshold = getattr(config, "audio_peak_db", 7.0)
    for peak in peaks:
        if (
            _valid_time(peak.time, duration)
            and math.isfinite(peak.strength)
            and peak.strength >= audio_threshold
        ):
            events.append(_Event(peak.time, min(4.0, 1.5 + peak.strength / 7.0), "volume_jump"))
    # Speech continuing after an emotional line is context; a silent tail is not.
    speech = transcript.words or transcript.segments
    speech_ends = sorted(item.end for item in speech if _valid_time(item.end, duration))
    events.sort(key=lambda event: event.time)
    event_times = [event.time for event in events]
    candidates: list[Clip] = []
    for end in cuts:
        if end < min_source:
            continue
        recent = events[bisect_left(event_times, end - config.peak_lookback):bisect_right(event_times, end)]
        if not recent:
            continue
        speech_index = bisect_right(speech_ends, end) - 1
        last_speech = speech_ends[speech_index] if speech_index >= 0 else 0.0
        last_content = max(recent[-1].time, last_speech)
        if end - last_content > config.max_silence_after_peak:
            continue

        # Take the strongest example of each signal, avoiding punctuation-count
        # inflation in dense dialogue while rewarding independent evidence.
        signals: dict[str, float] = {}
        reasons: set[str] = set()
        for event in recent:
            age = end - event.time
            weighted = event.score * (1.0 - 0.2 * age / config.peak_lookback)
            category = "trigger" if event.reason.startswith("trigger:") else event.reason
            signals[category] = max(signals.get(category, 0.0), weighted)
            reasons.add(event.reason)
        values = sorted(signals.values(), reverse=True)
        score = values[0] + sum(values[1:]) * 0.5
        if score < config.min_score:
            continue
        earliest, latest = max(0.0, end - max_source), end - min_source
        legal_starts = cuts[bisect_left(cuts, earliest):bisect_right(cuts, latest)]
        if legal_starts:
            start = min(legal_starts, key=lambda cut: (abs(end - cut - target_source), cut))
            reasons.add("start_at_scene_cut")
        else:
            start = min(latest, max(earliest, end - target_source))
            reasons.add("context_start_trimmed_for_duration")
        clip = Clip(start=start, end=end, score=round(score, 6), reasons=tuple(sorted(reasons)))
        # Floating point edges must never let an invalid source window through.
        out_duration = (clip.end - clip.start) / config.speed
        if config.min_duration - 1e-9 <= out_duration <= config.max_duration + 1e-9:
            candidates.append(clip)

    chosen: list[Clip] = []
    for clip in sorted(candidates, key=lambda c: (-c.score, c.end, c.start)):
        if all(_overlap(clip, other) <= config.max_overlap + 1e-12 for other in chosen):
            chosen.append(clip)
            if len(chosen) == config.max_clips:
                break
    return sorted(chosen, key=lambda c: (c.start, c.end))
