"""Pure CPU tests for cliffhanger placement and duration invariants."""

import unittest
from types import SimpleNamespace

from navalrep.config import PipelineConfig
from navalrep.models import AudioPeak, Segment, Transcript, Word
from navalrep.selection import select_clips


def transcript(text="¡No puede ser!", start=58.0, end=59.0):
    return Transcript(words=(), segments=(Segment(start, end, text),), language="es")


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.config = PipelineConfig()

    def test_short_scene_extends_context_and_duration_accounts_for_speed(self):
        clips = select_clips([0, 10, 20, 50, 60], transcript(), [], 60, self.config)
        self.assertEqual(len(clips), 1)
        clip = clips[0]
        self.assertEqual(clip.end, 60)
        self.assertIn(clip.start, [10, 20])
        self.assertTrue(35 <= (clip.end - clip.start) / 1.04 <= 55)
        self.assertIn("start_at_scene_cut", clip.reasons)

    def test_long_scene_trims_only_context_start(self):
        clips = select_clips([0, 60], transcript(), [], 60, self.config)
        self.assertEqual(len(clips), 1)
        self.assertAlmostEqual(clips[0].start, 60 - 45 * 1.04)
        self.assertEqual(clips[0].end, 60)
        self.assertIn("context_start_trimmed_for_duration", clips[0].reasons)

    def test_source_35_seconds_is_too_short_after_speed_up(self):
        self.assertEqual(select_clips([0, 35], transcript(end=34), [], 35, self.config), [])

    def test_emotion_in_middle_of_scene_does_not_qualify(self):
        self.assertEqual(select_clips([0, 60], transcript(start=29, end=30), [], 60, self.config), [])

    def test_no_evidence_produces_no_arbitrary_clips(self):
        self.assertEqual(select_clips([0, 60], transcript("Hoy hace buen tiempo."), [], 60, self.config), [])

    def test_question_without_trigger_still_qualifies(self):
        clips = select_clips([0, 60], transcript("¿Dónde está?"), [], 60, self.config)
        self.assertEqual(len(clips), 1)
        self.assertIn("question", clips[0].reasons)

    def test_silent_tail_is_rejected(self):
        self.assertEqual(select_clips([0, 60], transcript(start=55, end=56), [], 60, self.config), [])

    def test_speech_after_peak_can_carry_context_to_cut(self):
        data = Transcript(words=(), segments=(
            Segment(54, 56, "¡No puede ser!"), Segment(57, 59.5, "Mira esa puerta.")), language="es")
        self.assertEqual(len(select_clips([0, 60], data, [], 60, self.config)), 1)

    def test_word_times_prevent_false_peak_at_long_segment_end(self):
        data = Transcript(
            words=(Word(20, 21, "¡No"), Word(21, 22, "puede"), Word(22, 23, "ser!"), Word(58, 59, "Hola.")),
            segments=(Segment(20, 59, "¡No puede ser! Hola."),), language="es")
        self.assertEqual(select_clips([0, 60], data, [], 60, self.config), [])

    def test_trigger_matches_across_words_and_accents(self):
        data = Transcript(words=(Word(58, 58.5, "¿QUIÉN"), Word(58.5, 59, "es?")), segments=(), language="es")
        clips = select_clips([0, 60], data, [], 60, self.config)
        self.assertEqual(len(clips), 1)
        self.assertIn("trigger:quien es", clips[0].reasons)

    def test_loud_jump_can_qualify_without_text(self):
        data = Transcript(words=(), segments=(), language="es")
        clips = select_clips([0, 60], data, [AudioPeak(59, 12)], 60, self.config)
        self.assertEqual(len(clips), 1)
        self.assertIn("volume_jump", clips[0].reasons)

    def test_overlapping_candidates_keep_stronger_one(self):
        data = Transcript(words=(), segments=(
            Segment(58, 59, "¡No puede ser!"), Segment(63, 64, "¿Quién es?")), language="es")
        clips = select_clips([0, 60, 65], data, [AudioPeak(59, 14)], 65, self.config)
        self.assertEqual(len(clips), 1)
        self.assertEqual(clips[0].end, 60)

    def test_invalid_configuration_raises(self):
        config = SimpleNamespace(**{**self.config.__dict__, "speed": 0})
        with self.assertRaises(ValueError):
            select_clips([0, 60], transcript(), [], 60, config)

    def test_bad_boundaries_are_ignored_and_output_end_is_source_end(self):
        clips = select_clips([float("nan"), -1, 1000], transcript(), [], 60, self.config)
        self.assertEqual(len(clips), 1)
        self.assertEqual(clips[0].end, 60)


if __name__ == "__main__":
    unittest.main()
