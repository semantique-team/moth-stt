"""Legacy no-speech energy guard (E-011 follow-up; superseded by :mod:`stt.vad`, D25).

The CTC acoustic model is fine on silence, but the LM-assisted decoder has no blank prior, so
on digital silence it always finds a continuation ("non mi" on all 120 silence clips in E-011).
An energy check in the mel domain removes that failure without a VAD, which was out of scope
for v0.1 (``docs/06`` §1).

The shipping guard is now **Silero VAD** (:mod:`stt.vad`): the energy check cannot separate
loud noise from speech, while the VAD gates 100% of both. This module is kept because
``runs/E-005/eval_sets.py --no-speech-guard`` reproduces the E-011/E-016 artifacts, and because
its threshold documents the silence calibration.

The rule is deliberately blunt and calibrated on the cached shards rather than guessed: the
normalized log-mel of digital silence is flat at ``-1.5``, the quietest speech clip measured
(FLEURS-EN, 525 clips) still reaches ``+0.10``, and MUSAN noise/music reach ``+0.64``/``+1.04``.
A clip whose loudest mel bin is below ``-1.0`` therefore has no speech-like energy at all; the
threshold sits 20 dB above digital silence and 44 dB below the quietest speech clip.

Limitation, stated on purpose: this catches silence and near-silence, **not** loud noise or
music — that is what the VAD is for.
"""

from __future__ import annotations

import numpy as np

__all__ = ["SILENCE_MAX", "is_no_speech"]

#: Loudest normalized log-mel bin below which a clip carries no speech-like energy.
#: Kept identical to ``SILENCE_MAX`` in ``web/mel.js`` (``tests/test_no_speech.py`` checks it).
SILENCE_MAX = -1.0


def is_no_speech(mel: np.ndarray) -> bool:
    """True when ``mel [80, T]`` has no speech-like energy and should not be decoded."""
    return float(np.asarray(mel).max()) < SILENCE_MAX
