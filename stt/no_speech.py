"""No-speech guard for record-then-transcribe input (E-011 follow-up).

The CTC acoustic model is fine on silence, but the LM-assisted decoder has no blank prior, so
on digital silence it always finds a continuation ("non mi" on all 120 silence clips in E-011).
An energy check in the mel domain removes that failure without a VAD, which stays out of scope
for v0.1 (``docs/06`` §1).

The rule is deliberately blunt and calibrated on the cached shards rather than guessed: the
normalized log-mel of digital silence is flat at ``-1.5``, the quietest speech clip measured
(FLEURS-EN, 525 clips) still reaches ``+0.10``, and MUSAN noise/music reach ``+0.64``/``+1.04``.
A clip whose loudest mel bin is below ``-1.0`` therefore has no speech-like energy at all; the
threshold sits 20 dB above digital silence and 44 dB below the quietest speech clip.

Limitation, stated on purpose: this catches silence and near-silence, **not** loud noise or
music. Those need noise augmentation with empty targets in training (E-011, ``docs/10``).
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
