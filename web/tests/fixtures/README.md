# Browser test fixtures

| File | What | Source and licence |
|---|---|---|
| `clip.wav` | 5 s synthetic tone, 16 kHz mono PCM — the smoke/RTFx fixture | generated in-repo (sine); no third-party rights |
| `speech.wav` | 3 s of real Italian speech, 16 kHz mono PCM — the VAD positive case | FLEURS `it_it` test, clip `10713904656493882072`, seconds 4.25–7.25; **CC-BY 4.0** (`google/fleurs`), attribution in the release `NOTICE` |
| `lm_scores.json` | reference trigram scores for the JS LM parser test | produced by `stt/lm.py` on the E-008 LM |

`speech.wav` is the only committed third-party audio: it is small (94 KB) and it is the **only**
way to prove that the VAD guard does not reject real speech without shipping private recordings
(`tests/test_vad.py`, `web/tests/vad.spec.ts`). The window was chosen because every 32 ms frame
is speech (fraction 1.000 at the 0.5 window threshold), not because it flatters the model.
