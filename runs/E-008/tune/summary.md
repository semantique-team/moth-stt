# E-008 tuning — LM weight α on dev, then full dev

Subset: 200 clips from `mls_it_dev` + 200 from `ls_en_dev` (seed 0). Macro is the unweighted
mean of IT and EN WER. FLEURS, CV and T were not used for tuning.

| α | IT | EN | Macro |
|---|---|---|---|
| 0.05 | 0.3394 | 0.2701 | 0.3048 |
| 0.1 | 0.3284 | 0.2589 | 0.2937 |
| **0.2** | **0.3187** | 0.2465 | **0.2826** |
| 0.3 | 0.3262 | **0.2399** | 0.2831 |
| 0.5 | 0.3631 | 0.2597 | 0.3114 |
| 0.8 | 0.4639 | 0.3158 | 0.3898 |

Full dev confirmation (no LM: IT 0.3521, EN 0.2646, macro 0.3084):

| α | IT | EN | Macro |
|---|---|---|---|
| 0.2 | 0.3169 | 0.2240 | **0.2705** |
| 0.3 | 0.3232 | 0.2204 | 0.2718 |

α=0.2 was kept: best macro, and the held-out evaluation used it once. The LM is a BPE trigram
with counts pruned at 2 (195k bigram entries, 800k trigram entries; 3.53 MB gzip).
