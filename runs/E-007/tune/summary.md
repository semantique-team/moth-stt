# E-007 tuning — 200 clips per dev set (seed 0)

Selection sets: `mls_it_dev` + `ls_en_dev`; 200 clips each. FLEURS, CV and T were not used
for tuning. Macro is the unweighted mean of the IT and EN WERs.

| Config | IT | EN | Macro |
|---|---|---|---|
| greedy | 0.3617 | 0.2963 | 0.3290 |
| beam 3 | 0.3602 | 0.2927 | 0.3264 |
| beam 5 | 0.3597 | 0.2899 | 0.3248 |
| beam 8 | 0.3597 | 0.2897 | 0.3247 |
| beam 5, top-k 12 | 0.3597 | 0.2899 | 0.3248 |
| beam 5 + hot 0.5 | 0.3598 | 0.2897 | 0.3248 |
| beam 5 + hot 1.0 | 0.3598 | 0.2899 | 0.3249 |
| beam 5 + hot 2.0 | 0.3609 | 0.2904 | 0.3257 |
| beam 5 + hot 3.0 | 0.3624 | 0.2907 | 0.3265 |
| beam 5 + hot 1.0 + prefix 0.05 | 0.3600 | 0.2892 | 0.3246 |
| beam 5 + hot 2.0 + prefix 0.05 | 0.3609 | 0.2894 | 0.3252 |

Beam width 5 is the knee; top-k 8 is enough (12 is identical). Hotword bonuses are neutral on
these dev sets (they have almost no tech terms), so the held-out hotword run used the a-priori
bonus 1.0, not a value selected on T.
