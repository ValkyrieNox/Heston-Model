# FinFlow Heston — Results Summary (as of 2026-06-03)

Two-stage Flow-Matching world model for **3-regime Markov-switching Heston** (num_actions=3, seed 20260530).
All results below are on the regime task unless noted. Eval = free-running `rollout.py` → `evaluate_rollout.py`
(no teacher forcing). **raw** = uncalibrated; **cal** = `--calibrate-moments` (affine-pin pooled return mean/std
to the true data values — injects the 2 pricing-dominant numbers; legitimate for option pricing but masks raw
scale error). Pricing floor (real vs MC-oracle) = **0.165**. Real kurtosis = **4.60**.

## ① Teachers (Stage-1 FM)
| model | raw | cal | kurt | path (under p3_full_parallel/) |
|---|---|---|---|---|
| **FM teacher** (128/4) | **0.475** | 0.583 | 3.88 | eval_champion/eval_fm ; ckpt training/vol_fm,ret_fm |
| LWFM δ0.05 | 0.882 | (1.010) | 4.33 | eval_lwfm/eval_d0.05 ; ckpt training/vol_lwfm_d0.05 |
| LWFM δ0.03/0.08/0.12 | 0.928/0.888/0.908 | — | 4.38/4.28/4.29 | eval_lwfm/ |
| strong teacher (256/6, bs8192) | 1.057 | 0.582 | 4.08 | eval_b_strongteacher ; ckpt training/b_partner_recipe_teacher_0602 |
| **B1 (FM + sched-sampling 0.8)** | 0.872 | 0.348 | **4.41** | eval_B_0602/B1_sched08 ; ckpt training/B_0602 |
| B3 (strong+sched0.8+LWFM) FAILED | 7.999 | 0.721 | 4.30 | eval_B_0602/B3_strong_lwfm |

## ② Teacher + algorithm (combined / pathwise / ablations)  — "combined" = OUR original method
(FM/LWFM teacher + path-loss finetune: signature-kernel MMD + Sig-W1 + Energy + moment matching, REPLACING the QGAN WGAN-GP critic. code: pathwise_teacher_combined.py)
| model | raw | cal | kurt | path |
|---|---|---|---|---|
| pathwise (orig, QGAN-critic) | 1.746 | 0.420 | 4.21 | eval_pathwise/ |
| pathwise_guarded FAILED | 5.837 | 4.988 | 4.16 | eval_pathwise_guarded/ |
| combined (on LWFM teacher) | 1.717 | 0.362 | 4.22 | eval_combined_0602/ ; ckpt training/combined_0602 |
| **combined (on strong teacher)** | 2.595 | **0.232** | 3.66 | eval_b_partner_recipe_0602/ ; ckpt training/b_partner_recipe_pathwise_0602 |
| C1_push (stronger path-loss,15ep) | 1.678 | 0.394 | 3.68 | eval_C_0602/C1_push |
| C2_rawfix (heavy moment/anchor) | 2.595 | 0.180 | 3.63 | eval_C_0602/C2_rawfix |
| C3_more (E06, unfinished/uncollected) | ? | ? | ? | — (E06 down) |
| ablation e1_sigmmd / e2_sigw1 / e3_energy / ctl | 5.51/4.97/1.81/2.57 | 15.1/11.8/1.56/2.90 | 4.78/4.59/3.39/3.49 | eval_ablation_0601/ |
| mf_lwfm FAILED (tails kurt 12) | 5.788 | 7.894 | 12.14 | eval_mf_lwfm/ |

## ③ Distilled students (1-step + Lagrangian flow-map + NFE sweep)
| model | raw | cal | kurt | path |
|---|---|---|---|---|
| FM-MF | 9.084 | 1.442 | 3.52 | eval_champion/eval_mf, eval_calibrated/eval_mf |
| FM-CD | 4.810 | 0.537 | 4.48 | eval_champion/eval_cd, eval_calibrated/eval_cd |
| **combined-CD** | 1.875 | **0.170** | 3.11 | eval_distill_compare_0602/combined_cd ; ckpt training/distill_combined_cd |
| combined-MF FAILED | 7.536 | 37.6 | 2.49 | eval_distill_compare_0602/combined_mf |
| cdsc (CD of strong+combined) | 7.711 | 0.337 | 2.84 | eval_A_0602/cdsc ; ckpt training/A_cd_sc |
| **fmlag (Lagrangian-distill FM)** raw: k1=2.74/**k2=1.82**/k4=3.09/k8=3.44 ; cal: **k1=0.313**/k2=2.36/k4=3.75/k8=3.86 (kurt 4.86→8.10) | | | | eval_A_0602/fmlag_* ; ckpt training/A_flowmap_fmlag |
| sclag (Lagrangian strong+combined) raw k1..k8: 4.15/4.22/4.37/4.69 | | | ~3.3 | eval_A_0602/sclag_* |

## ④ Baselines (Quant GAN; Wiese 2020 + our moment-penalty + Lambert-W)
| model | raw | cal | kurt | path |
|---|---|---|---|---|
| QGAN ours best | 3.069 | 2.243 | 5.62 | eval_compare_0601/qgan_ours_best_raw, eval_calibrated/quant_gan |
| QGAN ours last | 3.608 | 0.074* | 5.30 | eval_qgan_audit_0601/, eval_champion/quant_gan_last |
| QGAN paper (moment=0) best/last | 5.65/5.43 | 1.607/0.674 | 10.21/6.39 | eval_compare_0601/qgan_paper_* |
| *(ref) partner FM teacher (single-Heston, DIFFERENT task)* | 2.540 | 0.174 | 3.93 | partner/ (on its own machine) |

\* QGAN-last cal 0.074 is a calibration artifact (raw 3.6); below the floor because calibration zeroes 2nd-moment sampling error.

## SOTA by metric
- **raw (self-consistent / free simulation)**: FM teacher **0.475**
- **cal (practical option pricing)**: combined-CD **0.170** > C2 0.180 > combined(strong) 0.232
- **kurtosis / balance**: B1 (kurt 4.41, cal 0.348)
- **few-step**: fmlag — raw best at 2 steps (1.82), cal best at 1 step (0.313)

## Key findings
1. **raw ↔ (cal/kurt) systematic trade-off**: every method that improves calibrated pricing or kurtosis costs raw self-consistency. FM teacher's raw 0.475 is a hard-to-beat sweet spot.
2. **QGAN 0.074 & partner 0.199 are calibration artifacts** (raw 3.6 / 2.54). Our FM teacher is self-consistent (raw≈cal). Calibration injects the true return mean/std (2 numbers); pricing RMSE is dominated by terminal variance which calibration fixes.
3. **combined = our original contribution**: two-stage FM world model + Lambert-W variance kernel + strictly-proper signature/energy path-loss finetune (replaces adversarial critic). Improves cal monotonically (0.583→0.420→0.362→0.232).
4. **CD distillation works, MeanFlow fails** (MF explodes tails, esp. on LWFM-based teachers). combined-CD reaches cal 0.170 (~floor).
5. **NFE knee differs by metric**: raw→2 steps, cal→1 step; more steps make flow-map students diverge (tails).
6. Partner trained on the EASIER single-Heston (no regimes, Carr-Madan); not directly comparable to our regime task.

## Code (GitHub branch `consolidated-all-code`, repo ValkyrieNox/Heston-Model)
- pathwise_teacher_combined.py — combined finetune (sig-MMD/Sig-W1/Energy + moment).
- distill_flow_map.py — **Lagrangian flow-map self-distillation** (Boffi/Albergo/Vanden-Eijnden, openreview Di5apl8HSH).
- rollout_fewstep.py — few-step (NFE 1/2/4/8) sampler for flow-map/MeanFlow students.
- REPRODUCE.md — full data-gen + train + eval commands. Data NOT in git (regen: generate_heston_data.py --regimes --seed 20260530 + generate_mc_oracle.py).

## Open / next
- Get raw AND cal both low (combined is cal-only). Idea: bake signature-kernel loss into TEACHER training (not post-hoc), per the strictly-proper-scoring-rule papers.
- CD multistep sampler not implemented (rollout_fewstep only covers flow-map/MeanFlow).
- C3 (E06) uncollected.
