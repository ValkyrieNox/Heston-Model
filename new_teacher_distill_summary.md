# New Teacher Distillation Summary

| Model | Kurt diff | Marg W1 | Total W1 | Sig-W1 | Pricing RMSE | Pricing MAPE | JSON |
|---|---:|---:|---:|---:|---:|---:|---|
| FM new teacher last | 0.2594 | 0.0030 | 0.0232 | 0.0061 | 1.7640 | 0.1857 | runs/experiments/p2_teacher_select_20260522T100554Z/evaluation/eval_fm_tf_p0_start6_last.json |
| old FM teacher | 1.6286 | 0.0017 | 0.2718 | 0.0795 | 4.7400 | 0.3526 | runs/experiments/p2_medium_complete/evaluation/eval_fm.json |
| old MF best CFG | 0.1157 | 0.0013 | 0.0792 | 0.0165 | 2.9719 | 0.2778 | runs/experiments/p2_medium_complete/evaluation/eval_mf.json |
| old CD | 0.7071 | 0.0009 | 0.0632 | 0.0122 | 3.6069 | 0.2832 | runs/experiments/p2_medium_complete/evaluation/eval_cd.json |
| QGAN last calibrated | 0.3802 | 0.0011 | 0.0321 | 0.0083 | 1.5141 | 0.1549 | runs/experiments/p2_medium_complete/evaluation/eval_quant_gan_last_calibrated.json |
