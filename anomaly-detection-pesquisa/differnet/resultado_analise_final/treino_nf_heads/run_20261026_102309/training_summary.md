# Resumo do Treinamento de NF Heads (Run: 20261026_102309)

Validation selects the epoch; holdout reports the final inference recipe. Missing metrics were not computed. Macro averages exclude failed/skipped classes.

| class                           | status      | recipe  | best_epoch | train_minutes | val_pixel_auroc | holdout_pixel_auroc | holdout_image_auroc | holdout_aupro | holdout_pixel_ap | holdout_pixel_f1 |
| ------------------------------- | ----------- | ------- | ---------- | ------------- | --------------- | ------------------- | ------------------- | ------------- | ---------------- | ---------------- |
| glass-insulator                 | SUCCESS     | general | 10.0       | 117.22        | 0.7001          | 0.6688              | 0.6955              | 0.4963        | 0.0011           | 0.0081           |
| lightning-rod-suspension        | SUCCESS     | general | 30.0       | 42.94         | 0.9049          | 0.9234              | 0.7414              | 0.7082        | 0.0381           | 0.0829           |
| polymer-insulator-upper-shackle | SUCCESS     | general | 27.0       | 66.72         | 0.7525          | 0.6249              | 0.6346              | 0.3976        | 0.0101           | 0.0302           |
| vari-grip                       | SUCCESS     | general | 5.0        | 69.53         | 0.8822          | 0.893               | 0.6452              | 0.6393        | 0.0635           | 0.1194           |
| yoke-suspension                 | SUCCESS     | general | 2.0        | 133.05        | 0.9291          | 0.9104              | 0.7459              | 0.7827        | 0.018            | 0.0559           |
| MACRO_AVERAGE                   | 5/5 classes | general |            |               | 0.8338          | 0.8041              | 0.6925              | 0.6048        | 0.0262           | 0.0593           |
