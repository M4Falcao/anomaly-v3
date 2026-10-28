# Resumo do Treinamento de NF Heads (Run: 20261025_181523)

Validation selects the epoch; holdout reports the final inference recipe. Missing metrics were not computed. Macro averages exclude failed/skipped classes.

| class                           | status      | recipe  | best_epoch | train_minutes | val_pixel_auroc | holdout_pixel_auroc | holdout_image_auroc | holdout_aupro | holdout_pixel_ap | holdout_pixel_f1 |
| ------------------------------- | ----------- | ------- | ---------- | ------------- | --------------- | ------------------- | ------------------- | ------------- | ---------------- | ---------------- |
| glass-insulator                 | SUCCESS     | general | 15.0       | 136.37        | 0.8807          | 0.9141              | 0.7314              | 0.5496        | 0.0024           | 0.0175           |
| lightning-rod-suspension        | SUCCESS     | general | 15.0       | 68.94         | 0.8492          | 0.8533              | 0.6913              | 0.5052        | 0.0167           | 0.0357           |
| polymer-insulator-upper-shackle | SUCCESS     | general | 30.0       | 117.53        | 0.8837          | 0.8585              | 0.5405              | 0.5484        | 0.0239           | 0.0454           |
| vari-grip                       | SUCCESS     | general | 10.0       | 97.85         | 0.8712          | 0.915               | 0.7286              | 0.6145        | 0.0828           | 0.1646           |
| yoke-suspension                 | SUCCESS     | general | 2.0        | 190.48        | 0.9054          | 0.9349              | 0.7621              | 0.6922        | 0.0087           | 0.0275           |
| MACRO_AVERAGE                   | 5/5 classes | general |            |               | 0.878           | 0.8952              | 0.6908              | 0.582         | 0.0269           | 0.0582           |
