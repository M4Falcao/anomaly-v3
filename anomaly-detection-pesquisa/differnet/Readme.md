# DifferNet

This is the official repository to the WACV 2021 paper "[Same Same But DifferNet: Semi-Supervised Defect Detection with Normalizing Flows](
https://arxiv.org/abs/2008.12577)" by Marco Rudolph, Bastian Wandt and Bodo Rosenhahn.

If the only reason you ended up here is because you made a typo on 'different' - what was our intention - here is a shortened summary: We introduce a method that is able to find anomalies like defects on image data without having some of them in the training set. Click [here](https://www.youtube.com/watch?v=lFxDtC34tk0) to watch our short presentation from WACV.

## Getting Started

You will need [Python 3.6](https://www.python.org/downloads) and the packages specified in _requirements.txt_.
We recommend setting up a [virtual environment with pip](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)
and installing the packages there.

Install packages with:

```
$ pip install -r requirements.txt
```

## Configure and Run

All configurations concerning data, model, training, visualization etc. can be made in _config.py_. The default configuration will run a training with paper-given parameters on the provided dummy dataset. This dataset contains images of 4 squares as normal examples and 4 circles as anomaly.

To start the training, just run _main.py_! If training on the dummy data does not lead to an AUROC of 1.0, something seems to be wrong. Don't be worried if the loss is negative. The loss reflects the negative log likelihood which may be negative.
Please report us if you have issues when using the code.

> **Run every command from this directory** (`differnet/`). Scripts resolve their
> imports from the project root and write their outputs to the folders listed below.

## Project Structure

```
differnet/
├── config.py                   # Single source of truth for all hyperparameters
├── main.py                     # Entry point: image-level training
│
├── core/                       # Library code (imported, never run directly)
│   ├── freia_funcs.py          # Normalizing-flow primitives (vendored FrEIA)
│   ├── model.py                # DifferNet / SEDifferNet / CBAMDifferNet
│   ├── utils.py                # Datasets, transforms, loss, Score_Observer
│   ├── multi_transform_loader.py # ImageFolder yielding N transforms per image
│   ├── localization.py         # Gradient-based anomaly map export
│   ├── train.py                # Image-level training loop
│   ├── eval_common.py          # Shared evaluation metrics + architecture lookup
│   ├── mlflow_utils.py         # Checkpoint saving, MLflow UI, tracking export
│   ├── xai.py                  # GradCAM target for flow models
│   └── paths.py                # PROJECT_ROOT anchors
│
├── scripts/                    # Runnable entry points
│   ├── train/
│   │   └── pixel_train_from_pretrained.py  # CFLOW pixel head on frozen backbone
│   ├── eval/
│   │   ├── score_image_folder.py       # Score a folder of images
│   │   ├── evaluate_full_pipeline.py   # Final combined image + pixel evaluation
│   │   ├── evaluate_cflow_no_smoothing.py # CFLOW without Gaussian smoothing (ablation)
│   │   ├── evaluate_cam_xai_metrics.py # GradCAM / XAI metrics
│   │   ├── evaluate_cam_sanity_check.py # Weight-randomization sanity check
│   │   ├── evaluate_model_roc.py       # ROC and score distribution for one model
│   │   ├── test_model_grad_maps.py     # Single-model image + pixel metrics
│   │   ├── test_single_checkpoint.py   # One checkpoint, full metric report
│   │   ├── sweep_checkpoints.py        # Sweep a folder of checkpoints
│   │   └── visualize_cam_metric_steps.py # Step-by-step metric visualization
│   ├── analysis/
│   │   ├── analyze_class.py            # Per-class deep-dive analysis
│   │   └── smoke_test_baseline.py      # Trivial all-normal baseline
│   └── tools/
│       ├── export_mlflow.py            # Zip the mlruns store
│       ├── batch_evaluate_cam_metrics.py # Run evaluate_cam_xai_metrics.py for all classes
│       ├── read_checkpoint_auroc.py    # Print AUROC history from a checkpoint
│       ├── generate_dummy_gt.py        # Placeholder masks for the dummy dataset
│       └── print_output_log.py         # Print a UTF-16/UTF-8 log file
│
├── pixel_pipeline/             # Self-contained pixel-level package (run with -m)
├── docs/                       # Method notes and derivations
├── legacy/                     # Frozen older code, kept for reproducibility
│
└── checkpoints/ models/ weights/ mlruns/ results*/ gradient_maps/ ...   # Outputs
```

### Common commands

| Task | Command |
| --- | --- |
| Image-level training | `python main.py` |
| Pixel-level head training | `python scripts/train/pixel_train_from_pretrained.py` |
| Final combined evaluation | `python scripts/eval/evaluate_full_pipeline.py` |
| CFLOW without smoothing | `python scripts/eval/evaluate_cflow_no_smoothing.py` |
| Smoothing ablation | `python scripts/eval/evaluate_cflow_no_smoothing.py --sigmas 0,2,4,6` |
| Sweep all checkpoints | `python scripts/eval/sweep_checkpoints.py --help` |
| Per-class analysis | `python scripts/analysis/analyze_class.py --help` |
| Pixel pipeline | `python -m pixel_pipeline.run --help` |
| Export MLflow runs | `python scripts/tools/export_mlflow.py` |

Note that `pixel_pipeline` uses relative imports and must be launched with
`python -m pixel_pipeline.<module>`, not by file path.

## Data

The given dummy dataset shows how the implementation expects the construction of a dataset. Coincidentally, the [MVTec AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad) is constructed in this way.

Set the variables _dataset_path_ and _class_name_ in _config.py_ to run experiments on a dataset of your choice. The expected structure of the data is as follows:

``` 
train data:

        dataset_path/class_name/train/good/any_filename.png
        dataset_path/class_name/train/good/another_filename.tif
        dataset_path/class_name/train/good/xyz.png
        [...]

test data:

    'normal data' = non-anomalies

        dataset_path/class_name/test/good/name_the_file_as_you_like_as_long_as_there_is_an_image_extension.webp
        dataset_path/class_name/test/good/did_you_know_the_image_extension_webp?.png
        dataset_path/class_name/test/good/did_you_know_that_filenames_may_contain_question_marks????.png
        dataset_path/class_name/test/good/dont_know_how_it_is_with_windows.png
        dataset_path/class_name/test/good/just_dont_use_windows_for_this.png
        [...]

    anomalies - assume there are anomaly classes 'crack' and 'curved'

        dataset_path/class_name/test/crack/dat_crack_damn.png
        dataset_path/class_name/test/crack/let_it_crack.png
        dataset_path/class_name/test/crack/writing_docs_is_fun.png
        [...]

        dataset_path/class_name/test/curved/wont_make_a_difference_if_you_put_all_anomalies_in_one_class.png
        dataset_path/class_name/test/curved/but_this_code_is_practicable_for_the_mvtec_dataset.png
        [...]
``` 

## Credits

Some code of the [FrEIA framework](https://github.com/VLL-HD/FrEIA) was used for the implementation of Normalizing Flows. Follow [their tutorial](https://github.com/VLL-HD/FrEIA) if you need more documentation about it.


## Citation
Please cite our paper in your publications if it helps your research. Even if it does not, you are welcome to cite us.

    @inproceedings { RudWan2021,
    author = {Marco Rudolph and Bastian Wandt and Bodo Rosenhahn},
    title = {Same Same But DifferNet: Semi-Supervised Defect Detection with Normalizing Flows},
    booktitle = {Winter Conference on Applications of Computer Vision (WACV)},
    year = {2021},
    month = jan
    }
    
Another paper link because you missed the first one:

* [Same Same But DifferNet: Semi-Supervised Defect Detection with Normalizing Flows](
https://arxiv.org/abs/2008.12577)


## License

This project is licensed under the MIT License.

 
