# RFscore-refined
An implementation of [RFscore](https://github.com/oddt/rfscorevs), allowing facile training and testing of large ensemble models.

## Setup

```bash
conda env create -f environment.yaml
conda activate rfscore
```

## Usage

### Single Model
```bash
# Train
python script.py --train --csv_file train.csv --data_dir data/structures --model_name rf_model

# Predict  
python script.py --predict --val_csv_file test.csv --val_data_dir data/structures --model_name rf_model

# Train and predict
python script.py --train --predict --csv_file train.csv --val_csv_file test.csv --data_dir data/structures --val_data_dir data/structures --model_name rf_model
```

### Ensemble Models
```bash
# Train 5 models (saves to model1/, model2/, ...)
python script.py --train --ensemble 5 --csv_file train.csv --data_dir data/structures --model_name rf_ensemble

# Predict with all 5 models
python script.py --predict --ensemble 5 --val_csv_file test.csv --val_data_dir data/structures --model_name rf_ensemble

# Train and predict in one go
python script.py --train --predict --ensemble 5 --csv_file train.csv --val_csv_file test.csv --data_dir data/structures --val_data_dir data/structures --model_name rf_ensemble
```

### Metric Calculation
Metric calculation on prediction results can be easily performed using the `metrics.py` or `metric-ensembles.py` scripts from the [utilities repo](https://github.com/savvag44/binding-model-utilities).
Note that the directory structure is appropriate for use with `metric-ensembles.py` by default, just pass the base model directory containing model{1..5} as the argument :)

## Input Format

The ToolboxSF .csv file format is used by default:
CSV files require columns: `key`, `protein`, `ligand`, `pk`
- `protein`: PDB filename 
- `ligand`: SDF filename
- `pk`: binding affinity value

An AEV-PLIG .csv file can easily be converted to the above using the `reformatter.py` or `reformat_train_valid_split.py` scripts from the [utilities repo](https://github.com/savvag44/binding-model-utilities).

## Output

- **Models**: Saved to `data/models/` (single) or `modelN/data/models/` (ensemble)
- **Predictions**: CSV with columns `key`, `pred`, `pk` 
- **Features**: Cached in `data/features/`
- **Failure reports**: Generated for processing errors

## Options

- `--cutoff`: Distance cutoff for features (default: 12Å)
- `--cache`: Enable caching (currently unused)

## Notes

- Features are automatically cached and reused
- Failed complexes are logged with detailed error reports
- NaN values in features are replaced with 0.0
- Each ensemble model uses different random seed for diversity
