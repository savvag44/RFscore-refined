# # Train 5 ensemble models
# python script.py --train --ensemble 5 --csv_file ... --data_dir ... --model_name rf_1x_hi_conf

# # Predict with all 5 models
# python script.py --predict --ensemble 5 --val_csv_file ... --val_data_dir ... --model_name rf_1x_hi_conf

# # Train and predict in one go
# python script.py --train --predict --ensemble 5 --csv_file ... --val_csv_file ...

from oddt.toolkits import ob
from oddt.scoring import descriptors
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import os
from argparse import ArgumentParser
from tqdm import tqdm
from joblib import Parallel, delayed
import pickle
import gc


def load_csv(csv_file, data_dir):
    """
    Loads CSV file specfiying data and adds full path to protein and ligand files

    Parameters
    ----------
    csv_file : str
        Path to CSV file
    data_dir : str
        Path to directory with protein and ligand files

    Returns
    -------
    keys : list
        List of keys
    protein_files : list
        List of paths to protein files
    ligand_files : list
        List of paths to ligand files
    pks : list
        List of pK values
    """
    df = pd.read_csv(csv_file)
    keys = df["key"].values
    protein_files = [os.path.join(data_dir, file) for file in df["protein"].values]
    ligand_files = [os.path.join(data_dir, file) for file in df["ligand"].values]
    pks = df["pk"].values
    return keys, protein_files, ligand_files, pks


def generate_feature(protein_file, ligand_file, cutoff, index=None):
    """
    Generates RFScore features for a single protein-ligand complex

    Parameters
    ----------
    protein_file : str
        Path to protein PDB file
    ligand_file : str
        Path to ligand SDF file
    cutoff : float
        Distance cutoff for features
    index : int, optional
        Index of the complex for error reporting

    Returns
    -------
    dict
        Dictionary with feature names and values, includes '_failed' key if processing failed
    """
    try:
        protein = next(ob.readfile("pdb", protein_file))
        protein.protein = True
        ligand = next(ob.readfile("sdf", ligand_file))
        rfscore_engine = descriptors.close_contacts_descriptor(
            protein=protein,
            cutoff=cutoff,
            ligand_types=[6, 7, 8, 9, 15, 16, 17, 35, 53],
            protein_types=[6, 7, 8, 16],
        )
        features = rfscore_engine.build(ligand)[0]
        result = {
            name: value
            for name, value in zip(rfscore_engine.titles, features)
        }
        # Check for NaN or infinite values
        nan_found = False
        for key, value in result.items():
            if pd.isna(value) or not pd.api.types.is_numeric_dtype(type(value)):
                result[key] = 0.0  # Replace with default value
                nan_found = True
        
        if nan_found:
            result['_nan_warning'] = True
            
        result['_failed'] = False
        return result
    except Exception as e:
        error_msg = f"Error processing complex {index if index is not None else 'unknown'}: {protein_file}, {ligand_file}: {e}"
        print(error_msg)
        # Return dict indicating failure
        return {'_failed': True, '_error': error_msg, '_protein': protein_file, '_ligand': ligand_file}


def batch_generate_features(csv_file, data_dir, cutoff):
    """
    Generates RFScore features for a list of protein-ligand complexes

    Parameters
    ----------
    csv_file : str
        Path to CSV file
    data_dir : str
        Path to directory with protein and ligand files
    cutoff : float
        Distance cutoff for features

    Returns
    -------
    features_df : pandas.DataFrame
        DataFrame with features for all data in CSV file
    pks : list
        List of pK values
    keys : list
        List of keys for each protein-ligand complex
    """
    keys, protein_files, ligand_files, pks = load_csv(csv_file, data_dir)
    if not os.path.exists("data/features"):
        os.makedirs("data/features")
    if not os.path.exists(
        f'data/features/{csv_file.split("/")[-1].split(".")[0]}_{cutoff}_features.csv'
    ):
        with Parallel(n_jobs=-1) as parallel:
            features = parallel(
                delayed(generate_feature)(protein_files[i], ligand_files[i], cutoff, i)
                for i in tqdm(range(len(keys)))
            )
        
        # Analyze failed complexes
        failed_complexes = []
        nan_complexes = []
        successful_features = []
        
        for i, feature_dict in enumerate(features):
            if feature_dict.get('_failed', False):
                failed_complexes.append({
                    'index': i,
                    'key': keys[i],
                    'protein': feature_dict.get('_protein', protein_files[i]),
                    'ligand': feature_dict.get('_ligand', ligand_files[i]),
                    'error': feature_dict.get('_error', 'Unknown error')
                })
            else:
                if feature_dict.get('_nan_warning', False):
                    nan_complexes.append({
                        'index': i,
                        'key': keys[i],
                        'protein': protein_files[i],
                        'ligand': ligand_files[i]
                    })
                # Remove metadata before adding to successful features
                clean_dict = {k: v for k, v in feature_dict.items() if not k.startswith('_')}
                successful_features.append(clean_dict)
        
        # Report results
        print(f"\n=== FEATURE GENERATION SUMMARY ===")
        print(f"Total complexes processed: {len(features)}")
        print(f"Successful: {len(successful_features)}")
        print(f"Failed: {len(failed_complexes)}")
        print(f"Had NaN values (replaced with 0.0): {len(nan_complexes)}")
        
        if failed_complexes:
            print(f"\n=== FAILED COMPLEXES ===")
            for failed in failed_complexes:
                print(f"Index {failed['index']}: Key '{failed['key']}'")
                print(f"  Protein: {failed['protein']}")
                print(f"  Ligand: {failed['ligand']}")
                print(f"  Error: {failed['error']}")
                print()
        
        if nan_complexes:
            print(f"\n=== COMPLEXES WITH NaN VALUES (FIXED) ===")
            for nan_complex in nan_complexes:
                print(f"Index {nan_complex['index']}: Key '{nan_complex['key']}'")
                print(f"  Protein: {nan_complex['protein']}")
                print(f"  Ligand: {nan_complex['ligand']}")
        
        print("Converting to DataFrame...")
        features_df = pd.DataFrame(successful_features)
        print(f"DataFrame created with shape: {features_df.shape}")
        print("Saving features to CSV...")
        features_df.to_csv(
            f'data/features/{csv_file.split("/")[-1].split(".")[0]}_{cutoff}_features.csv',
            index=False,
        )
        print("Features saved successfully")
        
        # Also save the failed complexes report
        if failed_complexes or nan_complexes:
            report_file = f'data/features/{csv_file.split("/")[-1].split(".")[0]}_{cutoff}_failures.txt'
            with open(report_file, 'w') as f:
                f.write("=== FEATURE GENERATION FAILURE REPORT ===\n\n")
                f.write(f"Total complexes processed: {len(features)}\n")
                f.write(f"Successful: {len(successful_features)}\n")
                f.write(f"Failed: {len(failed_complexes)}\n")
                f.write(f"Had NaN values (replaced with 0.0): {len(nan_complexes)}\n\n")
                
                if failed_complexes:
                    f.write("=== FAILED COMPLEXES ===\n")
                    for failed in failed_complexes:
                        f.write(f"Index {failed['index']}: Key '{failed['key']}\n")
                        f.write(f"  Protein: {failed['protein']}\n")
                        f.write(f"  Ligand: {failed['ligand']}\n")
                        f.write(f"  Error: {failed['error']}\n\n")
                
                if nan_complexes:
                    f.write("=== COMPLEXES WITH NaN VALUES (FIXED) ===\n")
                    for nan_complex in nan_complexes:
                        f.write(f"Index {nan_complex['index']}: Key '{nan_complex['key']}\n")
                        f.write(f"  Protein: {nan_complex['protein']}\n")
                        f.write(f"  Ligand: {nan_complex['ligand']}\n\n")
            print(f"Failure report saved to: {report_file}")
        
        # Clean up memory
        del features
        gc.collect()
    else:
        print("Loading cached features...")
        features_df = pd.read_csv(
            f'data/features/{csv_file.split("/")[-1].split(".")[0]}_{cutoff}_features.csv'
        )
        print(f"Loaded features with shape: {features_df.shape}")
        
        # Check if failure report exists and show summary
        report_file = f'data/features/{csv_file.split("/")[-1].split(".")[0]}_{cutoff}_failures.txt'
        if os.path.exists(report_file):
            print(f"Previous failure report found: {report_file}")
            with open(report_file, 'r') as f:
                lines = f.readlines()
                for line in lines[:10]:  # Show first 10 lines of report
                    print(line.strip())
    return features_df, pks, keys


def train_model(csv_file, data_dir, cutoff, random_state=None):
    """
    Trains a Random Forest model on RFScore features of protein-ligand complexes

    Parameters
    ----------
    csv_file : str
        Path to CSV file
    data_dir : str
        Path to directory with protein and ligand files
    cutoff : float
        Distance cutoff for features
    random_state : int, optional
        Random state for reproducibility

    Returns
    -------
    sklearn.ensemble.RandomForestRegressor
        Trained Random Forest model
    """
    try:
        features_df, pks, keys = batch_generate_features(csv_file, data_dir, cutoff)
        print("Ready to train model")
        print(f"Features shape: {features_df.shape}")
        print(f"Number of pK values: {len(pks)}")
        print(f"Features data types: {features_df.dtypes.value_counts()}")
        
        # Check for any NaN or infinite values
        if features_df.isnull().any().any():
            print("Warning: NaN values found in features")
            print(f"NaN count per column: {features_df.isnull().sum().sum()}")
        
        if not all(pd.api.types.is_numeric_dtype(features_df[col]) for col in features_df.columns):
            print("Warning: Non-numeric data types found in features")
        
        print(f"Creating Random Forest model (random_state={random_state})...")
        model = RandomForestRegressor(
            n_estimators=500,
            max_features="sqrt",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=random_state,
        )
        
        print("Starting model training...")
        model.fit(features_df, pks)
        print("Model training completed successfully!")
        return model
        
    except Exception as e:
        print(f"Error in train_model: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        raise


def predict(model, csv_file, data_dir, cutoff, output_dir="data/results"):
    """
    Predicts pK values for a list of protein-ligand complexes using a trained Random Forest model

    Parameters
    ----------
    model : sklearn.ensemble.RandomForestRegressor
        Trained Random Forest model
    csv_file : str
        Path to CSV file
    data_dir : str
        Path to directory with protein and ligand files
    cutoff : float
        Distance cutoff for features
    output_dir : str
        Directory to save results

    Returns
    -------
    pandas.DataFrame
        DataFrame with predicted pK values
    """
    features_df, pks, keys = batch_generate_features(csv_file, data_dir, cutoff)
    pred_pK = model.predict(features_df)
    return pd.DataFrame({"key": keys, "pred": pred_pK, "pk": pks})


def train_ensemble(args):
    """
    Train an ensemble of models, each in its own directory
    
    Parameters
    ----------
    args : ArgumentParser
        Parsed command line arguments
    """
    n_models = args.ensemble
    
    for i in range(1, n_models + 1):
        print(f"\n{'='*50}")
        print(f"Training model {i}/{n_models}")
        print(f"{'='*50}\n")
        
        # Create model directory
        model_dir = f"model{i}"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Create subdirectories
        model_data_dir = os.path.join(model_dir, "data")
        model_models_dir = os.path.join(model_data_dir, "models")
        model_results_dir = os.path.join(model_data_dir, "results")
        
        for dir_path in [model_data_dir, model_models_dir, model_results_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        
        # Train model with different random state
        model = train_model(args.csv_file, args.data_dir, args.cutoff, random_state=i)
        
        # Save model in model-specific directory
        model_path = os.path.join(model_models_dir, f"{args.model_name}.pkl")
        with open(model_path, "wb") as handle:
            pickle.dump(model, handle)
        print(f"Model {i} saved to {model_path}")
        
        # If predict flag is also set, run predictions
        if args.predict:
            results_df = predict(model, args.val_csv_file, args.val_data_dir, args.cutoff)
            results_path = os.path.join(
                model_results_dir,
                f'{args.model_name}_{args.val_csv_file.split("/")[-1]}'
            )
            results_df.to_csv(results_path, index=False)
            print(f"Predictions for model {i} saved to {results_path}")


def predict_ensemble(args):
    """
    Run predictions using an ensemble of models
    
    Parameters
    ----------
    args : ArgumentParser
        Parsed command line arguments
    """
    n_models = args.ensemble
    
    for i in range(1, n_models + 1):
        print(f"\n{'='*50}")
        print(f"Predicting with model {i}/{n_models}")
        print(f"{'='*50}\n")
        
        model_dir = f"model{i}"
        model_path = os.path.join(model_dir, "data", "models", f"{args.model_name}.pkl")
        
        if not os.path.exists(model_path):
            print(f"Warning: Model file not found at {model_path}, skipping...")
            continue
        
        with open(model_path, "rb") as handle:
            model = pickle.load(handle)
        
        results_df = predict(model, args.val_csv_file, args.val_data_dir, args.cutoff)
        
        results_dir = os.path.join(model_dir, "data", "results")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        results_path = os.path.join(
            results_dir,
            f'{args.model_name}_{args.val_csv_file.split("/")[-1]}'
        )
        results_df.to_csv(results_path, index=False)
        print(f"Predictions for model {i} saved to {results_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--csv_file", help="Training CSV file with protein, ligand and pk data"
    )
    parser.add_argument(
        "--data_dir", help="Training directory with protein and ligand files"
    )
    parser.add_argument(
        "--val_csv_file", help="Test CSV file with protein, ligand and pk data"
    )
    parser.add_argument(
        "--val_data_dir", help="Test directory with protein and ligand files"
    )
    parser.add_argument(
        "--model_name", help="Name of the model to be saved or to be loaded"
    )
    parser.add_argument(
        "--cutoff", help="Distance cutoff for features", default=12, type=int
    )
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--predict", action="store_true", help="Predict pK")
    parser.add_argument(
        "--ensemble", help="Train/predict ensemble of N models", type=int, default=0
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Cache temporary files in data/scratch directory",
    )
    args = parser.parse_args()
    
    # Ensemble mode
    if args.ensemble > 0:
        if args.train:
            train_ensemble(args)
        elif args.predict:
            predict_ensemble(args)
        else:
            print("Please specify --train and/or --predict with --ensemble")
    
    # Single model mode (original behavior)
    else:
        if args.train:
            if not os.path.exists("data/models"):
                os.makedirs("data/models")
            model = train_model(args.csv_file, args.data_dir, args.cutoff)
            with open(f"data/models/{args.model_name}.pkl", "wb") as handle:
                pickle.dump(model, handle)
        if args.predict:
            if not os.path.exists("data/models"):
                os.makedirs("data/models")
            with open(f"data/models/{args.model_name}.pkl", "rb") as handle:
                model = pickle.load(handle)
            results_df = predict(model, args.val_csv_file, args.val_data_dir, args.cutoff)
            if not os.path.exists("data/results"):
                os.makedirs("data/results")
            results_df.to_csv(
                f'data/results/{args.model_name}_{args.val_csv_file.split("/")[-1]}',
                index=False,
            )
        if not args.train and not args.predict:
            print("Please specify --train and/or --predict")
