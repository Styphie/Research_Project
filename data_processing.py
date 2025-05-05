# data_processing.py

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from collections import Counter
import numpy as np

# Define Amino Acids for PSC
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

def generate_ecfp4(smiles_string, n_bits=1024):
    """Generates ECFP4 fingerprint for a given SMILES string."""
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        return np.zeros(n_bits, dtype=int) # Return zero vector for invalid SMILES
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits) # Radius 2 for ECFP4
    return np.array(fp)

def calculate_aac(sequence):
    """Calculates Amino Acid Composition (AAC) for a protein sequence."""
    count = Counter(sequence)
    total = len(sequence)
    aac_vector = [count.get(aa, 0) / total for aa in AMINO_ACIDS]
    return np.array(aac_vector)

def process_data(file_path, ecfp_bits=1024):
    """Loads data, generates ECFP4 and AAC features, and combines them."""
    print(f"Processing {file_path}...")
    df = pd.read_csv(file_path)

    # Drop rows with missing SMILES or Target Sequence
    df.dropna(subset=['SMILES', 'Target Sequence'], inplace=True)

    # Generate ECFP4 features
    print("Generating ECFP4 features...")
    ecfp4_features = df['SMILES'].apply(lambda x: generate_ecfp4(x, n_bits=ecfp_bits))
    ecfp4_df = pd.DataFrame(ecfp4_features.tolist(), index=df.index)
    ecfp4_df.columns = [f'ECFP4_{i}' for i in range(ecfp_bits)]

    # Generate AAC features (as a basic PSC)
    print("Generating AAC features...")
    aac_features = df['Target Sequence'].apply(calculate_aac)
    aac_df = pd.DataFrame(aac_features.tolist(), index=df.index)
    aac_df.columns = [f'AAC_{aa}' for aa in AMINO_ACIDS]

    # Combine features and labels
    print("Combining features...")
    features_df = pd.concat([ecfp4_df, aac_df], axis=1)
    labels = df['Label']

    print(f"Finished processing {file_path}. Features shape: {features_df.shape}, Labels shape: {labels.shape}")
    return features_df, labels

if __name__ == "__main__":
    data_dir = "MolTrans/dataset/BIOSNAP/full_data"
    output_dir = "Research Project"
    import os
    os.makedirs(output_dir, exist_ok=True)

    ecfp_bits = 1024 # Standard size for ECFP4

    # Process train, validation, and test sets
    X_train, y_train = process_data(f"{data_dir}/train.csv", ecfp_bits=ecfp_bits)
    X_val, y_val = process_data(f"{data_dir}/val.csv", ecfp_bits=ecfp_bits)
    X_test, y_test = process_data(f"{data_dir}/test.csv", ecfp_bits=ecfp_bits)

    # Save processed data
    print("Saving processed data...")
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False, header=True)
    X_val.to_csv(f"{output_dir}/X_val.csv", index=False)
    y_val.to_csv(f"{output_dir}/y_val.csv", index=False, header=True)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False, header=True)

    print("Data processing complete. Processed files saved in", output_dir)

