"""
ORTHON Feature Export
=====================

Export ML features in various formats, including obfuscated blobs
for competitive ML scenarios.

Formats:
- parquet: Full feature matrix with column names (default)
- npz: Obfuscated NumPy blob (no column names)
- encrypted: AES-encrypted parquet (requires key)
"""

import hashlib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
import secrets


def obfuscate_features(
    df: pd.DataFrame,
    entity_col: str = 'entity_id',
    output_path: Optional[Path] = None,
    include_manifest: bool = False,
    seed: Optional[int] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """
    Export features as an obfuscated NumPy blob.

    The output contains:
    - X: Feature matrix (float32) with shuffled column order
    - entity_ids: Entity identifiers
    - n_features: Number of features
    - checksum: SHA256 of the feature matrix
    - version: ORTHON version

    Column names are NOT included. Feature order is randomized.

    Args:
        df: DataFrame with features
        entity_col: Name of entity ID column
        output_path: Output .npz path (default: features_obfuscated.npz)
        include_manifest: If True, also save a separate manifest (for ORTHON internal use)
        seed: Random seed for reproducible shuffling

    Returns:
        Tuple of (output_path, metadata_dict)
    """
    if seed is not None:
        np.random.seed(seed)

    # Separate entity IDs from features
    entity_ids = df[entity_col].values
    feature_cols = [c for c in df.columns if c != entity_col]

    # Shuffle column order
    shuffled_cols = feature_cols.copy()
    np.random.shuffle(shuffled_cols)

    # Extract feature matrix
    X = df[shuffled_cols].values.astype(np.float32)

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute checksum
    checksum = hashlib.sha256(X.tobytes()).hexdigest()

    # Metadata (NOT saved in blob)
    metadata = {
        'n_entities': len(entity_ids),
        'n_features': len(feature_cols),
        'checksum': checksum,
        'created_at': datetime.now().isoformat(),
        'version': '1.0.0',
        'format': 'orthon_obfuscated_v1',
    }

    # Secret manifest (maps shuffled index to original column name)
    # This is for ORTHON internal use only - never distributed
    manifest = {
        'column_map': {i: col for i, col in enumerate(shuffled_cols)},
        'original_order': feature_cols,
        'checksum': checksum,
    }

    # Output path
    if output_path is None:
        output_path = Path('features_obfuscated.npz')
    else:
        output_path = Path(output_path)

    # Save obfuscated blob
    np.savez_compressed(
        output_path,
        X=X,
        entity_ids=entity_ids,
        n_features=np.array([len(feature_cols)]),
        checksum=np.array([checksum]),
        version=np.array(['orthon_obfuscated_v1']),
    )

    # Optionally save manifest (separate file, not distributed)
    if include_manifest:
        manifest_path = output_path.with_suffix('.manifest.json')
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

    return output_path, metadata


def load_obfuscated_features(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load obfuscated feature blob.

    Returns:
        Tuple of (X feature matrix, entity_ids)
    """
    data = np.load(path, allow_pickle=True)
    X = data['X']
    entity_ids = data['entity_ids']
    return X, entity_ids


def export_encrypted_parquet(
    df: pd.DataFrame,
    output_path: Path,
    key: Optional[bytes] = None,
) -> Tuple[Path, bytes]:
    """
    Export features as an AES-encrypted parquet.

    Args:
        df: DataFrame with features
        output_path: Output path (.encrypted)
        key: Encryption key (generated if not provided)

    Returns:
        Tuple of (output_path, encryption_key)
    """
    try:
        from cryptography.fernet import Fernet
    except ImportError:
        raise ImportError("cryptography required: pip install cryptography")

    # Generate key if not provided
    if key is None:
        key = Fernet.generate_key()

    fernet = Fernet(key)

    # Serialize parquet to bytes
    parquet_bytes = df.to_parquet(index=False)

    # Encrypt
    encrypted = fernet.encrypt(parquet_bytes)

    # Save
    output_path = Path(output_path)
    output_path.write_bytes(encrypted)

    return output_path, key


def decrypt_parquet(path: Path, key: bytes) -> pd.DataFrame:
    """
    Decrypt an encrypted parquet file.

    Args:
        path: Path to encrypted file
        key: Decryption key

    Returns:
        Decrypted DataFrame
    """
    from cryptography.fernet import Fernet
    import io

    fernet = Fernet(key)
    encrypted = Path(path).read_bytes()
    decrypted = fernet.decrypt(encrypted)

    return pd.read_parquet(io.BytesIO(decrypted))


def create_competition_bundle(
    df: pd.DataFrame,
    output_dir: Path,
    entity_col: str = 'entity_id',
    name: str = 'orthon_features',
) -> Dict[str, Path]:
    """
    Create a competition-ready feature bundle.

    Creates:
    - {name}.npz: Obfuscated feature blob (distribute to competitors)
    - {name}_readme.txt: Usage instructions
    - {name}.manifest.json: Column mapping (KEEP SECRET)

    Args:
        df: Feature DataFrame
        output_dir: Output directory
        entity_col: Entity ID column name
        name: Bundle name prefix

    Returns:
        Dict of created file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create obfuscated blob
    blob_path, metadata = obfuscate_features(
        df,
        entity_col=entity_col,
        output_path=output_dir / f'{name}.npz',
        include_manifest=True,
    )

    # Create readme
    readme = f"""ORTHON Pre-computed Features
============================

File: {name}.npz
Entities: {metadata['n_entities']}
Features: {metadata['n_features']}
Checksum: {metadata['checksum'][:16]}...

Usage (Python):
---------------
import numpy as np

data = np.load('{name}.npz', allow_pickle=True)
X = data['X']           # Feature matrix (n_entities x n_features)
ids = data['entity_ids'] # Entity identifiers

# Use with sklearn
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X, y_train)

Notes:
------
- Features are pre-computed by ORTHON signal analysis
- Column names are intentionally omitted
- Feature order is randomized
- Do not attempt to reverse-engineer feature meanings

Generated: {metadata['created_at']}
ORTHON Version: {metadata['version']}
"""

    readme_path = output_dir / f'{name}_readme.txt'
    readme_path.write_text(readme)

    return {
        'blob': blob_path,
        'readme': readme_path,
        'manifest': output_dir / f'{name}.manifest.json',
    }


# CLI
if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("ORTHON Feature Export")
        print("=" * 40)
        print("\nUsage:")
        print("  python -m orthon.ml.feature_export <input.parquet> [output_dir] [--obfuscate]")
        print("\nExamples:")
        print("  python -m orthon.ml.feature_export ml_entity_features.parquet ./competition --obfuscate")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else input_path.parent
    obfuscate = '--obfuscate' in sys.argv

    df = pd.read_parquet(input_path)
    print(f"Loaded {input_path}: {df.shape[0]} rows x {df.shape[1]} columns")

    if obfuscate:
        paths = create_competition_bundle(df, output_dir, name=input_path.stem)
        print(f"\nCreated competition bundle:")
        print(f"  Blob:     {paths['blob']}")
        print(f"  Readme:   {paths['readme']}")
        print(f"  Manifest: {paths['manifest']} (KEEP SECRET)")
    else:
        output_path = output_dir / f"{input_path.stem}_export.parquet"
        df.to_parquet(output_path, index=False)
        print(f"Exported to {output_path}")
