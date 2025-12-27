import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional

def get_sha256(file_path: Path) -> str:
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def is_cache_valid(
    meta_path: Path,
    expected_config: Dict[str, Any],
    required_files: List[Path],
    verify: bool = False,
) -> bool:
    """
    Checks if the cache is valid based on meta.json and existence of files.
    """
    if not meta_path.exists():
        return False
        
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return False
        
    # 1. Check config parameters
    # The caller passes in a flattened dict of parameters to check
    for key, value in expected_config.items():
        # Check top-level or subsample level
        if key in meta:
            if meta[key] != value:
                return False
        elif "subsample" in meta and key in meta["subsample"]:
            if meta["subsample"][key] != value:
                return False
        else:
            return False

    # 2. Check file existence and sizes/hashes
    for file_path in required_files:
        if not file_path.exists():
            return False
            
        file_name = file_path.name
        file_meta = None
        
        # Look for file in meta["splits"]
        if "splits" in meta:
            # The key in splits might be the split name, and file might be a field
            for split_name, s_meta in meta["splits"].items():
                if s_meta.get("file") == file_name:
                    file_meta = s_meta
                    break
        
        if not file_meta:
            # Fallback for tokenizer where it might be top level or differently named
            if file_name == "tokenizer.json":
                if "tokenizer_sha256" in meta:
                    file_meta = {"sha256": meta["tokenizer_sha256"]}
            # Generic files check?
        
        if file_meta:
            # Check size if available
            if "size_bytes" in file_meta:
                if file_path.stat().st_size != file_meta["size_bytes"]:
                    return False
            
            # Optional SHA256 verification
            if verify and "sha256" in file_meta:
                if get_sha256(file_path) != file_meta["sha256"]:
                    return False

    return True
