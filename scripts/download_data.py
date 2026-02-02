#!/usr/bin/env python3
"""Download HotpotQA dataset."""

import requests
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# HotpotQA dataset URLs
DATASET_URLS = {
    'dev': 'http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json',
    'train': 'http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json'
}

DATA_DIR = Path('data/hotpotqa')


def download_file(url: str, filepath: Path) -> None:
    """Download file from URL."""
    logger.info(f"Downloading {url}...")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    logger.info(f"Saved to {filepath}")


def verify_dataset(filepath: Path) -> bool:
    """Verify dataset is valid JSON."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            num_examples = len(data)
        elif isinstance(data, dict):
            num_examples = len(data)
        else:
            return False
        
        logger.info(f"Dataset verified: {num_examples} examples")
        return True
    
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return False


def main():
    """Main download function."""
    logger.info("Downloading HotpotQA dataset...")
    
    for split, url in DATASET_URLS.items():
        filename = url.split('/')[-1]
        filepath = DATA_DIR / filename
        
        if filepath.exists():
            logger.info(f"{split} split already exists: {filepath}")
            if verify_dataset(filepath):
                continue
            else:
                logger.warning(f"Existing file is corrupted, re-downloading...")
        
        try:
            download_file(url, filepath)
            
            if verify_dataset(filepath):
                logger.info(f"✓ {split} split downloaded and verified")
            else:
                logger.error(f"✗ {split} split verification failed")
        
        except Exception as e:
            logger.error(f"Failed to download {split} split: {e}")
    
    logger.info("Download complete!")
    logger.info(f"Dataset location: {DATA_DIR.absolute()}")


if __name__ == "__main__":
    main()
