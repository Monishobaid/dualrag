"""Logging utilities for RAG experiments."""

import logging
import json
from pathlib import Path
from typing import Any, Optional
from datetime import datetime


def setup_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Setup a logger with console and optional file handlers.
    
    Args:
        name: Logger name
        log_file: Optional file path for logging
        level: Logging level
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)
    
    return logger


class JSONLWriter:
    """Writer for JSONL trace files."""
    
    def __init__(self, filepath: str):
        """
        Initialize JSONL writer.
        
        Args:
            filepath: Path to JSONL file
        """
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.file_handle = None
    
    def __enter__(self):
        """Context manager entry."""
        self.file_handle = open(self.filepath, 'w')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.file_handle:
            self.file_handle.close()
    
    def write(self, obj: dict) -> None:
        """
        Write a single JSON object as a line.
        
        Args:
            obj: Dictionary to write
        """
        if self.file_handle is None:
            raise RuntimeError("JSONLWriter must be used as context manager")
        
        json_line = json.dumps(obj, ensure_ascii=False)
        self.file_handle.write(json_line + '\n')
        self.file_handle.flush()
    
    def write_many(self, objects: list) -> None:
        """
        Write multiple JSON objects.
        
        Args:
            objects: List of dictionaries to write
        """
        for obj in objects:
            self.write(obj)


def read_jsonl(filepath: str) -> list:
    """
    Read a JSONL file.
    
    Args:
        filepath: Path to JSONL file
    
    Returns:
        List of parsed JSON objects
    """
    objects = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                objects.append(json.loads(line))
    return objects


def log_experiment_start(logger: logging.Logger, config: Any) -> None:
    """Log experiment start with configuration."""
    logger.info("="*80)
    logger.info("EXPERIMENT START")
    logger.info("="*80)
    logger.info(f"Method: {config.method}")
    logger.info(f"Max examples: {config.max_examples}")
    logger.info(f"Max hops: {config.max_hops}")
    logger.info(f"Top-K: {config.top_k}")
    logger.info(f"Reasoning stop: {config.enable_reasoning_stop}")
    logger.info(f"Retrieval stop: {config.enable_retrieval_stop}")
    logger.info(f"Seed: {config.seed}")
    logger.info(f"Run ID: {config.run_id}")
    logger.info("="*80)


def log_experiment_end(logger: logging.Logger, stats: dict) -> None:
    """Log experiment end with summary statistics."""
    logger.info("="*80)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("="*80)
    logger.info(f"Total examples: {stats.get('total_examples', 0)}")
    logger.info(f"Avg hops: {stats.get('avg_hops', 0):.2f}")
    logger.info(f"Avg tokens: {stats.get('avg_tokens_generated', 0):.1f}")
    logger.info(f"EM: {stats.get('em', 0):.3f}")
    logger.info(f"F1: {stats.get('f1', 0):.3f}")
    logger.info("="*80)
