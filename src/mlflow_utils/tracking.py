"""
MLflow tracking module.

This module provides a wrapper around MLflow for experiment tracking
and model management.
"""

import os
import mlflow
import mlflow.pytorch
import numpy as np
from typing import Dict, Any, Optional
import torch.nn as nn


class MLflowTracker:
    """
    Wrapper class for MLflow tracking.
    
    This class provides convenient methods for logging parameters,
    metrics, and models to MLflow.
    """
    
    def __init__(
        self,
        experiment_name: str = "lstm_stock_prediction",
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None
    ):
        """
        Initialize the MLflow tracker.
        
        Args:
            experiment_name: Name of the experiment
            tracking_uri: MLflow tracking URI
            artifact_location: Artifact location for the experiment
        """
        self.experiment_name = experiment_name
        
        # Set tracking URI if provided
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        # Set experiment
        mlflow.set_experiment(experiment_name)
        
        # Set artifact location if provided
        if artifact_location:
            os.makedirs(artifact_location, exist_ok=True)
    
    def start_run(self, run_name: Optional[str] = None):
        """
        Start an MLflow run.
        
        Args:
            run_name: Optional name for the run
        
        Returns:
            Active MLflow run
        """
        return mlflow.start_run(run_name=run_name)
    
    def end_run(self):
        """End the current MLflow run."""
        mlflow.end_run()
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters to MLflow.
        
        Args:
            params: Dictionary of parameters
        """
        mlflow.log_params(params)
    
    def log_param(self, key: str, value: Any) -> None:
        """
        Log a single parameter.
        
        Args:
            key: Parameter name
            value: Parameter value
        """
        mlflow.log_param(key, value)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log metrics to MLflow.
        
        Args:
            metrics: Dictionary of metrics
            step: Optional step number
        """
        mlflow.log_metrics(metrics, step=step)
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """
        Log a single metric.
        
        Args:
            key: Metric name
            value: Metric value
            step: Optional step number
        """
        mlflow.log_metric(key, value, step=step)
    
    def log_model(
        self,
        model: nn.Module,
        artifact_path: str,
        registered_model_name: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Log a PyTorch model to MLflow.
        
        Args:
            model: PyTorch model
            artifact_path: Path to save the model
            registered_model_name: Optional name for model registry
            **kwargs: Additional arguments for mlflow.pytorch.log_model
        """
        mlflow.pytorch.log_model(
            model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name,
            **kwargs
        )
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """
        Log an artifact to MLflow.
        
        Args:
            local_path: Path to the local artifact
            artifact_path: Optional path in the artifact store
        """
        mlflow.log_artifact(local_path, artifact_path=artifact_path)
    
    def log_dict(self, dictionary: Dict, artifact_file: str) -> None:
        """
        Log a dictionary as a JSON or YAML artifact.
        
        Args:
            dictionary: Dictionary to log
            artifact_file: Name of the artifact file
        """
        mlflow.log_dict(dictionary, artifact_file)
    
    def log_text(self, text: str, artifact_file: str) -> None:
        """
        Log text as an artifact.
        
        Args:
            text: Text to log
            artifact_file: Name of the artifact file
        """
        mlflow.log_text(text, artifact_file)
    
    def log_figure(self, figure, artifact_file: str) -> None:
        """
        Log a matplotlib figure as an artifact.
        
        Args:
            figure: Matplotlib figure
            artifact_file: Name of the artifact file
        """
        mlflow.log_figure(figure, artifact_file)
    
    def set_tag(self, key: str, value: str) -> None:
        """
        Set a tag for the run.
        
        Args:
            key: Tag name
            value: Tag value
        """
        mlflow.set_tag(key, value)
    
    def set_tags(self, tags: Dict[str, str]) -> None:
        """
        Set multiple tags for the run.
        
        Args:
            tags: Dictionary of tags
        """
        mlflow.set_tags(tags)


def create_tracker(
    experiment_name: str = "lstm_stock_prediction",
    tracking_uri: Optional[str] = None
) -> MLflowTracker:
    """
    Factory function to create an MLflow tracker.
    
    Args:
        experiment_name: Name of the experiment
        tracking_uri: Optional MLflow tracking URI
    
    Returns:
        MLflowTracker: The created tracker
    """
    return MLflowTracker(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri
    )


def log_training_config(
    tracker: MLflowTracker,
    config: Dict[str, Any]
) -> None:
    """
    Log training configuration to MLflow.
    
    Args:
        tracker: MLflowTracker instance
        config: Configuration dictionary
    """
    # Flatten nested config
    flat_config = {}
    for key, value in config.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat_config[f"{key}.{sub_key}"] = sub_value
        else:
            flat_config[key] = value
    
    tracker.log_params(flat_config)


def log_metrics_summary(
    tracker: MLflowTracker,
    metrics: Dict[str, float],
    prefix: str = ""
) -> None:
    """
    Log metrics summary with optional prefix.
    
    Args:
        tracker: MLflowTracker instance
        metrics: Metrics dictionary
        prefix: Optional prefix for metric names
    """
    prefixed_metrics = {}
    for key, value in metrics.items():
        if prefix:
            prefixed_metrics[f"{prefix}_{key}"] = value
        else:
            prefixed_metrics[key] = value
    
    tracker.log_metrics(prefixed_metrics)
