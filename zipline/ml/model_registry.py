"""
Model Registry for Zipline

This module provides centralized model management, versioning, and deployment
capabilities for machine learning models in algorithmic trading.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import json
import pickle
import os
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import joblib

from .base import MLModel

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for a registered model."""
    model_id: str
    name: str
    version: str
    model_type: str
    created_at: datetime
    updated_at: datetime
    description: str
    tags: List[str]
    performance_metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    feature_names: List[str]
    training_data_size: int
    model_size_bytes: int
    checksum: str
    status: str  # 'active', 'archived', 'deprecated'


class ModelRegistry:
    """
    Centralized model registry for managing ML models.
    """
    
    def __init__(self, registry_path: str = "./model_registry"):
        """
        Initialize the model registry.
        
        Parameters
        ----------
        registry_path : str
            Path to store model registry data.
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(exist_ok=True)
        
        self.models_path = self.registry_path / "models"
        self.models_path.mkdir(exist_ok=True)
        
        self.metadata_path = self.registry_path / "metadata"
        self.metadata_path.mkdir(exist_ok=True)
        
        self.models: Dict[str, ModelMetadata] = {}
        self._load_registry()
    
    def _load_registry(self):
        """Load existing model metadata from disk."""
        metadata_files = list(self.metadata_path.glob("*.json"))
        
        for metadata_file in metadata_files:
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                    # Convert datetime strings back to datetime objects
                    data['created_at'] = datetime.fromisoformat(data['created_at'])
                    data['updated_at'] = datetime.fromisoformat(data['updated_at'])
                    metadata = ModelMetadata(**data)
                    self.models[metadata.model_id] = metadata
            except Exception as e:
                logger.warning(f"Could not load metadata from {metadata_file}: {e}")
    
    def _save_metadata(self, metadata: ModelMetadata):
        """Save model metadata to disk."""
        metadata_file = self.metadata_path / f"{metadata.model_id}.json"
        
        # Convert datetime objects to strings for JSON serialization
        data = asdict(metadata)
        data['created_at'] = metadata.created_at.isoformat()
        data['updated_at'] = metadata.updated_at.isoformat()
        
        with open(metadata_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _generate_model_id(self, name: str, version: str) -> str:
        """Generate a unique model ID."""
        timestamp = datetime.now().isoformat()
        unique_string = f"{name}_{version}_{timestamp}"
        return hashlib.md5(unique_string.encode()).hexdigest()[:12]
    
    def _calculate_checksum(self, model_data: bytes) -> str:
        """Calculate checksum for model data."""
        return hashlib.sha256(model_data).hexdigest()
    
    def register_model(self,
                      model: MLModel,
                      name: str,
                      version: str,
                      description: str = "",
                      tags: Optional[List[str]] = None,
                      performance_metrics: Optional[Dict[str, float]] = None) -> str:
        """
        Register a model in the registry.
        
        Parameters
        ----------
        model : MLModel
            The model to register.
        name : str
            Model name.
        version : str
            Model version.
        description : str
            Model description.
        tags : List[str], optional
            Tags for the model.
        performance_metrics : Dict[str, float], optional
            Performance metrics for the model.
            
        Returns
        -------
        str
            Model ID.
        """
        # Generate model ID
        model_id = self._generate_model_id(name, version)
        
        # Serialize model
        model_data = pickle.dumps(model)
        model_size = len(model_data)
        checksum = self._calculate_checksum(model_data)
        
        # Save model to disk
        model_file = self.models_path / f"{model_id}.pkl"
        with open(model_file, 'wb') as f:
            f.write(model_data)
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            name=name,
            version=version,
            model_type=model.model_type,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            description=description,
            tags=tags or [],
            performance_metrics=performance_metrics or {},
            hyperparameters=getattr(model, 'hyperparameters', {}),
            feature_names=getattr(model, 'feature_names', []),
            training_data_size=getattr(model, 'training_data_size', 0),
            model_size_bytes=model_size,
            checksum=checksum,
            status='active'
        )
        
        # Save metadata
        self._save_metadata(metadata)
        self.models[model_id] = metadata
        
        logger.info(f"Registered model {name} v{version} with ID {model_id}")
        return model_id
    
    def get_model(self, model_id: str) -> Optional[MLModel]:
        """
        Retrieve a model from the registry.
        
        Parameters
        ----------
        model_id : str
            Model ID.
            
        Returns
        -------
        MLModel or None
            The model if found, None otherwise.
        """
        if model_id not in self.models:
            logger.warning(f"Model {model_id} not found in registry")
            return None
        
        metadata = self.models[model_id]
        model_file = self.models_path / f"{model_id}.pkl"
        
        if not model_file.exists():
            logger.error(f"Model file not found: {model_file}")
            return None
        
        try:
            with open(model_file, 'rb') as f:
                model_data = f.read()
            
            # Verify checksum
            calculated_checksum = self._calculate_checksum(model_data)
            if calculated_checksum != metadata.checksum:
                logger.error(f"Checksum mismatch for model {model_id}")
                return None
            
            # Deserialize model
            model = pickle.loads(model_data)
            return model
            
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
            return None
    
    def list_models(self, 
                   name: Optional[str] = None,
                   model_type: Optional[str] = None,
                   tags: Optional[List[str]] = None,
                   status: Optional[str] = None) -> List[ModelMetadata]:
        """
        List models in the registry with optional filtering.
        
        Parameters
        ----------
        name : str, optional
            Filter by model name.
        model_type : str, optional
            Filter by model type.
        tags : List[str], optional
            Filter by tags.
        status : str, optional
            Filter by status.
            
        Returns
        -------
        List[ModelMetadata]
            List of matching models.
        """
        filtered_models = []
        
        for metadata in self.models.values():
            # Apply filters
            if name and metadata.name != name:
                continue
            if model_type and metadata.model_type != model_type:
                continue
            if tags and not any(tag in metadata.tags for tag in tags):
                continue
            if status and metadata.status != status:
                continue
            
            filtered_models.append(metadata)
        
        # Sort by updated_at (most recent first)
        filtered_models.sort(key=lambda x: x.updated_at, reverse=True)
        return filtered_models
    
    def update_model(self,
                    model_id: str,
                    model: Optional[MLModel] = None,
                    description: Optional[str] = None,
                    tags: Optional[List[str]] = None,
                    performance_metrics: Optional[Dict[str, float]] = None,
                    status: Optional[str] = None) -> bool:
        """
        Update a model in the registry.
        
        Parameters
        ----------
        model_id : str
            Model ID to update.
        model : MLModel, optional
            New model instance.
        description : str, optional
            New description.
        tags : List[str], optional
            New tags.
        performance_metrics : Dict[str, float], optional
            New performance metrics.
        status : str, optional
            New status.
            
        Returns
        -------
        bool
            True if update successful, False otherwise.
        """
        if model_id not in self.models:
            logger.warning(f"Model {model_id} not found in registry")
            return False
        
        metadata = self.models[model_id]
        
        # Update model file if provided
        if model is not None:
            model_data = pickle.dumps(model)
            model_size = len(model_data)
            checksum = self._calculate_checksum(model_data)
            
            model_file = self.models_path / f"{model_id}.pkl"
            with open(model_file, 'wb') as f:
                f.write(model_data)
            
            metadata.model_size_bytes = model_size
            metadata.checksum = checksum
            metadata.hyperparameters = getattr(model, 'hyperparameters', {})
            metadata.feature_names = getattr(model, 'feature_names', [])
            metadata.training_data_size = getattr(model, 'training_data_size', 0)
        
        # Update metadata fields
        if description is not None:
            metadata.description = description
        if tags is not None:
            metadata.tags = tags
        if performance_metrics is not None:
            metadata.performance_metrics = performance_metrics
        if status is not None:
            metadata.status = status
        
        metadata.updated_at = datetime.now()
        
        # Save updated metadata
        self._save_metadata(metadata)
        
        logger.info(f"Updated model {model_id}")
        return True
    
    def delete_model(self, model_id: str) -> bool:
        """
        Delete a model from the registry.
        
        Parameters
        ----------
        model_id : str
            Model ID to delete.
            
        Returns
        -------
        bool
            True if deletion successful, False otherwise.
        """
        if model_id not in self.models:
            logger.warning(f"Model {model_id} not found in registry")
            return False
        
        try:
            # Delete model file
            model_file = self.models_path / f"{model_id}.pkl"
            if model_file.exists():
                model_file.unlink()
            
            # Delete metadata file
            metadata_file = self.metadata_path / f"{model_id}.json"
            if metadata_file.exists():
                metadata_file.unlink()
            
            # Remove from memory
            del self.models[model_id]
            
            logger.info(f"Deleted model {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting model {model_id}: {e}")
            return False
    
    def get_latest_model(self, name: str, model_type: Optional[str] = None) -> Optional[MLModel]:
        """
        Get the latest version of a model by name.
        
        Parameters
        ----------
        name : str
            Model name.
        model_type : str, optional
            Model type filter.
            
        Returns
        -------
        MLModel or None
            The latest model if found, None otherwise.
        """
        models = self.list_models(name=name, model_type=model_type, status='active')
        
        if not models:
            return None
        
        # Get the most recently updated model
        latest_model = models[0]
        return self.get_model(latest_model.model_id)
    
    def compare_models(self, model_id1: str, model_id2: str) -> Dict[str, Any]:
        """
        Compare two models.
        
        Parameters
        ----------
        model_id1 : str
            First model ID.
        model_id2 : str
            Second model ID.
            
        Returns
        -------
        Dict[str, Any]
            Comparison results.
        """
        if model_id1 not in self.models or model_id2 not in self.models:
            logger.warning("One or both models not found in registry")
            return {}
        
        metadata1 = self.models[model_id1]
        metadata2 = self.models[model_id2]
        
        comparison = {
            'model1': {
                'id': metadata1.model_id,
                'name': metadata1.name,
                'version': metadata1.version,
                'created_at': metadata1.created_at,
                'performance_metrics': metadata1.performance_metrics
            },
            'model2': {
                'id': metadata2.model_id,
                'name': metadata2.name,
                'version': metadata2.version,
                'created_at': metadata2.created_at,
                'performance_metrics': metadata2.performance_metrics
            },
            'differences': {}
        }
        
        # Compare performance metrics
        metrics1 = set(metadata1.performance_metrics.keys())
        metrics2 = set(metadata2.performance_metrics.keys())
        all_metrics = metrics1.union(metrics2)
        
        for metric in all_metrics:
            val1 = metadata1.performance_metrics.get(metric, None)
            val2 = metadata2.performance_metrics.get(metric, None)
            
            if val1 != val2:
                comparison['differences'][metric] = {
                    'model1': val1,
                    'model2': val2,
                    'difference': val2 - val1 if val1 is not None and val2 is not None else None
                }
        
        return comparison
    
    def export_model(self, model_id: str, export_path: str) -> bool:
        """
        Export a model to a file.
        
        Parameters
        ----------
        model_id : str
            Model ID to export.
        export_path : str
            Path to export the model to.
            
        Returns
        -------
        bool
            True if export successful, False otherwise.
        """
        model = self.get_model(model_id)
        if model is None:
            return False
        
        try:
            # Export model
            joblib.dump(model, export_path)
            
            # Export metadata
            metadata = self.models[model_id]
            metadata_path = export_path.replace('.pkl', '_metadata.json')
            
            data = asdict(metadata)
            data['created_at'] = metadata.created_at.isoformat()
            data['updated_at'] = metadata.updated_at.isoformat()
            
            with open(metadata_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Exported model {model_id} to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting model {model_id}: {e}")
            return False
    
    def import_model(self, model_path: str, metadata_path: str) -> Optional[str]:
        """
        Import a model from a file.
        
        Parameters
        ----------
        model_path : str
            Path to the model file.
        metadata_path : str
            Path to the metadata file.
            
        Returns
        -------
        str or None
            Model ID if import successful, None otherwise.
        """
        try:
            # Load model
            model = joblib.load(model_path)
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                data = json.load(f)
                data['created_at'] = datetime.fromisoformat(data['created_at'])
                data['updated_at'] = datetime.fromisoformat(data['updated_at'])
                metadata = ModelMetadata(**data)
            
            # Register the imported model
            return self.register_model(
                model=model,
                name=metadata.name,
                version=metadata.version,
                description=metadata.description,
                tags=metadata.tags,
                performance_metrics=metadata.performance_metrics
            )
            
        except Exception as e:
            logger.error(f"Error importing model: {e}")
            return None
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the model registry.
        
        Returns
        -------
        Dict[str, Any]
            Registry statistics.
        """
        if not self.models:
            return {
                'total_models': 0,
                'model_types': {},
                'status_distribution': {},
                'total_size_bytes': 0,
                'oldest_model': None,
                'newest_model': None
            }
        
        stats = {
            'total_models': len(self.models),
            'model_types': {},
            'status_distribution': {},
            'total_size_bytes': 0,
            'oldest_model': None,
            'newest_model': None
        }
        
        oldest_date = datetime.now()
        newest_date = datetime.min
        
        for metadata in self.models.values():
            # Model types
            model_type = metadata.model_type
            stats['model_types'][model_type] = stats['model_types'].get(model_type, 0) + 1
            
            # Status distribution
            status = metadata.status
            stats['status_distribution'][status] = stats['status_distribution'].get(status, 0) + 1
            
            # Total size
            stats['total_size_bytes'] += metadata.model_size_bytes
            
            # Date range
            if metadata.created_at < oldest_date:
                oldest_date = metadata.created_at
                stats['oldest_model'] = metadata.model_id
            
            if metadata.created_at > newest_date:
                newest_date = metadata.created_at
                stats['newest_model'] = metadata.model_id
        
        return stats 