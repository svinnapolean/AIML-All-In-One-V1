"""
Data Router

API endpoints for data management, upload, and preprocessing
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import io

from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel


router = APIRouter(prefix="/data", tags=["data"])


class DatasetInfo(BaseModel):
    """Dataset information model"""
    name: str
    size: int
    rows: int
    columns: int
    file_type: str
    created_at: str
    modified_at: str


class DataPreviewResponse(BaseModel):
    """Data preview response model"""
    filename: str
    shape: List[int]
    columns: List[str]
    dtypes: Dict[str, str]
    preview: List[Dict[str, Any]]
    statistics: Dict[str, Any]


class DataProcessingRequest(BaseModel):
    """Data processing request model"""
    filename: str
    operations: List[Dict[str, Any]]


@router.get("/", summary="List all datasets")
async def list_datasets():
    """Get a list of all uploaded datasets"""
    try:
        data_path = "data"
        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)
            return {"datasets": []}
        
        datasets = []
        for file in os.listdir(data_path):
            if file.endswith(('.csv', '.xlsx', '.json')):
                file_path = os.path.join(data_path, file)
                stat = os.stat(file_path)
                
                # Try to get dataset dimensions
                try:
                    if file.endswith('.csv'):
                        df = pd.read_csv(file_path, nrows=1)
                        full_df = pd.read_csv(file_path)
                    elif file.endswith('.xlsx'):
                        df = pd.read_excel(file_path, nrows=1)
                        full_df = pd.read_excel(file_path)
                    elif file.endswith('.json'):
                        df = pd.read_json(file_path, nrows=1)
                        full_df = pd.read_json(file_path)
                    
                    rows, columns = full_df.shape
                except Exception:
                    rows, columns = 0, 0
                
                datasets.append(DatasetInfo(
                    name=file,
                    size=stat.st_size,
                    rows=rows,
                    columns=columns,
                    file_type=file.split('.')[-1].upper(),
                    created_at=datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    modified_at=datetime.fromtimestamp(stat.st_mtime).isoformat()
                ))
        
        return {"datasets": datasets}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing datasets: {str(e)}")


@router.post("/upload", summary="Upload a dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a new dataset file"""
    try:
        # Validate file type
        if not file.filename or not file.filename.endswith(('.csv', '.xlsx', '.json')):
            raise HTTPException(status_code=400, detail="Only CSV, Excel, and JSON files are supported")
        
        # Ensure data directory exists
        data_path = "data"
        os.makedirs(data_path, exist_ok=True)
        
        # Save file
        file_path = os.path.join(data_path, file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Get file info
        stat = os.stat(file_path)
        
        return {
            "message": f"File {file.filename} uploaded successfully",
            "filename": file.filename,
            "size": stat.st_size,
            "uploaded_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")


@router.get("/{filename}/preview", response_model=DataPreviewResponse, summary="Preview dataset")
async def preview_dataset(
    filename: str,
    rows: int = Query(default=10, description="Number of rows to preview")
):
    """Get a preview of the dataset with basic statistics"""
    try:
        file_path = os.path.join("data", filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Load data based on file type
        if filename.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        elif filename.endswith('.json'):
            df = pd.read_json(file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        # Get preview data
        preview_data = df.head(rows).fillna("").to_dict('records')
        
        # Get basic statistics
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        statistics = {}
        
        if len(numeric_columns) > 0:
            stats_df = df[numeric_columns].describe()
            statistics = stats_df.to_dict()
        
        # Add data types
        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        return DataPreviewResponse(
            filename=filename,
            shape=list(df.shape),
            columns=df.columns.tolist(),
            dtypes=dtypes,
            preview=preview_data,
            statistics=statistics
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error previewing dataset: {str(e)}")


@router.get("/{filename}/download", summary="Download dataset")
async def download_dataset(filename: str):
    """Download a dataset file"""
    try:
        file_path = os.path.join("data", filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type='application/octet-stream'
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading dataset: {str(e)}")


@router.post("/{filename}/process", summary="Process dataset")
async def process_dataset(filename: str, request: DataProcessingRequest):
    """Apply processing operations to a dataset"""
    try:
        file_path = os.path.join("data", filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Load data
        if filename.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        elif filename.endswith('.json'):
            df = pd.read_json(file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        # Apply operations
        for operation in request.operations:
            op_type = operation.get('type')
            
            if op_type == 'drop_nulls':
                df = df.dropna()
            elif op_type == 'fill_nulls':
                fill_value = operation.get('value', 0)
                df = df.fillna(fill_value)
            elif op_type == 'drop_column':
                column = operation.get('column')
                if column in df.columns:
                    df = df.drop(columns=[column])
            elif op_type == 'rename_column':
                old_name = operation.get('old_name')
                new_name = operation.get('new_name')
                if old_name in df.columns:
                    df = df.rename(columns={old_name: new_name})
            elif op_type == 'normalize':
                columns = operation.get('columns', [])
                for col in columns:
                    if col in df.columns and df[col].dtype in ['int64', 'float64']:
                        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        
        # Save processed data
        processed_filename = f"processed_{filename}"
        processed_path = os.path.join("data", processed_filename)
        
        if filename.endswith('.csv'):
            df.to_csv(processed_path, index=False)
        elif filename.endswith('.xlsx'):
            df.to_excel(processed_path, index=False)
        elif filename.endswith('.json'):
            df.to_json(processed_path, orient='records')
        
        return {
            "message": "Dataset processed successfully",
            "original_filename": filename,
            "processed_filename": processed_filename,
            "original_shape": list(pd.read_csv(file_path).shape) if filename.endswith('.csv') else [],
            "processed_shape": list(df.shape),
            "operations_applied": len(request.operations)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing dataset: {str(e)}")


@router.delete("/{filename}", summary="Delete dataset")
async def delete_dataset(filename: str):
    """Delete a dataset file"""
    try:
        file_path = os.path.join("data", filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        os.remove(file_path)
        
        return {"message": f"Dataset {filename} deleted successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting dataset: {str(e)}")


@router.get("/{filename}/statistics", summary="Get dataset statistics")
async def get_dataset_statistics(filename: str):
    """Get comprehensive statistics for a dataset"""
    try:
        file_path = os.path.join("data", filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Load data
        if filename.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        elif filename.endswith('.json'):
            df = pd.read_json(file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        # Calculate comprehensive statistics
        stats = {
            "shape": list(df.shape),
            "columns": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_values": df.isnull().sum().to_dict(),
            "memory_usage": df.memory_usage(deep=True).to_dict()
        }
        
        # Numeric statistics
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            stats["numeric_statistics"] = df[numeric_columns].describe().to_dict()
        
        # Categorical statistics
        categorical_columns = df.select_dtypes(include=['object']).columns
        if len(categorical_columns) > 0:
            stats["categorical_statistics"] = {}
            for col in categorical_columns:
                stats["categorical_statistics"][col] = {
                    "unique_values": int(df[col].nunique()),
                    "most_frequent": df[col].mode().iloc[0] if not df[col].mode().empty else None,
                    "frequency": df[col].value_counts().head().to_dict()
                }
        
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting dataset statistics: {str(e)}")