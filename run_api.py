#!/usr/bin/env python3
"""
Script to start the FastAPI delivery time prediction server.
"""

import uvicorn
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from api.config import config


def main():
    """Start the FastAPI server."""
    print("="*60)
    print("DELIVERY TIME PREDICTION API".center(60))
    print("="*60)
    print(f"Starting server on {config.host}:{config.port}")
    print(f"Model path: {config.model_path}")
    print(f"Debug mode: {config.debug}")
    print("="*60)
    
    # Validate model exists
    if not config.validate_model_path():
        print(f"❌ ERROR: Model file not found at {config.model_path}")
        print("Please train a model first using the pipeline.")
        sys.exit(1)
    
    print(f"✅ Model found at {config.model_path}")
    print("🚀 Starting server...")
    print(f"📖 API docs available at: http://{config.host}:{config.port}/docs")
    print("="*60)
    
    # Start the server
    uvicorn.run(
        "api.app:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level=config.log_level.lower()
    )


if __name__ == "__main__":
    main()
