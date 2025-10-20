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
    # Validate model exists first
    if not config.validate_model_path():
        print("="*60)
        print("DELIVERY TIME PREDICTION API".center(60))
        print("="*60)
        print(f"❌ ERROR: No trained models found!")
        print(f"Expected model path: {config.model_path}")
        
        # Show available models if any exist
        available_models = config.get_available_models()
        if available_models:
            print(f"Available models: {', '.join(available_models)}")
        else:
            print("No models found in models/ directory")
        
        print("\nTo train a model, run:")
        print("  python tests/test_pipeline.py")
        print("  # or")
        print("  cd model_pipeline/examples && python train_model.py")
        print("="*60)
        sys.exit(1)
    
    # Model exists, proceed with startup
    print("="*60)
    print("DELIVERY TIME PREDICTION API".center(60))
    print("="*60)
    print(f"Starting server on {config.host}:{config.port}")
    print(f"Model: {config.get_model_name()}")
    print(f"Model path: {config.model_path}")
    print(f"Debug mode: {config.debug}")
    print("="*60)
    
    print(f"✅ Model '{config.get_model_name()}' loaded successfully")
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
