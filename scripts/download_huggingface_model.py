#!/usr/bin/env python3
"""
Script to pre-download HuggingFace models for offline use.
Run this on a machine with internet access, then copy the cache to your Docker host.
"""
import os
import sys

def download_model(model_name: str = "paraphrase-multilingual-MiniLM-L12-v2", 
                   cache_dir: str = None):
    """Download and cache a HuggingFace model"""
    print(f"📥 Downloading model: {model_name}")
    
    try:
        # Import here to catch errors gracefully
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            print(f"❌ Error importing sentence_transformers: {e}")
            print("💡 Install with: pip install sentence-transformers")
            return False
        
        # Download model (will cache automatically)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            print(f"📂 Using cache directory: {cache_dir}")
            model = SentenceTransformer(model_name, cache_folder=cache_dir)
        else:
            model = SentenceTransformer(model_name)
        
        # Test encoding to ensure model works
        test_text = "This is a test sentence for the embedding model."
        embeddings = model.encode(test_text)
        
        print(f"✅ Model downloaded successfully!")
        print(f"   Model dimension: {model.get_sentence_embedding_dimension()}")
        print(f"   Test embedding shape: {embeddings.shape}")
        
        # Print cache location
        if cache_dir:
            cache_location = cache_dir
        else:
            # Default HuggingFace cache locations
            cache_location = os.path.expanduser("~/.cache/huggingface")
            if not os.path.exists(cache_location):
                cache_location = os.path.expanduser("~/.cache/huggingface")
        
        print(f"📁 Cache location: {cache_location}")
        print(f"\n💡 To use this in Docker:")
        print(f"   1. Copy cache to: ${{DATA_ROOT}}/cache/huggingface")
        print(f"   2. Example: cp -r {cache_location}/* ${{DATA_ROOT}}/cache/huggingface/")
        print(f"   3. Or mount your local cache: ~/.cache/huggingface -> /cache/huggingface")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "paraphrase-multilingual-MiniLM-L12-v2"
    cache_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = download_model(model_name, cache_dir)
    sys.exit(0 if success else 1)

