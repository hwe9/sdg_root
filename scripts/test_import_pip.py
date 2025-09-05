# test_imports.py
import sys

def test_imports():
    try:
        import bcrypt
        print("✓ bcrypt imported successfully")
        
        # Test bcrypt functionality
        password = "test_password"
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        is_valid = bcrypt.checkpw(password.encode('utf-8'), hashed)
        print(f"✓ bcrypt functionality test: {'PASSED' if is_valid else 'FAILED'}")
        
    except ImportError as e:
        print(f"✗ bcrypt import failed: {e}")
    
    try:
        import jwt
        print("✓ jwt imported successfully")
        
        # Test JWT functionality
        payload = {"test": "data"}
        secret = "test_secret"
        token = jwt.encode(payload, secret, algorithm="HS256")
        decoded = jwt.decode(token, secret, algorithms=["HS256"])
        print(f"✓ jwt functionality test: {'PASSED' if decoded['test'] == 'data' else 'FAILED'}")
        
    except ImportError as e:
        print(f"✗ jwt import failed: {e}")
    except Exception as e:
        print(f"✗ jwt functionality test failed: {e}")

if __name__ == "__main__":
    test_imports()
