import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

def get_models():
    print(f"Checking API Key: {API_KEY[:5]}... (hidden)")
    
    # We ask Google's server directly
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        print("\nSUCCESS! Here are your available models:\n")
        available_models = []
        for model in data.get('models', []):
            # We only care about models that can 'generateContent' (Chat)
            if "generateContent" in model.get("supportedGenerationMethods", []):
                # Remove the "models/" prefix for easier reading
                clean_name = model['name'].replace("models/", "")
                print(f"✅ {clean_name}")
                available_models.append(clean_name)
        
        return available_models
    else:
        print(f"\n❌ ERROR: Could not connect. Status: {response.status_code}")
        print(f"Message: {response.text}")
        return []

if __name__ == "__main__":
    get_models()