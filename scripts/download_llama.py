import urllib.request
import zipfile
import json
import os
import sys

def main():
    print('Fetching latest release info...')
    req = urllib.request.Request('https://api.github.com/repos/ggml-org/llama.cpp/releases/latest')
    req.add_header('User-Agent', 'Mozilla/5.0')
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read())
        
        url = next(asset['browser_download_url'] for asset in data['assets'] if 'llama-b' in asset['name'] and 'win-cuda-12.4-x64.zip' in asset['name'])
        print(f'Downloading from: {url}')
        print('This might take a few minutes (approx 380MB). Please wait...')
        
        urllib.request.urlretrieve(url, 'llama.zip')
        print('Download complete!')
        
        print('Extracting to llama_bin...')
        with zipfile.ZipFile('llama.zip', 'r') as zip_ref:
            zip_ref.extractall('llama_bin')
            
        print('Success! llama-server.exe is ready.')
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
