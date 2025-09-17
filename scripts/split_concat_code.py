import os
from pathlib import Path

def split_files_by_structure(base_path, output_base_path):    
    
    os.makedirs(output_base_path, exist_ok=True)
    
    base_files_to_concat = [
        'docker-compose.yml',
        'Dockerfile', 
        'requirements.txt',
        'requirements-security.txt',
        '.env',
        '.gitignore',
        'nginx/nginx.conf',
        'redis/redis.conf'
    ]
    
    base_output_file = os.path.join(output_base_path, 'sdg_root_base_files.txt')
    
    with open(base_output_file, 'w', encoding='utf-8') as base_outfile:
        base_outfile.write(f"=== SDG Root Base Files ===\n\n")
        
        for filename in base_files_to_concat:
            file_path = os.path.join(base_path, filename)
            if os.path.exists(file_path) and os.path.isfile(file_path):
                base_outfile.write(f"### /source {file_path}\n")
                if is_text_file(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            content = infile.read()
                        base_outfile.write(content)
                    except UnicodeDecodeError:
                        try:
                            with open(file_path, 'r', encoding='latin-1') as infile:
                                content = infile.read()
                            base_outfile.write(content)
                        except Exception as e:
                            base_outfile.write(f"# Error reading file {file_path}: {e}\n")
                else:
                    file_size = os.path.getsize(file_path)
                    base_outfile.write(f"# Binary file (Size: {file_size} bytes) - Content not displayed\n")
                base_outfile.write('\n\n')
            else:
                base_outfile.write(f"### /source {file_path}\n")
                base_outfile.write(f"# File not found: {file_path}\n\n")
    
    print(f"✓ Basis-Dateien geschrieben nach: {base_output_file}")
    
    # 2. Für jedes Verzeichnis in src/ eine separate Datei erstellen
    src_path = os.path.join(base_path, 'src')
    
    if os.path.exists(src_path) and os.path.isdir(src_path):
        # Alle Unterverzeichnisse in src/ finden (außer .venv)
        src_directories = [d for d in os.listdir(src_path) 
                          if os.path.isdir(os.path.join(src_path, d)) and d != '.venv']
        
        for dir_name in src_directories:
            dir_path = os.path.join(src_path, dir_name)
            output_file = os.path.join(output_base_path, f'src_{dir_name}_all_files.txt')
            
            # Alle Dateien in diesem Verzeichnis und Unterverzeichnissen sammeln
            files_to_process = get_all_files_recursive(dir_path, exclude_dirs=['.venv'])
            files_to_process.sort()
            
            with open(output_file, 'w', encoding='utf-8') as outfile:
                outfile.write(f"=== SDG Source: {dir_name} ===\n\n")
                
                for file_path in files_to_process:
                    if os.path.exists(file_path) and os.path.isfile(file_path):
                        outfile.write(f"### /source {file_path}\n")
                        if is_text_file(file_path):
                            try:
                                with open(file_path, 'r', encoding='utf-8') as infile:
                                    content = infile.read()
                                outfile.write(content)
                            except UnicodeDecodeError:
                                try:
                                    with open(file_path, 'r', encoding='latin-1') as infile:
                                        content = infile.read()
                                    outfile.write(content)
                                except Exception as e:
                                    outfile.write(f"# Error reading file {file_path}: {e}\n")
                        else:
                            file_size = os.path.getsize(file_path)
                            outfile.write(f"# Binary file (Size: {file_size} bytes) - Content not displayed\n")
                        outfile.write('\n\n')
            
            print(f"✓ {len(files_to_process)} Dateien für '{dir_name}' geschrieben nach: {output_file}")
    else:
        print(f"⚠ src/ Verzeichnis nicht gefunden: {src_path}")

def get_all_files_recursive(root_path, exclude_dirs=None):
    if exclude_dirs is None:
        exclude_dirs = []
    
    all_files = []
    
    for root, dirs, files in os.walk(root_path):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)
    
    return all_files

def is_text_file(file_path):
    text_extensions = {
        '.txt', '.py', '.js', '.html', '.css', '.md', '.json', '.xml', '.yaml', '.yml',
        '.conf', '.cfg', '.ini', '.log', '.csv', '.sql', '.sh', '.bat', '.ps1',
        '.dockerfile', '.gitignore', '.env', '.properties', '.toml', '.requirements'
    }
    
    if Path(file_path).suffix.lower() in text_extensions:
        return True
    
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(1024)
            if b'\0' in chunk:
                return False
            try:
                chunk.decode('utf-8')
                return True
            except UnicodeDecodeError:
                return False
    except:
        return False

if __name__ == "__main__":
    base_path = '/mnt/gigabyte1tb/SDG/sdg_root'  
    output_base_path = '/mnt/gigabyte1tb/SDG/output_split_files'  
    
    split_files_by_structure(base_path, output_base_path)
    print("Fertig! Dateien wurden aufgeteilt.")
