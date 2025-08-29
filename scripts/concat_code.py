import os
from pathlib import Path

import os
from pathlib import Path

def write_files_with_directory_exclusion(folder_config, output_base_path):
    """
    Schreibt Dateien in Output-Files mit flexibler Verzeichnis-Ausschließung.
    
    Args:
        folder_config (dict): Konfiguration pro Ordner
            {ordnerpfad: {
                'files': ['specific1.py', 'specific2.py'] oder 'all',
                'extensions': ['.py', '.txt'] oder 'all',
                'output_name': 'custom_name.txt',
                'recursive': True/False,
                'exclude_dirs': ['__pycache__', '.git', 'node_modules'],  # NEU!
                'exclude_patterns': ['*.pyc', '*.log'],  # NEU!
                'exclude_extensions': ['.exe', '.bin']
            }}
        output_base_path (str): Basis-Pfad für alle Output-Files
    """
    
    for folder, config in folder_config.items():
        # Output-Datei bestimmen
        output_filename = config.get('output_name', f"{os.path.basename(folder)}_filtered.txt")
        output_path = os.path.join(output_base_path, output_filename)
        
        # Output-Verzeichnis erstellen
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as outfile:
            files_to_process = []
            exclude_dirs = config.get('exclude_dirs', [])
            exclude_patterns = config.get('exclude_patterns', [])
            
            # Rekursive Suche mit Verzeichnis-Ausschluss
            def get_files_with_exclusion(root_path):
                result_files = []
                
                for root, dirs, files in os.walk(root_path):
                    # Verzeichnisse ausschließen (modifiziert dirs in-place)
                    dirs[:] = [d for d in dirs if not should_exclude_directory(d, exclude_dirs, root)]
                    
                    for file in files:
                        file_path = os.path.join(root, file)
                        if not should_exclude_file(file_path, exclude_patterns):
                            result_files.append(file_path)
                
                return result_files
            
            # Bestimme welche Dateien verarbeitet werden sollen
            if config.get('recursive', False):
                if config['files'] == 'all':
                    # Alle Dateien rekursiv mit Ausschlüssen
                    files_to_process = get_files_with_exclusion(folder)
                else:
                    # Spezifische Dateien rekursiv suchen
                    all_files = get_files_with_exclusion(folder)
                    specific_files = set(config['files'])
                    files_to_process = [f for f in all_files if os.path.basename(f) in specific_files]
            else:
                # Nur Hauptordner (ohne Rekursion)
                if config['files'] == 'all':
                    files_to_process = [os.path.join(folder, f) for f in os.listdir(folder) 
                                     if os.path.isfile(os.path.join(folder, f)) 
                                     and not should_exclude_file(os.path.join(folder, f), exclude_patterns)]
                else:
                    for filename in config['files']:
                        file_path = os.path.join(folder, filename)
                        if os.path.exists(file_path) and not should_exclude_file(file_path, exclude_patterns):
                            files_to_process.append(file_path)
            
            # Extension-Filter anwenden
            if 'extensions' in config and config['extensions'] != 'all':
                allowed_extensions = [ext.lower() for ext in config['extensions']]
                files_to_process = [f for f in files_to_process 
                                  if Path(f).suffix.lower() in allowed_extensions]
            
            # Ausschließen bestimmter Dateiendungen
            if 'exclude_extensions' in config:
                excluded_extensions = [ext.lower() for ext in config['exclude_extensions']]
                files_to_process = [f for f in files_to_process 
                                  if Path(f).suffix.lower() not in excluded_extensions]
            
            # Dateien sortieren
            files_to_process.sort()
            
            # Jede Datei verarbeiten
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
            
            print(f"✓ {len(files_to_process)} Dateien geschrieben nach: {output_path}")
            if exclude_dirs:
                print(f"  → Ausgeschlossene Verzeichnisse: {exclude_dirs}")

def should_exclude_directory(dirname, exclude_dirs, current_path):
    """Prüft ob ein Verzeichnis ausgeschlossen werden soll"""
    # Einfacher Name-Match
    if dirname in exclude_dirs:
        return True
    
    # Pattern-Match (z.B. für versteckte Ordner)
    import fnmatch
    for pattern in exclude_dirs:
        if fnmatch.fnmatch(dirname, pattern):
            return True
    
    # Pfad-basierte Ausschlüsse
    full_path = os.path.join(current_path, dirname)
    for exclude_pattern in exclude_dirs:
        if exclude_pattern.startswith('/') and exclude_pattern in full_path:
            return True
    
    return False

def should_exclude_file(filepath, exclude_patterns):
    """Prüft ob eine Datei aufgrund von Patterns ausgeschlossen werden soll"""
    import fnmatch
    filename = os.path.basename(filepath)
    
    for pattern in exclude_patterns:
        if fnmatch.fnmatch(filename, pattern) or fnmatch.fnmatch(filepath, pattern):
            return True
    return False

def is_text_file(file_path):
    """Prüft ob eine Datei als Text lesbar ist"""
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

# BEISPIEL-VERWENDUNG mit Verzeichnis-Ausschluss:
if __name__ == "__main__":
    
    folder_config = {
        # SDG-Projekt komplett, aber ohne unnötige Ordner
        '/mnt/gigabyte1tb/SDG': {
            'files': 'all',
            'extensions': ['.py', '.yml', '.yaml', '.txt', '.md', '.env', '.dockerfile'],
            'output_name': 'sdg_project_clean.txt',
            'recursive': True,
            'exclude_dirs': [
                'Book',
                'oss20b',
                'threads',
                'data', 
                '__pycache__',     # Python Cache
                '.git',            # Git Repository 
                '.pytest_cache',   # Pytest Cache
                'node_modules',    # Node.js Dependencies
                '.venv',           # Virtual Environment
                'venv',            # Virtual Environment
                '.idea',           # PyCharm Files
                '.vscode',         # VS Code Files
                'logs',            # Log-Verzeichnisse
                'temp',            # Temporäre Dateien
                '.*'               # Alle versteckten Ordner
            ],
            'exclude_patterns': [
                '*.pyc',           # Python Compiled Files
                '*.pyo',           # Python Optimized Files  
                '*.log',           # Log Files
                '*.tmp',           # Temporary Files
                '.DS_Store',       # macOS Files
                'Thumbs.db'        # Windows Thumbnails
            ],
            'exclude_extensions': ['.exe', '.dll', '.so', '.dylib', '.bin']
        },
        
        # Nur Source-Code, sehr restriktiv
        # '/home/user/sdg_project/src': {
        #     'files': 'all',
        #     'extensions': ['.py'],
        #     'output_name': 'source_code_only.txt',
        #     'recursive': True,
        #     'exclude_dirs': ['__pycache__', 'tests', '.pytest_cache'],
        #     'exclude_patterns': ['test_*.py', '*_test.py']
        # }
    }




output_base_path = '/mnt/gigabyte1tb/SDG/sdg_root'
write_files_with_directory_exclusion(folder_config, output_base_path)

print("Fertig! Output-File erzeugt.")
