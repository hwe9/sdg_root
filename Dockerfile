# Basierend auf dem offiziellen Transformer-Inference-Image
FROM cr.weaviate.io/semitechnologies/transformers-inference:sentence-transformers-multi-qa-MiniLM-L6-cos-v1

# Temporär als root, um Pakete zu installieren (bei Minimal-Images oft Standard)
USER root

# Installiere Debug- und Diagnose-Tools
# Für Debian/Ubuntu-Basis (bei den meisten KI-Images Standard)
RUN apt-get update && apt-get install -y wget curl

# Optional: weitere Tools für wissenschaftliches Debuggen
# RUN apt-get install -y procps net-tools

# (Falls benötigt:) Wechsle zurück zum originalen Nutzer, der App ausführt
# In Weaviate-Images ist meist root default, dann kann die Zeile entfallen
# USER hwe

# Keine CMD nötig – wird vom Baseimage geerbt


