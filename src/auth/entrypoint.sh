set -euo pipefail

# Zielverzeichnis aus ENV (standard-kompatibel zum SecretsManager)
SECRET_DIR="${SECRET_STORE_DIR:-/data/secrets}"
export SECRET_STORE_DIR="$SECRET_DIR"

# Ordner anlegen und Rechte setzen; falls RO, nur warnen (Service kann dann nicht persistieren)
if ! install -d -m 0770 -o authuser -g authgroup "$SECRET_DIR"; then
  echo "WARN: cannot ensure secret dir $SECRET_DIR (read-only?)." >&2
fi

umask 007
exec gosu authuser:authgroup "$@"