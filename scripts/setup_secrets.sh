set -euo pipefail

# Standard: aktuelle UID/GID; optional via Parametern überschreiben
UID_SET="${1:-$(id -u)}"
GID_SET="${2:-$(id -g)}"
TARGET_DIR="${3:-./secrets}"

echo ">> Zielverzeichnis: ${TARGET_DIR}"
mkdir -p "${TARGET_DIR}"

# Besitz setzen
echo ">> Setze Besitzer auf ${UID_SET}:${GID_SET}"
sudo chown -R "${UID_SET}:${GID_SET}" "${TARGET_DIR}"

# Rechte setzen (rwx für Owner/Group, none für Others)
echo ">> Setze Rechte 770"
chmod 770 "${TARGET_DIR}"

# SELinux-Hinweis (falls relevant)
if command -v getenforce >/dev/null 2>&1; then
  SELINUX_STATE="$(getenforce || true)"
  if [[ "${SELINUX_STATE}" != "Disabled" ]]; then
    echo ">> SELinux erkannt (${SELINUX_STATE}). In docker-compose ggf. Volume-Option ':z' verwenden."
  fi
fi

echo ">> Fertig. Verzeichnisstruktur:"
ls -ld "${TARGET_DIR}"