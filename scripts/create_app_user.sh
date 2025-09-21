set -eu

if [ "$#" -ne 5 ]; then
  echo "Usage: $0 <username> <groupname> <uid> <gid> <own_dir>" >&2
  exit 1
fi

USER_NAME="$1"
GROUP_NAME="$2"
USER_ID="$3"
GROUP_ID="$4"
OWN_DIR="$5"

# Gruppe anlegen (falls nicht vorhanden)
if ! getent group "$GROUP_ID" >/dev/null 2>&1 && ! getent group "$GROUP_NAME" >/dev/null 2>&1; then
  groupadd -g "$GROUP_ID" "$GROUP_NAME"
fi

# Benutzer anlegen (falls nicht vorhanden)
if ! id -u "$USER_NAME" >/dev/null 2>&1; then
  useradd -m -u "$USER_ID" -g "$GROUP_ID" -s /bin/sh "$USER_NAME"
fi

# Zielverzeichnis sicherstellen
mkdir -p "$OWN_DIR"
chown -R "$USER_ID":"$GROUP_ID" "$OWN_DIR"

# Ausgabe (zur Diagnose im Buildlog hilfreich)
echo "Created/ensured user=$USER_NAME(uid=$USER_ID) group=$GROUP_NAME(gid=$GROUP_ID), owned $OWN_DIR"
