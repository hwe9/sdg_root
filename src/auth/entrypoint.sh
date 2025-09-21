#!/bin/sh
set -e

install -d -m 0770 -o authuser -g authgroup /app/secrets
umask 007
exec gosu authuser:authgroup "$@"
