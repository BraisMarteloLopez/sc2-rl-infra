#!/usr/bin/env bash
# Arranca/para el servidor VNC en Brais para ver agentes en vivo (vía B; ver
# NOTES.md §7). El servidor escucha solo en loopback; el túnel a Windows lo hace
# MobaXterm como SSH gateway (jump host) -> localhost:5901.
#
# Uso:  tools/vnc.sh {start|stop|status|restart}
# Variables opcionales:  VNC_DISPLAY (def. :1)   VNC_GEOMETRY (def. 1600x900)
set -euo pipefail

DISPLAY_NUM="${VNC_DISPLAY:-:1}"
GEOMETRY="${VNC_GEOMETRY:-1600x900}"
PORT=$((5900 + ${DISPLAY_NUM#:}))

ensure_xstartup() {
    local f="$HOME/.vnc/xstartup"
    if [[ ! -f "$f" ]]; then
        mkdir -p "$HOME/.vnc"
        cat > "$f" <<'EOF'
#!/bin/sh
unset SESSION_MANAGER DBUS_SESSION_BUS_ADDRESS
exec fluxbox
EOF
        chmod +x "$f"
        echo "Creado $f (fluxbox)."
    fi
}

start() {
    ensure_xstartup
    vncserver "$DISPLAY_NUM" -geometry "$GEOMETRY" -localhost yes
    echo "VNC en $DISPLAY_NUM (loopback, puerto $PORT)."
    echo "MobaXterm: sesión VNC -> host 'localhost:$PORT'; Network settings -> SSH gateway = Brais."
}

stop()   { vncserver -kill "$DISPLAY_NUM"; }
status() { vncserver -list; }

case "${1:-}" in
    start)   start ;;
    stop)    stop ;;
    status)  status ;;
    restart) stop || true; start ;;
    *) echo "Uso: $0 {start|stop|status|restart}" >&2; exit 2 ;;
esac
