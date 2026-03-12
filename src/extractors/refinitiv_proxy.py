"""
Mini-proxy local que reenvía peticiones a Workspace en Windows
reescribiendo el header Host a 'localhost:9000'.
Necesario porque Workspace solo acepta conexiones con Host=localhost.
Ejecutar en background antes de usar refinitiv-data.
"""

import http.server
import urllib.request
import sys

WINDOWS_HOST = "172.23.160.1"
WINDOWS_PORT = 9001
LOCAL_PORT = 9000


class ProxyHandler(http.server.BaseHTTPRequestHandler):
    """Reenvía cada petición a Windows cambiando el header Host."""

    def do_request(self):
        url = f"http://{WINDOWS_HOST}:{WINDOWS_PORT}{self.path}"

        # Leer body si existe
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else None

        # Construir petición con Host=localhost
        req = urllib.request.Request(url, data=body, method=self.command)
        for key, val in self.headers.items():
            if key.lower() == "host":
                req.add_header("Host", "localhost:9000")
            else:
                req.add_header(key, val)
        if "Host" not in self.headers:
            req.add_header("Host", "localhost:9000")

        try:
            resp = urllib.request.urlopen(req, timeout=30)
            self.send_response(resp.status)
            for key, val in resp.getheaders():
                if key.lower() not in ("transfer-encoding",):
                    self.send_header(key, val)
            self.end_headers()
            self.wfile.write(resp.read())
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            for key, val in e.headers.items():
                if key.lower() not in ("transfer-encoding",):
                    self.send_header(key, val)
            self.end_headers()
            self.wfile.write(e.read())
        except Exception as e:
            self.send_response(502)
            self.end_headers()
            self.wfile.write(str(e).encode())

    do_GET = do_request
    do_POST = do_request
    do_PUT = do_request
    do_DELETE = do_request

    def log_message(self, format, *args):
        """Solo mostrar errores, no cada petición."""
        if args and "200" not in str(args[0]):
            sys.stderr.write(f"  proxy: {args}\n")


if __name__ == "__main__":
    server = http.server.HTTPServer(("127.0.0.1", LOCAL_PORT), ProxyHandler)
    print(f"Proxy local escuchando en 127.0.0.1:{LOCAL_PORT}")
    print(f"Reenviando a {WINDOWS_HOST}:{WINDOWS_PORT} con Host=localhost:9000")
    print("Ctrl+C para detener")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nProxy detenido.")
