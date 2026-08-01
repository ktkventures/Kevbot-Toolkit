"""TCP relay counting real wire bytes (incl. TLS framing) per direction.
Billed egress = server->client bytes on the wire."""
import socket, threading, time

class Relay:
    def __init__(self, upstream_host, upstream_port):
        self.uh, self.up = upstream_host, upstream_port
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind(("127.0.0.1", 0)); self.srv.listen(16)
        self.port = self.srv.getsockname()[1]
        self.rx = self.tx = self.conns = self.active = 0
        self._lock = threading.Lock()
        threading.Thread(target=self._accept_loop, daemon=True).start()

    def reset(self):
        with self._lock:
            self.rx = self.tx = self.conns = 0

    def wait_idle(self, timeout=10.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self._lock:
                if self.active == 0:
                    return True
            time.sleep(0.01)
        return False

    def _accept_loop(self):
        while True:
            try:
                c, _ = self.srv.accept()
            except OSError:
                return
            with self._lock:
                self.conns += 1; self.active += 1
            threading.Thread(target=self._handle, args=(c,), daemon=True).start()

    def _handle(self, c):
        try:
            u = socket.create_connection((self.uh, self.up), timeout=30)
        except Exception:
            with self._lock: self.active -= 1
            c.close(); return
        def pump(src, dst, which):
            try:
                while True:
                    b = src.recv(65536)
                    if not b: break
                    dst.sendall(b)
                    with self._lock:          # incremental: no end-of-conn race
                        if which == "rx": self.rx += len(b)
                        else: self.tx += len(b)
            except Exception:
                pass
            finally:
                try: dst.shutdown(socket.SHUT_WR)
                except Exception: pass
        t1 = threading.Thread(target=pump, args=(u, c, "rx"), daemon=True)
        t2 = threading.Thread(target=pump, args=(c, u, "tx"), daemon=True)
        t1.start(); t2.start(); t1.join(); t2.join()
        for s in (c, u):
            try: s.close()
            except Exception: pass
        with self._lock: self.active -= 1
