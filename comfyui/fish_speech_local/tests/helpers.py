import io, json, os, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch
import numpy as np
import soundfile as sf

def wav_bytes(rate=22050):
    data=io.BytesIO();t=np.arange(int(rate*.2))/rate
    sf.write(data,.1*np.sin(2*np.pi*440*t),rate,format="WAV",subtype="PCM_16")
    return data.getvalue()


def response_audio(audio_format='wav', streaming=False):
    if audio_format == 'wav':
        data = wav_bytes()
        return data[:4]+b'\xff'*4+data[8:40]+b'\xff'*4+data[44:] if streaming else data
    rate = 44100 if audio_format in ('mp3', 'pcm') else 48000
    samples = .1*np.sin(2*np.pi*440*np.arange(int(rate*.2))/rate)
    if audio_format == 'pcm':
        return (samples*32768).astype('<i2').tobytes()
    data = io.BytesIO()
    sf.write(data, samples, rate, format='MP3' if audio_format == 'mp3' else 'OGG',
             subtype='MPEG_LAYER_III' if audio_format == 'mp3' else 'OPUS')
    return data.getvalue()

class MockFish:
    def __init__(self):
        self.requests = []
        self.active = 0
        self.max_active = 0
        self.error = False
        self.bad_audio = False
        self.fail_texts = set()
        self.guard = threading.Lock()

    def __enter__(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.send_response(200); self.end_headers()
                self.wfile.write(wav_bytes() if self.path == '/reference.wav' else b'{"status":"ok"}')

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                with owner.guard:
                    owner.requests.append((self.path, payload, dict(self.headers)))
                    owner.active += 1
                    owner.max_active = max(owner.max_active, owner.active)
                try:
                    time.sleep(0.04)
                    error = owner.error or payload.get('text') in owner.fail_texts
                    self.send_response(500 if error else 200)
                    self.end_headers()
                    self.wfile.write(b'{"error":"test failure"}' if error else
                                     (b'{"not":"audio"}' if owner.bad_audio else
                                      response_audio(payload.get('format', 'wav'), payload.get('streaming', False))))
                finally:
                    with owner.guard:
                        owner.active -= 1

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.env = patch.dict(os.environ, {'FISH_API_URL': f'http://127.0.0.1:{self.server.server_port}',
                                          'FISH_API_KEY': '', 'FISH_REQUEST_TIMEOUT': '10'})
        self.env.start()
        return self

    def __exit__(self, *args):
        self.server.shutdown(); self.server.server_close(); self.thread.join(timeout=3); self.env.stop()
