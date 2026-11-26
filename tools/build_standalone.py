"""Standalone build: game.html with every sound in public/audio embedded (music re-encoded at 96 kbps to keep it
under 16 MB), for sharing as a single file.  Usage: python3 tools/build_standalone.py OUTPUT.html"""
import base64, pathlib, re
import sys
repo = pathlib.Path(__file__).resolve().parent.parent
src = (repo / 'game.html').read_text()
emb = {}
import subprocess
import tempfile; tmp = pathlib.Path(tempfile.mkdtemp())
for f in sorted((repo / 'public/audio').iterdir()):
    if f.suffix in ('.mp3', '.wav', '.ogg', '.m4a'):
        data = f.read_bytes()
        if f.stem in ('menu', 'drive', 'results'):  # lighter copies of the music so the page stays under the size limit
            o = tmp / (f.stem + '.mp3')
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(f), '-c:a', 'libmp3lame', '-b:a', '96k', '-ar', '44100', str(o)], check=True)
            data = o.read_bytes()
        emb[f.stem] = base64.b64encode(data).decode()
js = '<script>window.__hsxAudio={' + ','.join(f'"{k}":"{v}"' for k, v in emb.items()) + '};</script>\n'
# vehicle models are fetched from models/ next to the page; --embed-models puts them inside it (for offline tests)
models = {f.stem: base64.b64encode(f.read_bytes()).decode() for f in sorted((repo / 'public/models').glob('*.cjm'))} if '--embed-models' in sys.argv else {}
js += '<script>window.__hsxModels={' + ','.join(f'"{k}":"{v}"' for k, v in models.items()) + '};</script>\n'
i = src.index('<script type="importmap">')
out = src[:i] + js + src[i:]
args = [a for a in sys.argv[1:] if not a.startswith('--')]
dest = pathlib.Path(args[0] if args else 'crash-junction-standalone.html'); dest.write_text(out)
print('artifact build', len(out) // 1024, 'KB with', len(emb), 'sounds')
