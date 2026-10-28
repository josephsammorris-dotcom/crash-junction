"""Build public/index.html from the single-file game source (game.html).

game.html loads three.js and cannon-es from a CDN so it also runs as a standalone file.
For hosting, this swaps the import map to the copies in public/vendor/ and wraps the page
in a full HTML document (doctype, viewport, icons) so phones treat it as a proper web app.

    python3 tools/build_public.py
"""
import re, pathlib
root = pathlib.Path(__file__).resolve().parent.parent
src = (root / 'game.html').read_text()
importmap = ('<script type="importmap">{"imports":{"three":"./vendor/three/build/three.module.js",'
             '"three/addons/":"./vendor/three/examples/jsm/","cannon-es":"./vendor/cannon-es.js"}}</script>')
src, n = re.subn(r'<script type="importmap">.*?</script>', importmap, src, count=1, flags=re.S)
assert n == 1, 'import map not found'
title = re.search(r'<title>.*?</title>', src).group(0)
src = src.replace(title, '', 1)
icon = ("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='12' fill='%231A1D25'/%3E"
        "%3Cpath d='M10 44 L54 44' stroke='%23FFD23F' stroke-width='6'/%3E%3Cpath d='M20 34 L32 14 L44 34 Z' fill='%23E8463A'/%3E%3C/svg%3E")
head = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover, user-scalable=no">
<meta name="theme-color" content="#1A1D25">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="description" content="Burnout 2-style crash junction: one hit, total carnage. Pass-and-play for 1-4 players.">
<link rel="icon" href="{icon}">
{title}
'''
body_start = src.index('<canvas id="scene"')
out = head + src[:body_start] + '</head>\n<body>\n' + src[body_start:] + '\n</body>\n</html>\n'
(root / 'public' / 'index.html').write_text(out)
print('wrote public/index.html', len(out), 'bytes')
