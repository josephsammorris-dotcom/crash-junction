# Crash Junction

A Burnout 2-style Crash mode for the browser, built with three.js and cannon-es.
Roll down off Mesa Ridge into a busy US desert crossroads, make one hit, and watch the pile-up grow:
semis jackknife, buses topple, cars roll, fires spread and fuel tankers explode.
Your score is the damage multiplied by the number of vehicles wrecked.

Pass-and-play for 1–4 players: three runs each, and your best run counts.

## How to play

| | Keyboard | Touch | Controller |
|---|---|---|---|
| Steer | ← → or A / D | ‹ › buttons | Left stick or d-pad |
| Boost (refills) | Space | Boost | RT or A |
| Brake | ↓ | Brake | LT or X |
| Finish / next | Enter | Finish button | Start or A |
| Replay the crash | Replay button | Replay button | X on the results screen |
| Mute | M | | |

## What's in it

- Sculpted, deformable vehicles that dent where they're hit, with burnt-out, gutted shells after fires
- Traffic with panicking drivers, skids, jackknifes, rollovers, spreading fires and exploding tankers
- Traffic keeps arriving after the crash: most queue or edge round on the sand, a few plough in
- A camera that circles the crash and zooms in on explosions, fires and rollovers
- A cinematic replay of every run (the simulation is deterministic, so replays are exact)
- Procedural 140 BPM trance soundtrack and crash sound effects (Web Audio, no audio files)
- Layouts for desktop, phones (portrait and landscape) and game controllers, with rumble

## Files

- `game.html`: the whole game in one file (loads three.js and cannon-es from a CDN, so it also runs on its own)
- `public/`: the hosted site, built from `game.html` with the libraries served from `public/vendor/`
- `tools/build_public.py`: rebuilds `public/index.html` after `game.html` changes
- `tools/build_standalone.py`: a single-file copy with all sounds embedded
- `wrangler.jsonc`: Cloudflare Workers config that serves `public/`

## Hosting

Cloudflare Workers serves `public/` as a static site and republishes on every push to `main`.

## Credits

three.js (MIT) and cannon-es (MIT); their licences are in `public/vendor/`.
Sound effects from Stunt Rally 3 and SuperTuxKart contributors (CC BY / CC BY-SA / CC0): see `public/audio/CREDITS.md`.
