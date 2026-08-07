[← Remote and hosted engines](10-remote-and-providers.md) · [Index](README.md) · Next: [Advanced workflows →](12-advanced-workflows.md)

# 11. The arcade

Waiting on a long generation is the one moment a coding agent has nothing for you to do. So there are five games behind slash commands, and they are meant to be played **during** a turn: type one while the model is streaming and it opens as a translucent layer over the live output, which keeps scrolling underneath.

| Command | |
|---|---|
| `/pelota` | pong against a five-level CPU |
| `/breakout` | knock the wall down, five walls deep |
| `/invaders` | hold the line against the marching fleet |
| `/centipede` | shoot it apart before it walks into you |
| `/frogger` | cross the road, then ride the river home |

## Arguments

Two, combinable — `/breakout new sound`:

- **`new`** (or `reset`) — deal a fresh game. Without it, a command **resumes the game you left**. Each game keeps its own slot, so you can close pelota, run three turns, open breakout, and come back to pelota exactly where it was. A finished game is not kept, so the next command deals a new one.
- **`sound`** — blips on. Off by default; `b` toggles it in-game.

## Controls

Arrows, or `hjkl`, or `wasd`. Space serves or fires. `p` pauses, `t` switches between the translucent and opaque layer, `b` toggles sound, `Esc` or `q` leaves.

The mouse works everywhere: the wheel and trackpad steer, click and drag place the paddle or the ship, and clicking fires in the two shooters.

While a game is up, the **first `Ctrl-C` closes it and a second interrupts the model**, so you are never locked out of stopping a turn.

Pelota has one extra move: hold **Shift** while steering and the paddle shrinks to a third of its length, but a hit that lands leaves at triple speed — usually past the CPU. The boost lasts exactly one crossing.

## Two honest notes

**Translucency is not alpha.** A terminal cell holds one character and one pair of colors; there is nothing to composite. What happens instead is that these games are sparse, so the layer underneath is dimmed rather than erased and the glyphs land in the gaps. It reads as a veil and the model's output stays legible behind it, but it is a trick, not blending.

**Sound is the terminal bell**, and nothing else. That is deliberate: it adds zero bytes to the binary, where real audio would mean a synthesis crate and a system audio dependency. The cost is that `BEL` has no pitch and no length, so the only thing distinguishing one cue from another is how many — one blip for a hit, two for a life lost, three for a level. Terminals set to a visual bell flash instead, which is why it is off unless you ask.

## The screensaver

Leave the prompt idle and plank puts an ambient screen up. `ui.screensaverFace` picks which:

| Value | |
|---|---|
| `matrix` | the falling glyphs — the default |
| `starfield` | a perspective starfield rushing past the edges |
| `random` | a fresh coin flip each time it opens |

`ui.screensaver` says when: `1m` (the default), `2m`, `5m`, or `never`. Both cycle in `/config` rather than needing to be typed.

Any key or mouse event brings the UI back, and the event that wakes it is swallowed rather than acted on — waking a screensaver should not leave a stray character in your prompt or click a button you could not see. It never comes up mid-turn or over a dialog.

The screensaver is **not** behind `ui.easterEggs`. Turning the games off still leaves you an idle screen, because a game you invoke and what an unattended terminal shows are different decisions. Set `ui.screensaver` to `never` if you want neither.

## Turning them off

They live behind `ui.easterEggs`, on by default. Setting it to `false` does more than hide them: they stop being commands at all, so `/pelota` goes to the model as an ordinary prompt exactly like any other unrecognized slash line. That is the honest behaviour for a shared or managed install that wants no games in it, and the startup line names the setting when it is off — so a `settings.json` cannot quietly remove them without saying so.

None of them appear in `/help` or the completion popup. That is the point.

---

Next: [Advanced workflows →](12-advanced-workflows.md)
