# Lichess bot

Plays the RL model (`reinforcement-learning/chess_rl_model_v5.pt`) on Lichess. The bot uses
[lichess-bot](https://github.com/lichess-bot-devs/lichess-bot) to connect to Lichess, and
`reinforcement-learning/uci_engine.py` exposes the policy network and MCTS as a UCI engine.

```
run_bot.sh ──► lichess-bot (Lichess API) ──► engine.sh ──► uci_engine.py ──► fast_mcts + PolicyNetwork (GPU)
```

## One-time setup

1. **Create a new Lichess account for the bot.** It must have no games played, because only unused
   accounts can be upgraded to BOT.
2. **Create an API token** while logged in as the bot:
   <https://lichess.org/account/oauth/token/create?scopes[]=bot:play&description=lichess-bot>
   (keep the `bot:play` scope).
3. **Save the token** (this file is gitignored):
   ```bash
   echo 'lip_xxxxxxxxxxxxxxxx' > chess/lichess_bot/.token
   ```
4. **Upgrade the account to BOT and start the bot.** You only need `-u` once. The upgrade is
   permanent: the account can never play as a human again.
   ```bash
   chess/lichess_bot/run_bot.sh -u
   ```

## Running

```bash
chess/lichess_bot/run_bot.sh
```

Leave it running. Open `https://lichess.org/@/<bot-name>` from your own account and click
**Challenge to a game**. Bot accounts can't join the lobby or play correspondence-length clocks
over 3 hours. The bot accepts standard chess at bullet, blitz, rapid and classical speeds, plays
one game at a time, and saves PGNs to `chess/lichess_bot/game_records/`.

To have the bot challenge other bots on its own, set `matchmaking.allow_matchmaking: true` in
`config.yml`.

## Tuning

| Where | Setting | Effect |
|---|---|---|
| `config.yml` → `uci_options` | `Nodes` | Max MCTS simulations per move (0 = limited by the clock). Use this to make the bot weaker or more consistent. |
| | `Batch` | Leaves per GPU call. 8 gives ~1300 sims/s on an RTX 2060; 1 gives ~230. |
| | `Move Overhead` | ms reserved for network lag. Raise it if the bot loses on time. |
| `config.yml` → `challenge` | `time_controls`, `max_base`, `concurrency`, `modes` | Which challenges the bot accepts. |
| env | `MODEL_PATH=/path/to/ckpt.pt` | Play a different checkpoint (e.g. a newer `checkpoints_v5/ckpt_*.pt`). |

## Testing locally without Lichess

`engine.sh` is a standard UCI engine, so you can also load it in a GUI (Cute Chess, Arena,
En Croissant) or test it by hand:

```bash
printf 'uci\nisready\nposition startpos moves e2e4\ngo movetime 2000\nquit\n' | chess/lichess_bot/engine.sh
```
