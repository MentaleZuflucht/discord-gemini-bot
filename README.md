# discord-gemini-bot

A Discord bot with one slash command, `/prompt`. It sends your message to the best free Gemini model your API key can use and replies with the answer.

- Picks the model from a ranked list of free-tier Gemini models and checks which ones your key actually has.
- If a model hits its quota, it moves on to the next one. With several API keys, it rotates between them.
- Prompts are never logged or stored.

The default personality is a sarcastic roast bot. To change it, edit `SYSTEM_INSTRUCTION` in `gemini_client.py`.

## Discord setup

1. Create an application in the [Discord Developer Portal](https://discord.com/developers/applications).
2. Under **Bot**, copy the token. No privileged intents are needed, so leave them all off.
3. Under **OAuth2 → URL Generator**, select the `bot` and `applications.commands` scopes. Leave all bot permissions unchecked.
4. Open the generated URL to add the bot to your server.

The bot needs no intents and no permissions. It never reads messages, and slash command replies are sent as interaction responses, which don't depend on the bot's channel permissions.

## Configuration

Copy `.env.example` to `.env` and fill it in:

```env
DISCORD_TOKEN=your-discord-bot-token

# One key, or several separated by commas. They are rotated between requests.
GEMINI_API_KEYS=first-gemini-api-key,second-gemini-api-key

# Optional. Set this while testing so /prompt shows up immediately in one server.
# Leave it empty for a global command (can take up to an hour to appear).
DISCORD_GUILD_ID=

# Optional. DEBUG, INFO, WARNING or ERROR. Defaults to INFO.
LOG_LEVEL=INFO
```

Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey).

## Running

### Docker

```sh
docker compose up -d
```

This pulls the prebuilt image from GHCR. Logs are kept in the `logs` volume.

### Windows

Run `run.bat`. It creates a virtual environment and installs the requirements on first run, then starts the bot.

### Manually

Requires Python 3.12 or newer.

```sh
python -m venv .venv
.venv/bin/pip install -r requirements.txt   # Windows: .venv\Scripts\pip
.venv/bin/python bot.py                     # Windows: .venv\Scripts\python
```

## Usage

```
/prompt message: why is my code not working
```

The reply shows the answer, your prompt in small text underneath, and the model that answered.
