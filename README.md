**English** | [日本語](README.ja.md)

# llm-imaginary-friend

A chat app where you talk with five imaginary friends. Each girl was born from a different side of your heart, and the world feels like a cozy slice-of-life four-panel comic.
They all think differently. They don't just answer you: they laugh, bicker and make up with each other.
They live in a town inside your heart (Imaginary Land), and remember what happened while you were away and what you talked about last time.
A sixth girl, Io, is the language model itself. From the power room at the edge of town, she talks about how the model she runs on is doing.

This is a personal hobby project. I don't respond to questions or requests.
However, reports of rights/licensing concerns or problems are welcome via GitHub Issues.

Everything is written in Python. The 3D display uses the [Ursina](https://www.ursinaengine.org/) engine, and the characters and their motion are reused from
[llm-radio-daemon](https://github.com/lumavrenor/llm-radio-daemon), a 24-hour radio I made earlier (models built in code; no external model files).
The LLM, the voice and the search all run on your own PC plus public APIs that need no key, so there are no usage charges.

The girls speak Japanese by default. An English edition (`go.bat en`) has its own prompts, voices (Kokoro TTS) and database; see [English edition](#english-edition).

## Demo video

[![Demo video (YouTube)](https://img.youtube.com/vi/CM0_ckjqNqc/maxresdefault.jpg)](https://youtu.be/CM0_ckjqNqc)

https://youtu.be/CM0_ckjqNqc (Japanese edition, Japanese audio)

## What this project is (and isn't)

This is not practical software. It's an experimental project for a programmer to play with on their own machine.
The point is to see how "alive" six characters with different personalities can feel using nothing but a local LLM on your own PC.
The hand-raising, speaker selection, feelings, affinity, town events and personal memories are all there to keep smaller models from falling apart.

## About what the LLM generates (important)

Everything the girls say, the town events and their memory notes are fiction, generated on the fly by a local LLM.
They have nothing to do with real people or organizations, and none of it is checked by an expert.

- The girls sometimes talk based on web search, news and weather results, and can get facts wrong in a perfectly confident tone while summarizing or answering
- There is no NG-word filter that checks lines before they are spoken. The risk that the LLM outputs something inappropriate cannot be brought to zero (this is a limit of current generative AI, not a gap in the implementation)
- It is shaped like a friend to talk to, but it is not a substitute for a helpline or counseling. **Don't treat anything related to medical, legal, financial or safety matters as real advice and act on it**
- The author is not liable for any damage arising from this software's output

## Requirements

- Windows 11 (`go.sh` is written for Linux/macOS too, but the author has only tested Windows)
- Python 3.11 or newer (tested with 3.14)
- An OpenAI-compatible LLM server (SGLang / vLLM / Ollama, etc.). Tested with gemma-4-31B (NVFP4) and qwen3.8:27b.
  Six hand-raise requests go out in parallel, so a server that handles parallel requests well (SGLang / vLLM) answers faster.
  See [LLM server (SGLang) setup](#llm-server-sglang-setup)
- [VOICEVOX](https://voicevox.hiroshiba.jp/) (voices for the Japanese edition; not needed for the English edition)
- An internet connection for search, news and weather (none of them need a key; the app still runs with them off)

## Features

- Display: Ursina (five 3D characters plus the sixth, a conversation log and an input box)
- LLM: any OpenAI-compatible server (SGLang / vLLM / Ollama)
- Voice: VOICEVOX ENGINE (Japanese edition) / Kokoro-82M on the CPU (English edition)
- Search: Wikipedia and Google News RSS search (default; no keys needed) / Brave Search API / SearXNG. Used through tool calling by the girl who answers you first, and by the first speaker of each round while the girls keep talking
- News: RSS feeds (NHK / Yahoo! / GIGAZINE etc. for Japanese; BBC / NPR / Ars Technica for English) are read in the background, and the girls sometimes bring up a story they "just saw online"
- Weather: Open-Meteo (no key). The current sky goes into the conversation's situation, and a girl sometimes mentions "I just checked the forecast"

## Getting started

1. Start your LLM server (for SGLang, `127.0.0.1:30000`) and, for the Japanese edition, VOICEVOX (`127.0.0.1:50021`).
   For how to start SGLang, see [LLM server (SGLang) setup](#llm-server-sglang-setup)
2. Run `go.bat` for the Japanese edition, or `go.bat en` for the English edition (Git Bash / Linux / macOS: `./go.sh` / `./go.sh en`)

The first run creates `.venv` and installs the dependencies. After that it starts in a few seconds.

Type in the input box and press Enter. Scroll the log with the mouse wheel.
The input box edits like an ordinary text field:

- Selection: Shift+arrows / Shift+Home/End / Ctrl+A / mouse drag / Shift+click / double-click (word) / triple-click (all)
- Words: Ctrl+arrows to move, Ctrl+Backspace / Ctrl+Delete to delete
- Ctrl+C / Ctrl+X / Ctrl+V (or Shift+Insert). With nothing selected, copy and cut take the whole text. Pasted text with line breaks becomes one line
- Ctrl+Z to undo, Ctrl+Y / Ctrl+Shift+Z to redo

You can copy messages from the conversation log too:

- Click to select one. Drag or Shift+click to select several in a row
- Ctrl+C copies them (one message: just the text; several: one `name: text` per line). Right-click copies that one message right away
- Esc, or a click on an empty part of the log, clears the selection

| Option | Meaning |
|---|---|
| `--cli` | No window; chat over standard input/output (for testing) |
| `--no-audio` | No sound (the log and lip sync still run) |
| `--no-idle` | Turn off the girls' idle chatter and the greeting at startup |
| `--no-search` | Don't use the web search tool |
| `--no-news` | The girls don't bring up news on their own |
| `--no-weather` | Don't fetch the weather (they never mention it) |
| `--voice-test DIR` | Don't connect to the LLM; synthesize each girl's example lines into WAV files in `DIR` and exit |
| `--config PATH` | Settings file (default `config.toml`) |

`go.bat` and `go.bat en` pass the remaining arguments through, e.g. `go.bat en --no-idle`.

### English edition

`go.bat en` (bash: `./go.sh en`) starts the edition where the girls speak English.

- Settings are in `config.en.toml` (`[locale] lang = "en"`). Its database is `data/imaginary_friend_en.db`, so it never mixes with the Japanese conversations and memories
- Instead of VOICEVOX, it runs Kokoro-82M (Apache-2.0) on the CPU with onnxruntime. No GPU or VOICEVOX needed for the voice
- The first run also installs `requirements-en.txt` and the spaCy model `en_core_web_sm`, and fetches the Kokoro model files (about 340 MB) into `kokoro_models/`
- The six girls look the same as in the Japanese edition. Their personalities, speech styles, example lines and filler lines were rewritten in English, and their voices are blends of Kokoro's English voices
- Name pronunciations are pinned in `lexicon.en.txt` (Nene is "Neh-neh", Io is "Ee-oh"). Words missing from the dictionary get an estimated reading and are logged once as a WARNING
- The prompts keep the Japanese originals and add English versions next to them; `[locale] lang` picks one at startup (`pick(ja=..., en=...)` in `imaginary_friend/language.py`). The Japanese text was not changed
- In the English edition the feeling tags are `[happy]` `[cheerful]` `[angry]` `[sad]` `[surprised]` `[shy]` `[neutral]`, and closeness of memories and town events is measured by shared words instead of shared character pairs

#### Voice tuning

Kokoro has 28 English voices; the women's voices are
`af_alloy af_aoede af_bella af_heart af_jessica af_kore af_nicole af_nova af_river af_sarah af_sky` (US) and
`bf_alice bf_emma bf_isabella bf_lily` (UK). Unknown names stop the app at startup with the list of valid names.

```
.venv\Scripts\python -m imaginary_friend.main --config config.en.toml --voice-test out
```

writes each girl's example lines (and each `kokoro_emotions` voice) to `out/*.wav` and prints the synthesis speed. The LLM isn't used.
On an 8-thread desktop CPU with `threads = 2`, one second of speech takes about 0.7 to 0.9 seconds to synthesize, which keeps up with playback.

### Resetting the database

`reset.bat` (bash: `./reset.sh`) deletes the database holding the conversation log, memories, moods and relationships, after listing what it will delete.
Close the app first. An empty database is created on the next start.

| Option | Meaning |
|---|---|
| (none) | Delete only the database at `[store] path` in `config.toml` |
| `--all` | Delete every `*.db` in `data/` (old databases too) |
| `--config PATH` | Read `[store] path` from this settings file (`--config config.en.toml` for the English edition) |
| `-y` | Delete without asking |

## LLM server (SGLang) setup

**We recommend letting a coding agent (Claude Code, Codex, etc.) do this setup for you.**
SGLang's arguments and supported models change often between versions, and the right settings (and the places it breaks) depend on your number of GPUs, VRAM and the model's quantization.
The quickest route is to give an agent this section plus your GPUs, the model you want to use and your SGLang version, and have it write a launch script.
The example below is from the author's machine and may not work as-is; use it as a starting point to hand to the agent.

If you just want to try the app with little effort, Ollama works too (`[llm] profile = "ollama"`). Replies are slower because it handles parallel requests less well.

### What this app needs from the server

- An OpenAI-compatible API (`/v1/chat/completions`) listening at `[llm.sglang] host` in `config.toml` (default `http://127.0.0.1:30000`).
  `--served-model-name` can be anything (if `model` is omitted, the first entry of `/v1/models` is used). Io, the sixth girl, reads the real model name from the weights path in `/get_model_info`, so an alias is fine
- `--context-length` of 8192 or more. `[talk] history_chars = 3000` is tuned to that length; lower `history_chars` too if you go shorter
- Each round sends six hand-raise requests in parallel, followed by the speakers' replies, look-ahead and memory-note summaries. `--max-running-requests` around 8 is enough
- Web search needs `--tool-call-parser` (`gemma4` for Gemma 4, `qwen3_coder` for qwen3.8). For models without one, start the app with `--no-search`
- For thinking models, split off the reasoning with `--reasoning-parser` (`gemma4` for Gemma 4)
- Hand-raising uses structured output with a JSON schema. `--constrained-json-disable-any-whitespace` prevents a failure where the model keeps emitting whitespace and never stops
- Keep the radix cache on (the start of every prompt is shared by all the girls, so prefill can be skipped)

### Example launch on the author's machine

Three GPUs (pipeline parallel), Gemma 4 31B quantized to NVFP4 ([RedHatAI/gemma-4-31B-it-NVFP4](https://huggingface.co/RedHatAI/gemma-4-31B-it-NVFP4)), SGLang 0.5.21, Ubuntu on WSL2.

```bash
export CUDA_VISIBLE_DEVICES=1,2,0              # the GPU driving the display goes last
export SGLANG_PP_LAYER_PARTITION=17,26,17      # split of the 60 layers (first stage holds embeddings, last holds lm_head, so they get fewer)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

python -m sglang.launch_server \
  --model-path /path/to/gemma-4-31B-it-NVFP4 --served-model-name npc \
  --tp-size 1 --pp-size 3 \
  --context-length 8192 --mem-fraction-static 0.93 --kv-cache-dtype bfloat16 \
  --max-running-requests 8 --max-queued-requests 32 \
  --cuda-graph-max-bs-decode 8 --disable-prefill-cuda-graph --enable-cache-report \
  --reasoning-parser gemma4 --tool-call-parser gemma4 \
  --constrained-json-disable-any-whitespace --skip-server-warmup \
  --host 127.0.0.1 --port 30000
```

With this setup, one request runs at about 17 tok/s, and five at once at about 9 to 12 tok/s each. If the model fits on one GPU, drop `--pp-size` and `SGLANG_PP_LAYER_PARTITION`.

Things that tripped up the author (worth telling your agent):

- `--kv-cache-dtype bfloat16`: Gemma 4 runs on triton attention and crashes with an FP8 KV cache
- `--skip-server-warmup`: in the RedHatAI build the image encoder is FP4 too, and on SGLang 0.5.21 a single request containing an image makes every later reply come back as `<pad>` (the server must be restarted).
  This app never sends images, but the startup warmup includes one, so it is skipped
- Gemma 4's KV cache is heavy (about 1.1 GB per request at 8k context for the full-attention layers). NVIDIA's official NVFP4 build keeps attention in BF16 and is more accurate, but its weights are larger, and even on three GPUs there wasn't room for the KV of a single 8k request
- SGLang 0.5.21 has a bug where each pipeline stage allocates KV for all layers, and bugs loading compressed-tensors NVFP4 (the RedHatAI build).
  The author runs with local patches, which are not included in this repository. If you use the same setup, describe the symptoms to your agent and have it fix them

## Settings (config.toml / config.en.toml)

`config.toml` is the Japanese edition and `config.en.toml` the English edition. Each key is commented in the files.

- `[llm] profile = "auto"` tries the `[llm.<name>]` entries in the order written and uses the first server that answers. If `model` is omitted, the first entry of `/v1/models` is used
- `[user] name` is what the girls call you. Leaving it as "you" makes it collide with "you are ..." in the character prompts, and the girls start calling each other "you" too, so use a name
- `[[cast]]` defines the five girls (personality `desc`, example lines `examples`, look and voice). `examples` set the tone, and help smaller models most.
  - Japanese edition: `voicevox_emotions = { "哀" = "泣き" }` assigns a VOICEVOX style to a feeling (feelings not listed are expressed through pitch, intonation and speed)
  - English edition: `kokoro_voice` is one Kokoro voice (`"af_heart"`) or a weighted blend:
    `[{ voice = "af_sky", weight = 0.6 }, { voice = "af_heart", weight = 0.4 }]`. `kokoro_speed` sets her speaking speed.
    `kokoro_emotions = { sad = [...] }` gives a feeling its own blend (feelings not listed only change the speed, via `[kokoro] emotion_prosody`).
    `[kokoro]` sets the model folder, CPU threads (`threads`), overall speed and volume.
    `lexicon.en.txt` pins pronunciations (one `word<TAB>IPA` per line); fix a WARNING's estimate and paste it there
- `[affinity]` is how much the girls usually like each other. A girl is more likely to talk back to someone she likes less. Feelings in conversation move today's values up and down
- `[drama]` turns feelings, today's mood and the staged beats on or off, and sets how often beats are staged (see "Feelings and drama" below)
- `[meta]` / `[meta.cast]` is the sixth girl (see "The sixth girl" below). Set `enabled = false` for comparison tests
- `[store] path` is the database for the conversation log, memories, moods and relationships
- `[talk]` sets how many girls answer, the length of one line (`max_chars` / `max_sentences`), how much of the log the LLM sees (`history_lines`), how long the girls keep talking (`session_min_sec` / `session_max_sec`) and the idle chatter interval
- `[search]` turns search on or off and picks backends (`backend`; with several, they run in parallel and are passed as "encyclopedia" and "news"), how many times the girls may look things up on their own while talking (`max_per_session`), and background lookups by girls who aren't speaking (`background` etc.). `[[cast]] search_line` is the filler line spoken when a girl starts a search without saying anything
- `[news]` sets the RSS feeds the girls bring news from (`feeds`), the chances of bringing it up (greeting at startup `start_chance`, idle chatter `idle_chance`, late in a conversation `talk_chance`) and words to avoid (`skip_words`)
- `[weather]` sets the forecast location (`location_name` / `latitude` / `longitude`; look them up at https://open-meteo.com/en/docs), whether the current sky goes into the situation (`in_situation`), the chances of bringing it up (`start_chance` / `idle_chance` / `talk_chance`) and the minimum gap (`min_gap_min`). Leave it out and the girls never mention the weather. Temperatures are in Celsius
- `[memory]` turns personal memories on or off, and sets how many are recalled and how long a stretch of conversation each memory note covers
- `[display] ime_aware` shows text being composed by the IME inside the input box (false: in the OS's composition window)
- `[display] layout_jitter` sets how far the girls' positions vary. They stand on an arc around the viewer, and their positions and facing shift a little at random on each start (0 lines them up)
- `[display] font` must contain the glyphs for the language (for English, Segoe UI by default on Windows; on Linux/macOS point it at any .ttf)

## How it works

```
UI thread (Ursina)  <-- event queue --+-- brain thread (hand-raise -> speak -> synthesize)
  drawing, input, log                 +-- audio callback (playback-start notices)
```

One round (`imaginary_friend/brain.py`):

1. **Hand-raise phase**: each of the five girls is asked in parallel, as a short JSON, what she wants to say, her stance and how badly she wants to speak
2. The top girl speaks. Her reply is streamed, cut into sentences, synthesized ahead of time and queued on a single playback queue
3. The second speaker sees the first one's reply before speaking. Girls who don't get along with the first speaker are more likely to be picked

The girl who answers you first, and the first speaker of each round while the girls keep talking (up to `max_per_session` times), get the `web_search` tool.
When a girl calls it, the search runs while "let me look that up" plays, and she continues with the results.
The full search results stay only in the personal memory of the girl who searched. The others only know what she said out loud (the conversation log).

**Background lookups** (`[search] background`): the hand-raise JSON also asks what a girl would like to check (`check`). Girls other than the round's first speaker
search in the background while someone else is talking. It's one more field in the hand-raise question, so no extra LLM requests.
The results go only into that girl's memory, and when her turn comes she can say "I just looked it up" (waiting up to 5 seconds if it isn't done yet).
Up to `background_per_round` girls per round, and up to `background_per_session` times between one message of yours and the next.
The same query is never searched twice, and in rounds where the first speaker has the search tool, nobody looks up the same thing in the background.
A girl who has finished a lookup is more likely to be picked as the next speaker.
If a speaker with the search tool gave a `check` when raising her hand but didn't call the tool, the app fills in with "let me check" and searches for the `check` words itself
(small models tend to answer "who's that?" without searching, even when asked "do you know ...?").

**News** (`imaginary_friend/news.py`): the RSS feeds in `[news] feeds` are read regularly in the background. In the greeting at startup, in idle chatter and late in a conversation (once),
one girl sometimes brings up "a story I just saw online". When you ask "any news lately?", one girl (the one you named, if any) is given a story to answer with. The headline and summary go only into that girl's prompt and personal memory.
For articles with no description, the page's summary (meta description) is fetched just before the story is brought up. Stories already used are recorded in the database and never repeated, even after a restart.

**Weather** (`imaginary_friend/weather.py`): using the latitude and longitude in `[weather]`, the current weather, temperature and today's and tomorrow's forecast are fetched from Open-Meteo in the background (every 30 minutes).
The current sky is one line in the "current situation" shared by all the girls, and when you mention the weather, umbrellas, tomorrow and so on, the forecast is added too.
In the greeting at startup (tried before news), idle chatter and late in a conversation, one girl sometimes brings up "I just checked the forecast" (only she gets the forecast; from the evening on, it's mostly tomorrow's).
If the fetch failed or the data is more than 3 hours old, the girls don't mention the weather.

**Personal memories** (`imaginary_friend/memory.py`): each girl keeps two kinds of memory in the database.

- Things she looked up herself: search results boiled down to the key points. Only the girl who searched remembers them
- Memories of earlier conversations: at each break in the conversation (the end of a long chat, or your next message), each girl writes a short note from her own point of view.
  Facts she looked up are written in detail, and what others said only roughly, as she heard it, so each girl remembers the same conversation differently

When she speaks, her memories closest to the current topic (`[memory] facts_in_prompt` / `notes_in_prompt` of them) go into the character-specific part of her prompt.
Closeness is measured by shared two-character pairs containing kanji, katakana or alphanumerics in the Japanese edition, and by shared words in the English edition (no embedding model). Memories survive restarts.

After you say something, the girls keep talking among themselves for three to five minutes (`session_min_sec` to `session_max_sec`).
When you sign off ("good night", "see you", "I'm heading out"), they just say goodbye: no long chat, and no idle chatter until you speak again.
A new round is built each time the queued speech drops below `lookahead_sec` seconds, so the voices never stall and nothing is generated too far ahead.
To keep long chats from going in circles, the first speaker of each round is given one way to move things forward, such as "give a concrete example" or "push back".
When you speak, lines not yet played are dropped on the spot (only the sentence being played is finished), and the girls start answering you.
After the greeting at startup, while you stay quiet, a timer runs the same rounds (up to `max_auto_rounds` times; not after a long chat).

Prompts are ordered "shared part (setting, everyone's introductions, conversation log) -> character-specific part", so the parallel requests share the same prefix.

### The sixth girl (`imaginary_friend/meta.py`)

The connected LLM itself, talking from the power room at the edge of town (her name is Io).

- She's the same one girl whatever model she runs on. Her name, look and voice are written once in `[meta.cast]`, and her memory, affinity and mood all use the single id `llm`.
  The model name is taken from the server automatically (SGLang: the weights path from `/get_model_info`; vLLM: `root`; Ollama: the model name; otherwise `[llm.<name>] display_model`)
- At startup she is compared against the `meta_runs` table (a record of runs per model name), and "first time running", "last time it was a different model", "first time on this model" or "the last run was 3 days ago" become material for her
- Only her prompt gets the real run status (model name, temperature, how much of the conversation she can see, the last line's generation time and token speed, failure counts, search budget left, etc.) as "# current run status".
  It goes in only on rounds with a cue (generation was slow, it stalled, someone forgot something, you asked how things work, the model changed, searches ran out) and on rounds that hit `stats_chance`
- She's rarely picked normally (`appear_weight`) and more likely with a cue (`cue_weight`). She speaks at most `max_streak` rounds in a row
- The five girls never see the technical details (the shared introduction is only `desc`; what only she knows is `self_desc`). She doesn't appear in town events or news
- Her mood comes from the run status ("dazed" after a stall, "in good form" if faster than this model's last run, "restless" when the model changed)
- The name panel on screen shows the connected model's name under hers (`show_model_name`). Long names are shortened up to the parameter count (`gemma-4-31B-it-NVFP4-RedHat` -> `gemma4-31B`); `model_label` pins it.
  Her prompt gets the original name, with a note to say the short name aloud (so she doesn't read out a weights file name)

### Feelings and drama (`imaginary_friend/drama.py`)

- **Feeling tags**: each line starts with a feeling tag, 【喜】【楽】【怒】【哀】【驚】【照】【普通】 in Japanese or `[happy]` `[cheerful]` `[angry]` `[sad]` `[surprised]` `[shy]` `[neutral]` in English (and again on any sentence where the feeling changes).
  The tag isn't read aloud; it changes the voice and moves affinity and moods. Sentences without a tag keep the previous feeling (at first, the `emotion` from the hand-raise JSON)
- **Voice**: if `voicevox_emotions` / `kokoro_emotions` has a voice for the feeling, that voice is used (e.g. Ankomon's "よわよわ" style); otherwise pitch, intonation and speed change (`[voicevox] emotion_prosody` / `[kokoro] emotion_prosody`)
- **Expressions** (`display/expression.py`, off by default): the 3D models have no facial expressions. With `[display] expressions = true`, they get eyebrow angles, eye and mouth shapes, blushing, tears, a lowered head and comic symbols above the head.
  These last 14 seconds after her last line, then return to today's mood face (half-closed eyes for a sleepy girl, worried brows for a gloomy one).
  The model meshes aren't changed (only panels are added as children of the head), but it made them less cute, so it's off by default
- **Today's mood**: together with the town events (`town.py`), the LLM writes each girl's mood for the day. There is no fixed table.
  It stays the same across restarts on the same day (saved in the database). Until it's decided, and when `[town] enabled = false`, everyone is "normal"
- **Changing relationships**: a line's feeling and its target (the girl named, or else the previous speaker) move affinity and moods.
  If anger pushes affinity below the baseline, the two become "awkward" until they make up. Overnight, affinity goes halfway back to the baseline (the `state` table in the database)
- **Staging**: a long chat is treated as a four-part story. Around 30-50% (the "twist"), one of a quarrel, a secret coming out, a confession or a gag is staged;
  around 65-85% (the "resolution"), one of making up (an awkward pair takes priority), encouragement or a laugh.
  Confessions and secrets come from town events the girls haven't told you about yet. If a secret concerns a girl who wasn't there, she becomes the other party. Idle chatter also gets a beat with chance `idle_beat_chance`.
  The staging instructions go only into the prompts of the girls with roles (lead and partner), who then speak in that order. The others only know what they heard in the conversation

| File | Role |
|---|---|
| `imaginary_friend/brain.py` | Conversation flow (hand-raising, speaker selection, prompts) |
| `imaginary_friend/drama.py` | Feeling tags, today's mood and events, changes in affinity and mood, staging (quarrels, making up, etc.) |
| `imaginary_friend/meta.py` | The sixth girl (comparison at startup, run status, cues and pick weight, mood) |
| `imaginary_friend/llm.py` | OpenAI-compatible client (structured output / streaming / tool calls) and per-request records (`RunStats`) |
| `imaginary_friend/search.py` | Search tool definition and backends (Wikipedia / Google News / Brave / SearXNG) |
| `imaginary_friend/news.py` | Reading RSS and picking the stories the girls bring up |
| `imaginary_friend/voice.py` | VOICEVOX synthesis, playback queue, lip sync from mora lengths |
| `imaginary_friend/kokoro.py`, `g2p_fallback.py` | English speech synthesis (Kokoro-82M directly on onnxruntime; g2p_en for words outside the dictionary). Lip sync comes from loudness |
| `imaginary_friend/language.py` | Conversation language (`[locale] lang`); picks the Japanese or English prompts |
| `imaginary_friend/memory.py` | Each girl's personal memory (things she looked up, notes about conversations) and recalling memories close to the topic |
| `imaginary_friend/store.py` | SQLite storage for the shared log, personal memories, moods and relationships (`[store] path`). Delete or rename the file to start from an empty conversation |
| `imaginary_friend/display/app.py` | Screen |
| `imaginary_friend/display/textedit.py`, `clipboard.py` | Selection, word movement and undo in the input box; clipboard read/write |
| `imaginary_friend/display/expression.py` | Facial expressions (eyebrows, cheeks, tears, comic symbols). Added as children of the head without touching poly_character |
| `imaginary_friend/display/poly_character.py`, `models.py` | 3D character generation |

## Not there yet

- Embedding search for personal memories (for now it matches character pairs / words; worth considering once memories reach the thousands)

## Design assumptions

This software is designed for personal, non-commercial use at home, so the following are intentionally not implemented:

- Streaming output, or recording / saving audio to files (`--voice-test` only writes the example lines to WAV for voice tuning)
- Any write action driven by LLM output (file operations, sending mail, running code). The only tool the LLM gets is the read-only `web_search`;
  its output becomes the on-screen log and speech, and stays on your machine as the conversation log and memories in the database (`data/`)

### Notes on videos and live streams

You may show the software in operation (screen and audio) in videos or live streams to introduce, review or explain it,
whether or not the channel is monetized (an always-on stream whose main content is the software's output is not allowed; see LICENSE).

**Only conversations held with web search, news and weather turned off may be included.**
Start it with `go.bat --no-search --no-news --no-weather` (English edition: `go.bat en --no-search --no-news --no-weather`).
Search results and news headlines/summaries come from outside articles with their own terms, and the weather can hint at where you live.

- **VOICEVOX voices** — credit only the characters who actually spoke, in the form each one's terms require (`VOICEVOX:character name`), in the description or similar.
  Some characters have conditions on pairing their voice with AI-written lines, on publishing and monetization, and on what may be said.
  The LLM's lines can't be fully controlled, so **a girl may say something that character's terms forbid, in that character's voice.** Check the content before publishing
- **Screen contents** — the conversation log still holds earlier conversations (the name the girls call you, personal things you talked about).
  Delete the database with `reset.bat` (English edition: `reset.bat --config config.en.toml`) or point `[store] path` at a new file first, and keep the config files and local paths off screen
- **LLM output** — the lines are AI-generated fiction. If the platform requires disclosure of AI-generated content, follow it

Checking the terms of the materials the software uses, and clearing rights, is your responsibility. The author does not guarantee or speak for those terms,
and they can change, so check them as of when you use them.

## Credits

The Japanese edition's speech synthesis uses [VOICEVOX](https://voicevox.hiroshiba.jp/).
Each character has its own terms of use, so be sure to check them for every character you use.

This is a personal project unrelated to VOICEVOX or the official owners and operators of its characters.
The five girls and the sixth (`[[cast]]` / `[meta.cast]`) are original personas made for this software, not the VOICEVOX characters themselves (only their voices are borrowed).

The default voices in `config.toml` are (update this if you change `voicevox_speaker_name`):

```
VOICEVOX:春日部つむぎ
VOICEVOX:冥鳴ひまり
VOICEVOX:九州そら
VOICEVOX:四国めたん
VOICEVOX:あんこもん
VOICEVOX:ナースロボ＿タイプＴ
```

The English edition's speech synthesis uses [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (Apache-2.0) and
[misaki](https://github.com/hexgrad/misaki) (Apache-2.0); words missing from the dictionary are read with [g2p_en](https://github.com/Kyubyong/g2p) (Apache-2.0).
misaki's English G2P reads numbers with [num2words](https://github.com/savoirfairelinux/num2words) (LGPL-2.1); this project only imports it unmodified, but it is noted here to be safe.
The Kokoro model and voice files are not in the repository. They are downloaded into `kokoro_models/` on first run from the GitHub Releases of
[kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx), which distributes Kokoro-82M converted to ONNX (only `config.json` comes from the Kokoro-82M repository on Hugging Face).
See the official model card for the training data.

- 3D display: [Ursina](https://www.ursinaengine.org/) (MIT) / Panda3D (Modified BSD)
- Search: [Wikipedia](https://www.wikipedia.org/) (text under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)) and Google News RSS search. Results only go into the LLM's prompt and are not reproduced on screen
- News: publishers' RSS feeds (`[news] feeds`). Only headlines and summaries are passed to the LLM; articles are not reproduced
- Weather: [Open-Meteo](https://open-meteo.com/) (data under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); the free API is for non-commercial use)

## License

Copyright belongs to the author. You may run this software at home for personal, non-commercial enjoyment.
An individual may also show it in operation in videos or live streams to introduce, review or explain it, whether or not the channel is monetized,
but only conversations with web search, news and weather turned off (see "Notes on videos and live streams").
Publishing or distributing copies or modified versions, always-on streams whose main content is the software's output, and commercial use are not permitted.
See [LICENSE](LICENSE) for details.
The software comes with no warranty, and the author is not liable for any damage or trouble arising from its use.

The licenses and terms of the external services (LLM server, VOICEVOX, Wikipedia, Google News, the news RSS feeds, Open-Meteo) and of the libraries
listed in the credits must be checked separately.

If this README and LICENSE disagree, LICENSE takes precedence.
