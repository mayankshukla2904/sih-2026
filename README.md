# Sahayak — ultra-low-resource wake word (SIH 26172)

Read **[GUIDE.md](GUIDE.md)** first. That is the one file that explains the
problem, the theory, the hardware, and every decision.

## ESP32-S3 (node, already flashed)

USB-C is enough to talk to the board. This is an **S3 DevKit**, not Nano.
Wire the INMP441 to header pins **15 / 16 / 17** (SCK / WS / SD). 3.3 V only. GPIO 4/5/6 stayed silent on this DevKit.

```bash
source .venv/bin/activate
python -m tools.s3_monitor
# speak: speech= should flip 0 → 1
# reflash: cd firmware && ../.venv/bin/pio run -t upload
```

## Laptop (today, no Pi)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m tools.smoke_test
python -m tools.make_demo_wav
python -m server.asr_server --no-ui &
python -m node.main --source file --wav data/demo_speech.wav --detector stub
# Training clips: data/marvin/, data/sahayak/
# Speech Commands v2 unpack: data/_cache/speech_commands_v0.02/ (not the .tar.gz)
# Small Vosk model: models/vosk-model-small-en-us-0.15/
```

## Pi 4B (blank SD)

Image the card with SSH + Wi-Fi, then I take over. Full checklist:

**[docs/pi-first-boot.md](docs/pi-first-boot.md)**

After SSH works:

```bash
./tools/push_to_pi.sh pi@<pi-ip>
```

Then on the Pi: enable I2S (`docs/i2s-setup.md`), `pip install -r requirements.txt`,
start `server.asr_server`, start `node.main --source i2s`.

## Change the keyword

Edit `KEYWORD` in `shared/config.py`, then:

```bash
pip install -r requirements-train.txt
python -m training.ingest_speech_commands
python -m training.train
```
