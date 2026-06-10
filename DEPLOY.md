# Deploying to Hugging Face Spaces

The repo is Space-ready: the `README.md` YAML header configures a **Docker SDK**
Space, the `Dockerfile` builds a CPU-only image, and `serve/app.py` serves an
interactive upload page at `/` plus the JSON API at `/predict`.

The only thing not in git is the model weight (`checkpoints/combined_final.pt`,
~32 MB) — it's committed to the Space via **Git LFS**.

## One-time setup

1. **Create a Hugging Face account** → https://huggingface.co/join
2. **Create a Space**: https://huggingface.co/new-space
   - Owner: you · Space name: `ant-royalty-detector`
   - **SDK: Docker** → *Blank* template
   - Visibility: Public
3. **Create a write token**: https://huggingface.co/settings/tokens → *New token*
   → type **Write**. Copy it.

Then come back here and give Claude the Space URL — Claude runs the push.

## What the push does (Claude handles this)

```bash
# add the Space as a remote (uses your username/space name)
git remote add space https://huggingface.co/spaces/<user>/ant-royalty-detector

# track the model with LFS and force-add it past .gitignore
git lfs install
git lfs track "*.pt"
git add .gitattributes
git add -f checkpoints/combined_final.pt
git commit -m "Add model weights for Space (LFS)"

# push (prompts for HF username + the write token as password)
git push space main
```

HF then builds the Docker image and boots the Space. First build takes a few
minutes (installing torch). When it's live:

- **Demo UI:** `https://<user>-ant-royalty-detector.hf.space/`
- **API:** `POST https://<user>-ant-royalty-detector.hf.space/predict` (multipart image)
- **Health:** `/health` · **Metrics:** `/metrics` · **Swagger:** `/docs`

## Notes

- The image is CPU-only (the Dockerfile installs torch from the CPU wheel index)
  so it fits the free Space tier. Inference is ~50–180 ms/image.
- To update the model later: replace `checkpoints/combined_final.pt`, commit,
  `git push space main`. The Space rebuilds automatically.
- `data/` is gitignored everywhere — only code + the one model weight ship.
