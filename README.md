# Setup

```bash
# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# clone repos
git clone https://github.com/willccbb/gum-prime.git
git clone https://github.com/willccbb/prime-rl
ln -s ~/gum-prime/configs/haiku ~/prime-rl/configs/

# install dependencies
cd prime-rl
uv sync && uv sync --all-extras
```

# Run

```bash
uv run rl --trainer @ configs/haiku/train.toml --orchestrator @ configs/haiku/orch.toml --inference @ configs/haiku/infer.toml
```
