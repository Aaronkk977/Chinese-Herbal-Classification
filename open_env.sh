#!/usr/bin/env bash
set -euo pipefail

# open_env.sh
# Create or attach to a tmux session and activate a conda environment inside it.
# Usage: ./open_env.sh [-e envname] [-s sessionname]

ENV_NAME="herbal"
SESSION_NAME=""

usage() {
  cat <<EOF
Usage: $0 [-e env] [-s session]
  -e env       conda environment name to activate (default: herbal)
  -s session   tmux session name (default: <env>_session)
EOF
  exit 1
}

while getopts "e:s:h" opt; do
  case $opt in
    e) ENV_NAME="$OPTARG" ;;
    s) SESSION_NAME="$OPTARG" ;;
    h) usage ;;
    *) usage ;;
  esac
done

if [ -z "$SESSION_NAME" ]; then
  SESSION_NAME="${ENV_NAME}_session"
fi

# Find conda's conda.sh
CONDA_SH=""
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE=$(conda info --base 2>/dev/null || true)
  if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    CONDA_SH="$CONDA_BASE/etc/profile.d/conda.sh"
  fi
fi

FALLBACKS=(
  "$HOME/miniconda3/etc/profile.d/conda.sh"
  "$HOME/mambaforge/etc/profile.d/conda.sh"
  "$HOME/anaconda3/etc/profile.d/conda.sh"
  "/tmp2/b12902115/miniconda3/etc/profile.d/conda.sh"
)
for p in "${FALLBACKS[@]}"; do
  if [ -z "$CONDA_SH" ] && [ -f "$p" ]; then
    CONDA_SH="$p"
  fi
done

if [ -z "$CONDA_SH" ]; then
  echo "Error: could not locate conda.sh. Please install Miniconda/Anaconda or adjust this script." >&2
  exit 1
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "Error: tmux is not installed. Please install tmux to use this script." >&2
  exit 1
fi

CURDIR="$PWD"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "Attaching to existing tmux session '$SESSION_NAME'..."
  tmux attach-session -t "$SESSION_NAME"
  exit 0
fi

echo "Creating tmux session '$SESSION_NAME' and activating conda env '$ENV_NAME'..."

# Create a detached session that starts in the current directory, then send commands
# into the session so the activation runs inside the tmux shell (persisting state).
tmux new-session -s "$SESSION_NAME" -d -c "$CURDIR"

# Source conda and activate the environment inside the tmux session
tmux send-keys -t "$SESSION_NAME" "source \"$CONDA_SH\"" C-m
tmux send-keys -t "$SESSION_NAME" "conda activate \"$ENV_NAME\"" C-m
tmux send-keys -t "$SESSION_NAME" "cd \"$CURDIR\"" C-m

# No need to exec a new shell; the activated environment will persist in the session
tmux attach-session -t "$SESSION_NAME"
