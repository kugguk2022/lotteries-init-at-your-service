#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "usage: $0 BUNDLE_DIRECTORY SPACE_ID COMMIT_MESSAGE" >&2
  exit 2
fi

bundle="$1"
space_id="$2"
message="$3"

test -d "$bundle" || {
  echo "Space bundle does not exist: $bundle" >&2
  exit 1
}
test -n "${HF_TOKEN:-}" || {
  echo "HF_TOKEN is required" >&2
  exit 1
}

uploaded=false
for attempt in 1 2 3; do
  if hf upload "$space_id" "$bundle" . \
    --repo-type space \
    --commit-message "$message"; then
    uploaded=true
    break
  fi
  echo "::warning::Space API upload attempt $attempt/3 failed"
  if [[ "$attempt" -lt 3 ]]; then
    sleep "$((attempt * 10))"
  fi
done

if [[ "$uploaded" == true ]]; then
  exit 0
fi

echo "::warning::Space upload API remained unavailable; publishing through Git"
git_root="$(mktemp -d "${RUNNER_TEMP:-/tmp}/lottobench-space-git.XXXXXX")"
git clone --depth 1 "https://huggingface.co/spaces/$space_id" "$git_root"
rsync -a --delete --exclude=.git "$bundle/" "$git_root/"
git -C "$git_root" config user.name "github-actions[bot]"
git -C "$git_root" config user.email "41898282+github-actions[bot]@users.noreply.github.com"
git -C "$git_root" add -A

if git -C "$git_root" diff --cached --quiet; then
  echo "Space Git repository already contains the validated bundle"
  exit 0
fi

git -C "$git_root" commit -m "$message"
authorization="$(printf '__token__:%s' "$HF_TOKEN" | base64 -w0)"
git -C "$git_root" \
  -c http.extraheader="Authorization: Basic $authorization" \
  push origin HEAD:main

local_sha="$(git -C "$git_root" rev-parse HEAD)"
remote_sha="$(git -C "$git_root" ls-remote origin refs/heads/main | cut -f1)"
test "$local_sha" = "$remote_sha" || {
  echo "Space Git push did not advance main to the validated bundle" >&2
  exit 1
}
echo "Published Space bundle through Git at $remote_sha"
