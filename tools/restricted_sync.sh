#!/usr/bin/env bash
# Restricted NPB tables <-> private repo checkout.
#
# baseball-data.com and npb.jp grant no redistribution right, so the
# player-level tables are kept in the PRIVATE repo
# yasumorishima/npb-restricted-inputs (layout mirrors this tree, with
# MD5SUMS.txt). Workflows check it out at ./_restricted and use this script.
# The definition of the restricted set lives only here (is_restricted) and in
# .gitignore; keep the two in step.
#
#   restricted_sync.sh verify          md5sum -c inside the private checkout, no unlisted files
#   restricted_sync.sh pull            verify, then copy the restricted files into the workspace
#   restricted_sync.sh push "<msg>"    copy restricted files back, regenerate MD5SUMS.txt,
#                                      commit + push to the private repo if anything changed
#   restricted_sync.sh guard           fail if any restricted path is tracked or staged here
#   restricted_sync.sh hf-stage <dir>  copy data/ to <dir> without restricted files (HF mirror)
#   restricted_sync.sh list            list restricted files present in the workspace
#
# All paths are relative to the repository root (the script cd's there).
set -euo pipefail
export LC_ALL=C

cd "$(dirname "$0")/.."
PRIV="${NPB_RESTRICTED_DIR:-_restricted}"

die() { echo "::error::restricted_sync: $*" >&2; exit 1; }

# Path (relative to repo root) -> 0 if restricted.
is_restricted() {
  local p="${1#./}"
  [[ "$p" =~ ^data/raw/npb_standings_[^/]*\.csv$ ]] && return 1   # team standings stay public
  [[ "$p" =~ ^data/raw/ ]] && return 0
  [[ "$p" =~ ^research/stan/data/raw/ ]] && return 0
  [[ "$p" =~ ^data/projections/npb_sabermetrics_[^/]*\.csv$ ]] && return 0
  [[ "$p" =~ ^research/team_2026/games_2026[^/]*\.csv$ ]] && return 0
  [[ "$p" =~ ^_restricted(/|$) ]] && return 0
  return 1
}

workspace_restricted_files() {
  {
    find data/raw research/stan/data/raw -type f 2>/dev/null || true
    find data/projections -maxdepth 1 -type f -name 'npb_sabermetrics_*.csv' 2>/dev/null || true
    find research/team_2026 -maxdepth 1 -type f -name 'games_2026*.csv' 2>/dev/null || true
  } | sed 's|^\./||' | while IFS= read -r f; do
    if is_restricted "$f"; then echo "$f"; fi
  done | sort -u
}

listed_files() {   # paths listed in the private MD5SUMS.txt
  sed 's/^[0-9a-f]\{32\}  //' "$PRIV/MD5SUMS.txt"
}

private_files() {  # data files present in the private checkout
  (cd "$PRIV" && find . -type f ! -path './.git/*' ! -path ./README.md ! -path ./MD5SUMS.txt | sed 's|^\./||' | sort)
}

cmd_verify() {
  [ -d "$PRIV/.git" ] || die "$PRIV is not a git checkout"
  [ -s "$PRIV/MD5SUMS.txt" ] || die "$PRIV/MD5SUMS.txt is missing or empty"
  (cd "$PRIV" && md5sum -c --quiet --strict MD5SUMS.txt) || die "md5sum -c failed in $PRIV"
  if [ "$(listed_files | sort)" != "$(private_files)" ]; then
    diff <(listed_files | sort) <(private_files) >&2 || true
    die "files in $PRIV differ from MD5SUMS.txt"
  fi
  echo "verify: $(listed_files | wc -l) files OK in $PRIV"
}

cmd_pull() {
  cmd_verify
  local n=0 skipped=0 f
  while IFS= read -r f; do
    if is_restricted "$f"; then
      mkdir -p "$(dirname "$f")"
      cp -- "$PRIV/$f" "$f"
      n=$((n + 1))
    else
      skipped=$((skipped + 1))   # public files (standings): the public copy is canonical
    fi
  done < <(listed_files)
  [ "$n" -gt 0 ] || die "no restricted files copied"
  # the copies in the workspace must match the private checksums
  grep -E '^[0-9a-f]{32}  ' "$PRIV/MD5SUMS.txt" | while IFS= read -r line; do
    f="${line#*  }"
    if is_restricted "$f"; then echo "$line"; fi
  done | md5sum -c --quiet --strict - || die "workspace copies do not match MD5SUMS.txt"
  echo "pull: copied $n restricted files into the workspace (left $skipped public files untouched)"
}

cmd_push() {
  local msg="${1:-}" f n=0 branch
  [ -n "$msg" ] || die "push needs a commit message"
  [ -d "$PRIV/.git" ] || die "$PRIV is not a git checkout"
  while IFS= read -r f; do
    mkdir -p "$PRIV/$(dirname "$f")"
    cp -- "$f" "$PRIV/$f"
    n=$((n + 1))
  done < <(workspace_restricted_files)
  [ "$n" -gt 0 ] || die "no restricted files in the workspace to push back"
  # keep the private mirror of public files (standings) in step with the workspace
  while IFS= read -r f; do
    if ! is_restricted "$f" && [ -f "$f" ]; then cp -- "$f" "$PRIV/$f"; fi
  done < <(listed_files)
  # regenerate MD5SUMS.txt: "md5  path", sorted, README.md / MD5SUMS.txt excluded
  local tmp; tmp=$(mktemp)   # outside $PRIV so find cannot see it
  private_files | (cd "$PRIV" && while IFS= read -r f; do md5sum -- "$f"; done) > "$tmp"
  mv -- "$tmp" "$PRIV/MD5SUMS.txt"; chmod 644 "$PRIV/MD5SUMS.txt"
  cmd_verify
  cd "$PRIV"
  git add -A
  if git diff --cached --quiet; then
    echo "push: private repo unchanged"
    return 0
  fi
  git diff --cached --stat
  branch=$(git symbolic-ref --short HEAD) || die "private checkout is on a detached HEAD"
  git -c user.name='github-actions[bot]' -c user.email='github-actions[bot]@users.noreply.github.com' \
    commit -q -m "$msg"
  git push origin "HEAD:refs/heads/$branch"
  echo "push: committed and pushed $(git rev-parse --short HEAD) to the private repo"
}

cmd_guard() {
  local bad=0 f
  # -z: paths unquoted and NUL-separated (with quoting, non-ASCII names would not match)
  while IFS= read -r -d '' f; do
    if is_restricted "$f"; then echo "::error::restricted file tracked or staged in the public repo: $f" >&2; bad=1; fi
  done < <(git ls-files -z; git diff --cached --name-only -z --no-renames)
  [ "$bad" -eq 0 ] || die "refusing to commit restricted files to the public repo"
  echo "guard: no restricted file tracked or staged ($(git ls-files | wc -l) tracked, $(git diff --cached --name-only | wc -l) staged)"
}

cmd_hf_stage() {
  local dest="${1:-}" removed=0 rel
  [ -n "$dest" ] || die "hf-stage needs a destination directory"
  case "$(realpath -m "$dest")" in "$(pwd)"|"$(pwd)"/*) die "hf-stage destination must be outside the repo";; esac
  rm -rf -- "$dest"; mkdir -p -- "$dest"
  cp -a data/. "$dest"/
  while IFS= read -r rel; do
    if is_restricted "data/$rel"; then rm -f -- "$dest/$rel"; removed=$((removed + 1)); fi
  done < <(cd "$dest" && find . -type f | sed 's|^\./||')
  while IFS= read -r rel; do
    is_restricted "data/$rel" && die "restricted file left in HF staging: data/$rel"
  done < <(cd "$dest" && find . -type f | sed 's|^\./||')
  echo "hf-stage: $(cd "$dest" && find . -type f | wc -l) files staged in $dest, $removed restricted files excluded"
}

case "${1:-}" in
  verify) cmd_verify ;;
  pull) cmd_pull ;;
  push) shift; cmd_push "$@" ;;
  guard) cmd_guard ;;
  hf-stage) shift; cmd_hf_stage "$@" ;;
  list) workspace_restricted_files ;;
  is-restricted) shift; is_restricted "$1" ;;
  *) echo "usage: $0 {verify|pull|push <msg>|guard|hf-stage <dir>|list|is-restricted <path>}" >&2; exit 2 ;;
esac
