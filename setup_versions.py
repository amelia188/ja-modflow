# -*- coding: utf-8 -*-
"""
setup_versions.py — put the ja-modflow project under Git with one tag per model version.

What it does (run it once, from the ja-modflow folder):
  1. Writes/updates .gitignore so model outputs, binaries and caches stay out of Git.
  2. Builds the model's history as real commits, one per version, all at ONE path
     (modflowandfiles/run_model.py), and tags them:
        v19  run_model19.py                          (MODFLOW-2005, steady state)
        v20  run_model20.py                          (transient recharge, April 2026 report)
        v21  run_modflow_transient_recharge_v21.py   (end-of-month heads, 3 cycles, observed heads)
     so `git diff v20 v21` shows exactly what changed between versions.
  3. Commits the current project (v22, MODFLOW 6) on top and tags it v22.

It never edits, moves or deletes your files: the history commits are built directly in
Git's object store from the old scripts you point it at. If the folder is already a Git
repository with commits, your existing history is left untouched; the v19-v21 commits are
added as a separate line of history and only the tags point at them.

Usage
  python setup_versions.py --dry-run                 # show what would happen
  python setup_versions.py                           # do it (finds the old scripts by name)
  python setup_versions.py --v19 path\\to\\run_model19.py --v20 path\\to\\run_model20.py

  Any version whose script is not found is skipped with a message.

After it finishes
  git log --oneline --decorate --all     # the commits and tags
  git diff v21 v22 -- modflowandfiles/run_model.py
  git push origin main --tags            # when the GitHub remote is set up

Every model run then records its Git commit in runs/<model>/run_config.json, and warns if
there were uncommitted changes, so commit before any run you will report.
"""

import os
import sys
import glob
import argparse
import subprocess
import tempfile

MODEL_PATH = "modflowandfiles/run_model.py"    # one path for every version of the model

VERSIONS = [  # tag, default file name(s) to look for, commit message
    ("v19", ["run_model19.py"],
     "Model v19: MODFLOW-2005, single steady-state period, uniform recharge 1e-4 m/d, K 20 m/d, CHD 5 m"),
    ("v20", ["run_model20.py", "run_modflow_transient_recharge.py"],
     "Model v20: transient, 1 spin-up + 12 monthly periods, IDW recharge x 0.17, K 50 m/d, CHD 100 m "
     "(April 2026 report)"),
    ("v21", ["run_modflow_transient_recharge_v21.py"],
     "Model v21: end-of-month heads, 3 cycles, no-flow north, observed heads and calibration stats, "
     "K 200 m/d, CHD 52 m. Known issue: LPF rewetting activated cells outside the basin."),
]
V22_MESSAGE = ("Model v22: MODFLOW 6 port; outputs in runs/ beside the scripts; run_config.json and grid.npz "
               "per run; flow from DATA-SPDIS; export and 3D viewer read grid.npz. Fixes v21 rewetting "
               "of inactive cells.")

GITIGNORE = """# --- model outputs (rebuild them by running the scripts) ---
runs/
mf2005_*/
**/hydrowindy_data/
hydro-windy-app/data/*.json
*.hds
*.cbc
*.lst
*.list
*.ucn
# --- binaries and caches ---
*.exe
mf6
mf2005
__pycache__/
*.pyc
.ipynb_checkpoints/
.DS_Store
Thumbs.db
"""

BIG_FILE_MB = 50


def git(*args, check=True, env=None, input_text=None):
    r = subprocess.run(["git", *args], capture_output=True, text=True, env=env, input=input_text)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed:\n{r.stderr.strip()}")
    return r.stdout.strip()


def find_script(names, explicit):
    if explicit:
        return explicit if os.path.isfile(explicit) else None
    for n in names:
        hits = [h for h in glob.glob(os.path.join("**", n), recursive=True) if ".git" not in h]
        if hits:
            return sorted(hits, key=len)[0]
    return None


def update_gitignore(dry):
    existing = open(".gitignore").read() if os.path.exists(".gitignore") else ""
    missing = [l for l in GITIGNORE.splitlines() if l and not l.startswith("#") and l not in existing.splitlines()]
    if not missing:
        print("· .gitignore already covers outputs and binaries")
        return
    print(f"· .gitignore: adding {len(missing)} entries ({', '.join(missing[:5])}{'…' if len(missing) > 5 else ''})")
    if not dry:
        with open(".gitignore", "a") as f:
            f.write(("\n" if existing and not existing.endswith("\n") else "") + GITIGNORE)


def big_files():
    out = []
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in (".git", "runs", "__pycache__", "node_modules")
                   and not d.startswith("mf2005_")]
        for f in files:
            p = os.path.join(root, f)
            try:
                if os.path.getsize(p) > BIG_FILE_MB * 1e6:
                    out.append((p, os.path.getsize(p) / 1e6))
            except OSError:
                pass
    return out


def commit_single_file(content_path, parent, message, tag_date=None):
    """Build a commit whose tree holds only MODEL_PATH, without touching the working tree or index."""
    blob = git("hash-object", "-w", content_path)
    with tempfile.TemporaryDirectory() as td:
        env = dict(os.environ, GIT_INDEX_FILE=os.path.join(td, "index"))
        git("update-index", "--add", "--cacheinfo", f"100644,{blob},{MODEL_PATH}", env=env)
        tree = git("write-tree", env=env)
    args = ["commit-tree", tree, "-m", message] + (["-p", parent] if parent else [])
    return git(*args)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v19"); ap.add_argument("--v20"); ap.add_argument("--v21")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    dry = a.dry_run

    if not os.path.isfile(MODEL_PATH):
        sys.exit(f"Run this from the ja-modflow folder: {MODEL_PATH} (the v22 model) was not found here.")
    try:
        git("--version")
    except Exception:
        sys.exit("Git is not installed or not on PATH. Install Git for Windows, then rerun.")

    print(f"{'DRY RUN: ' if dry else ''}setting up versions in {os.getcwd()}\n")
    is_repo = os.path.isdir(".git")
    has_commits = is_repo and git("rev-parse", "--verify", "-q", "HEAD", check=False) != ""
    tags = set(git("tag", check=False).split()) if is_repo else set()

    if not is_repo:
        print("· git init (new repository, branch 'main')")
        if not dry:
            git("init", "-b", "main")
    else:
        print(f"· existing repository ({'has history: it will not be rewritten' if has_commits else 'no commits yet'})")
    if not dry and not git("config", "user.name", check=False):
        print("  NOTE: set your name and email first:  git config --global user.name \"Amelia ...\"  "
              "and  git config --global user.email \"...\"")
        sys.exit(1)

    update_gitignore(dry)
    for p, mb in big_files():
        print(f"  WARNING: {p} is {mb:.0f} MB. GitHub refuses files over 100 MB; consider Git LFS or .gitignore.")

    # history commits v19 -> v21
    explicit = {"v19": a.v19, "v20": a.v20, "v21": a.v21}
    parent = None
    for tag, names, msg in VERSIONS:
        path = find_script(names, explicit[tag])
        if tag in tags:
            print(f"· {tag}: tag already exists, skipped")
            continue
        if not path:
            print(f"· {tag}: script not found ({' / '.join(names)}), skipped. Use --{tag} PATH to point at it.")
            continue
        print(f"· {tag}: commit {path} as {MODEL_PATH}, tag {tag}")
        if not dry:
            parent = commit_single_file(path, parent, msg)
            git("tag", "-a", tag, parent, "-m", msg)

    # current project = v22 on main
    if "v22" in tags:
        print("· v22: tag already exists; commit your changes normally")
        return
    print(f"· v22: commit the current project on 'main' and tag v22")
    if dry:
        print("\nDry run only. Rerun without --dry-run to apply.")
        return
    if not has_commits and parent:
        git("update-ref", "refs/heads/main", parent)     # main continues from v21
        git("reset", "-q")                               # index matches that commit; files untouched
    tracked_ignored = [f for f in git("ls-files", "-ci", "--exclude-standard", check=False).splitlines() if f]
    if tracked_ignored:
        print(f"· untracking {len(tracked_ignored)} output/binary files committed earlier "
              f"(your copies are kept): {', '.join(tracked_ignored[:4])}{'…' if len(tracked_ignored) > 4 else ''}")
        git("rm", "-r", "-q", "--cached", "--", *tracked_ignored)
    git("add", "-A")
    if git("status", "--porcelain"):
        git("commit", "-q", "-m", V22_MESSAGE)
    git("tag", "-a", "v22", "HEAD", "-m", V22_MESSAGE)
    print("\n✓ Done.\n")
    print(git("log", "--oneline", "--decorate", "--all", "-n", "8"))
    print("\nNext: create the GitHub repository (or use the existing ja-modflow one), then\n"
          "  git remote add origin https://github.com/<you>/ja-modflow.git   (if not already set)\n"
          "  git push origin main --tags")


if __name__ == "__main__":
    main()
