"""Repository-root CLI. Merely inspecting paths never starts GPU work."""
from pathlib import Path
import sys
import subprocess
ROOT=Path(__file__).resolve().parent
CODE=ROOT/'scripts/experiments'
sys.path.insert(0,str(CODE))

def main():
    if len(sys.argv)<2 or sys.argv[1]=='paths':
        from common.local_paths import DEFAULTS,local_path
        for name in DEFAULTS:
            p=local_path(name);print(f'{name}: {p} (exists={p.exists()})')
        return
    args=sys.argv[1:]
    if args[0]=='analyze-data':
        raise SystemExit(subprocess.call([sys.executable,str(ROOT/'scripts/analyze_dataset.py'),*args[1:]],cwd=ROOT))
    # Config paths supplied to train/prepare/preflight are relative to experiments,
    # or explicit absolute paths. Other CLI arguments retain the existing API.
    if args[0] in {'check','train','retrain','prepare','preview'} and len(args)>1:
        p=Path(args[1]);args[1]=str(p.resolve() if p.exists() else CODE/p)
    # Explicit output/run paths are interpreted relative to the caller, not cwd=CODE.
    if '--output' in args:
        index=args.index('--output')+1
        args[index]=str(Path(args[index]).resolve())
    if args[0] in {'select','predict'} and len(args)>1:
        args[1]=str(Path(args[1]).resolve())
    if args[0]=='compare':
        for index in range(1,len(args)):
            if args[index].startswith('--'):break
            args[index]=str(Path(args[index]).resolve())
    raise SystemExit(subprocess.call([sys.executable,str(CODE/'run.py'),*args],cwd=CODE))

if __name__=='__main__':main()
