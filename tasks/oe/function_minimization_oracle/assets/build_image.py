"""Build only the public candidate image from an already cached pinned base."""

import argparse
from pathlib import Path
import subprocess

from oracle_runtime import BASE_IMAGE, runtime_assets_sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag',default='sciel-oracle-v1:development')
    args=parser.parse_args()
    if not args.tag.startswith('sciel-oracle-'):
        parser.error('use an owned sciel-oracle- image tag')
    subprocess.run(['docker','image','inspect',BASE_IMAGE],stdout=subprocess.DEVNULL,
                   check=True,timeout=10)
    subprocess.run(['docker','build','--pull=false','--network=none',
        '--label','sciel.oracle.assets.sha256='+runtime_assets_sha256(),
        '-t',args.tag,str(Path(__file__).resolve().parent)],check=True,timeout=120)
    print(subprocess.check_output(['docker','image','inspect',args.tag,
        '--format','{{.Id}}'],text=True,timeout=10).strip())


if __name__=='__main__':
    main()
