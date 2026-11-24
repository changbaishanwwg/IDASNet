"""White-dark calibration for arrays used by the optional image application."""
from pathlib import Path
import argparse
import numpy as np


def calibrate(sample,white,dark):
    sample,white,dark=(np.asarray(x,dtype=np.float32) for x in (sample,white,dark))
    denominator=white-dark
    if np.any(~np.isfinite(denominator)) or np.any(denominator<=0):
        raise ValueError('White minus dark must be finite and positive at every supplied position')
    return (sample-dark)/denominator


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['sample','white','dark','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result=calibrate(*(np.load(path,allow_pickle=False) for path in [args.sample,args.white,args.dark]))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.save(args.output,result)


if __name__=='__main__':
    main()
