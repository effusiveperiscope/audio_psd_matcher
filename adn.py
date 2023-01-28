import soundfile as sf
import matplotlib.pyplot as plt
import scipy
import numpy as np
import argparse
import pathlib
import os

NUM_TAPS=99

def avg_psd(paths):
    psds = []
    samplerate = 0
    f = []
    for path in paths:
        with sf.SoundFile(path) as fi:
            signal = fi.read()
            samplerate = fi.samplerate
            f, psd = scipy.signal.welch(signal, fs=samplerate)
            psds.append(psd)
    res_psd = np.mean(psds,(0))
    # plt.plot(f, res_psd)
    # plt.xscale('log')
    # plt.show()
    return res_psd, samplerate

def spect_norm(ref_psd, target_file, samplerate):
    with sf.SoundFile(target_file) as fi:
        signal = fi.read()
        f, psd = scipy.signal.welch(signal, samplerate)
        psd_div = psd / ref_psd
        eq_gain = 1 / (1e-6 + psd_div) ** 0.5
        eq_filter = scipy.signal.firwin2(NUM_TAPS,f,eq_gain,fs=samplerate)
        return scipy.signal.convolve(signal, eq_filter)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_dir",type=str, required=True,
        help="directory of target files")
    parser.add_argument("--source_dir",type=str, required=True,
        help="directory of source files")
    parser.add_argument("--output_dir",type=str, default='.',
        help="directory of output, current directory by default")
    parser.add_argument("--output_filts",help="Output filters as json",
        action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.target_dir):
        print("Nonexistent target directory "+args.target_dir)
        return
    if not os.path.exists(args.source_dir):
        print("Nonexistent source directory "+args.source_dir)
        return
    if not os.path.exists(args.output_dir):
        print("Nonexistent output directory "+args.output_dir)
        return

    target_files = [os.path.join(args.target_dir,f)
        for f in os.listdir(args.target_dir) if
        os.path.isfile(os.path.join(args.target_dir,f))]
    source_files = [os.path.join(args.source_dir,f)
        for f in os.listdir(args.source_dir) if
        os.path.isfile(os.path.join(args.source_dir,f))]
    print("Averaging target spectra...")
    avgpsd, samplerate = avg_psd(target_files)
    print("Finished averaging target spectra.")

    print("Processing source files...")
    for s in source_files:
        print("Processing "+s)
        sig = spect_norm(avgpsd, s, samplerate)
        out_name = pathlib.Path(os.path.basename(s)).stem + '_out'
        out_name = pathlib.Path(out_name).with_suffix('.wav')
        sf.write(os.path.join(args.output_dir,str(out_name)),sig, samplerate)
    print("Done.")

if __name__ == '__main__':
    main()
