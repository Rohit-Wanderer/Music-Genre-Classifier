"""Script extracts the MFCC from the dataset and makes
frequency "prints" of all .wav files passed as input.
"""

from scikits.talkbox.features import mfcc
import scipy.io.wavfile
import numpy as np
import sys
import os

# Given a wavfile, computes mfcc and saves mfcc data
def create_ceps(wavfile):
	sampling_rate, song_array = scipy.io.wavfile.read(wavfile)
	"""Get MFCC
	ceps  : ndarray of MFCC
	mspec : ndarray of log-spectrum in the mel-domain
	spec  : spectrum magnitude
	"""
	ceps, mspec, spec = mfcc(song_array)
	write_ceps(ceps, wavfile)

# Saves mfcc data
def write_ceps(ceps, wavfile):
	base_wav, ext = os.path.splitext(wavfile)
	data_wav = base_wav + ".ceps"
	np.save(data_wav, ceps)


def main():
	train_dirs = sys.argv[1:]
	for train_dir in train_dirs:
		os.chdir(train_dir)
		for wavfile in os.listdir(train_dir):
			# print(wavfile)
			create_ceps(wavfile)
		os.system("rm *.wav")
		os.system("ls")

if __name__ == "__main__":
	main()
