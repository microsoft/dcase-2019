import os

import librosa
import numpy as np
from numpy import concatenate, std, abs
from numpy.fft import ifft, fftfreq
from numpy.random import normal
from sklearn.externals import joblib


def add_background_noise(audio_array, background_amplitude_min_max=(0.25, 0.5),
                         background_proportion_min_max=(0.2, 0.8), background_level_delta=4,
                         noise_type='pink', fmin=0):
    """
    Noise code adapted from: https://github.com/felixpatzelt
    Gaussian (1/f)**beta noise.

    Based on the algorithm in:
    Timmer, J. and Koenig, M.:
    On generating power law noise.
    Astron. Astrophys. 300, 707-710 (1995)

    Normalised to unit variance

    :param audio_array: 1D numpy array from an audio file
    :param background_amplitude_min_max: minimum and maximum absolute value of the amplitude of the noise by itself
                                        (0 <= noise <= 1)
    :param background_proportion_min_max: min and max proportion of background noise to sound (0 <= poportion <= 1)
    :param background_level_delta: the step size of dB change of augmentation
    :param noise_type: white, pink, brown
    :param fmin: low frequency cutoff
    :return: list of arrays, including the original and those augmented with background noise
    """

    sounds_with_background = [audio_array]
    changes = [(None, None)]
    bkgrd_db_min, bkgrd_db_max = background_amplitude_min_max
    bkgrd_level_delta = background_level_delta
    min_bkgrd_proportion, max_bkgrd_proportion = background_proportion_min_max

    noise_to_exponent = {'white': 0, 'pink': 1, 'brown': 2}
    noise_type = noise_type
    if noise_type not in noise_to_exponent.keys():
        raise Exception('invalid noise type; valid types = white, pink, brown')
    exponent = noise_to_exponent[noise_type]
    samples = len(audio_array)

    # frequencies (we assume a sample rate of one)
    f = fftfreq(samples)

    # scaling factor for all frequencies
    ## though the fft for real signals is symmetric,
    ## the array with the results is not - take neg. half!
    s_scale = abs(concatenate([f[f < 0], [f[-1]]]))

    ## low frequency cutoff?!?
    # TODO: determine if fmin is necessary
    if fmin:
        ix = sum(s_scale > fmin)
        if ix < len(f):
            s_scale[ix:] = s_scale[ix]
    s_scale = s_scale ** (-exponent / 2.)

    # scale random power + phase
    sr = s_scale * normal(size=len(s_scale))
    si = s_scale * normal(size=len(s_scale))
    if not (samples % 2): si[0] = si[0].real

    s = sr + 1J * si

    # this is complicated... because for odd sample numbers,
    ## there is one less positive freq than for even sample numbers
    s = concatenate([s[1 - (samples % 2):][::-1], s[:-1].conj()])

    # time series
    y = ifft(s).real.astype(np.float32)
    y = y / std(y)
    y = librosa.util.normalize(y, norm=np.inf, axis=None)
    for delta_amp in np.arange(bkgrd_db_min, bkgrd_db_max + 1e-8, bkgrd_level_delta):
        if delta_amp + abs(audio_array).max() > 1.0:
            break
        elif delta_amp / abs(audio_array).max() > max_bkgrd_proportion:
            continue
        elif delta_amp / abs(audio_array).max() < min_bkgrd_proportion:
            continue

        sounds_with_background.append((audio_array + y * delta_amp))
        changes.append((delta_amp, noise_type))

    return sounds_with_background, changes

def shift_pitch(audio_array, sr, semitone_delta_min_max=(3, 3), semitones_stride=3,):
    """
    shifts pitch of audio example up and/or down
    :param audio_array: 1D numpy array from an audio file
    :param sr: sample rate (int)
    :param semitone_delta_min_max: min and max number of semitones to decrease and increase pitch
    :param semitones_stride: number of semitones to change per stride
    :return: list of audio arrays, including original and those with shifted pitch
    """

    down_max, up_max = semitone_delta_min_max
    stride = semitones_stride
    Y = [audio_array]
    changes = [0]

    for delta in np.arange(-down_max, up_max + 1e-8, stride):
        if abs(delta) < 1e-8:
            continue
        Y.append(librosa.effects.pitch_shift(audio_array, sr, delta, bins_per_octave=12))
        changes.append(delta)
    return Y, changes

def change_volume(audio_array=None, db_change=1, amplitude_min_max=(0.5, 1), max_dB_decrease_increase=(4, 4)):
    """
    Given a wav_array, creates more wav_arrays of different volumes that fit the parameters
    Returns: tuple: a list of 1d numpy arrays for each volume change, a list of number of dBs changed from original

    """
    wav_array = audio_array
    min_amplitude, max_amplitude = amplitude_min_max
    max_decrease, max_increase = max_dB_decrease_increase

    all_wavs = [wav_array]
    changes = [0]

    # increase volume by stride until max_amplitude or max_increase reached
    temp_y = wav_array * db_change
    increase = 0
    while abs(temp_y).max() <= max_amplitude:
        if max_increase and increase < max_increase:
            all_wavs.append(temp_y)
            temp_y = temp_y * (10**(db_change / 20))
            increase += 1
            changes.append(increase)
        else:
            break

    # decrease volume by stride until min_amplitude or max_decrease reached
    temp_y = wav_array / db_change
    decrease = 0
    while abs(temp_y).max() >= min_amplitude:
        if max_decrease and decrease < max_decrease:
            all_wavs.append(temp_y)
            temp_y = temp_y / (10**(db_change / 20))
            decrease += 1
            changes.append(-decrease)
        else:
            break
    return all_wavs, changes


def make_log_mel_spectrogram(audio_array, sr):
    S = librosa.feature.melspectrogram(y=audio_array, sr=sr)
    S = librosa.power_to_db(S, ref=np.max)
    return S


def save_augmented_files(audio_file_path, label, save_dir, volume=True, pitch=True, bkgrd=True, **kwargs):
        y, sr = librosa.load(audio_file_path, **kwargs, res_type='kaiser_fast')
        file_name, ext = os.path.splitext(os.path.basename(audio_file_path))
        os.makedirs(save_dir, exist_ok=True)
        file_name = file_name + '_pitch_changes-0_volume_changes-0_background_changes-None-None.pkl'
        if volume:
            with_volume, changes = change_volume(y, **kwargs)
            vol_names = [file_name.replace('volume_changes-0', f'volume_changes-{c}') for c in changes]
        else:
            with_volume = [y]
            vol_names = [file_name]
        with_pitch_shift = []
        if pitch:
            pitch_names = []
            for f, n in zip(with_volume, vol_names):
                pitch_arrays, changes = shift_pitch(f, sr, **kwargs)
                with_pitch_shift.extend(pitch_arrays)
                pitch_names.extend([file_name.replace('pitch_changes-0', f'pitch_changes-{c}') for c in changes])

        else:
            with_pitch_shift = with_volume
            pitch_names = vol_names
        with_bkgrd_noise = []
        bkgrd_names = []
        if bkgrd:
            for f, n in with_pitch_shift, vol_names:
                bkgrd, changes = add_background_noise(f, **kwargs)
                with_bkgrd_noise.extend(bkgrd)
                bkgrd_names.extend([file_name.replace('background_changes-None-None',
                                                      f'background_changes-{c1}-{c2}') for c1, c2 in changes])

        else:
            with_bkgrd_noise = with_pitch_shift
            bkgrd_names = pitch_names
        labels = [label for i in range(len(with_bkgrd_noise))]
        for name, array, aug_label in zip(bkgrd_names, with_bkgrd_noise, labels):
            array = make_log_mel_spectrogram(array, sr)
            joblib.dump((array, label), os.path.join(save_dir, name))

