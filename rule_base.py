


import os
import numpy as np
import matplotlib.pyplot as plt
import librosa




# FFT통해 Frequency 정보 추출
def fft(sig, FFT_SIZE=60, HZ=125, NFFT=2500):
    ham_window = np.hamming(NFFT) + 1e-100  # 1e-100: for recovery, devide by non-zero value
    freq = np.fft.fftfreq(NFFT, 1 / HZ)
    fft_sig = np.fft.fft(sig.reshape(-1, NFFT) * ham_window) / NFFT
    ifft_sigs = np.fft.ifft(fft_sig * NFFT) / ham_window
    fft_sig = fft_sig[:, :FFT_SIZE]
    return freq[:FFT_SIZE], abs(fft_sig[:, :FFT_SIZE]), ifft_sigs



# ------------------------------------------------------------------------------------------------
# Read wav and plot signal and fft
if True:
    AUDIO_PATH = 'dataset/'
    SAMPLE_RATE = 22050
    NFFT = 32768            # 16384
    FREQ_LIMIT = 512
    #
    res_dic = {
        'voice': {'ori_area': [], 'norm_area': [], 'norm_median': [], 'norm_std': [], 're_median': [], 're_std': []},
        'music': {'ori_area': [], 'norm_area': [], 'norm_median': [], 'norm_std': [], 're_median': [], 're_std': []},
        'noise': {'ori_area': [], 'norm_area': [], 'norm_median': [], 'norm_std': [], 're_median': [], 're_std': []},
        'silence': {'ori_area': [], 'norm_area': [], 'norm_median': [], 'norm_std': [], 're_median': [], 're_std': []},
    }
    for dirname, subdirs, files in os.walk(AUDIO_PATH):
        AUDIO_TYPE = dirname.replace('\\', '/').split('/')[-1]
        if AUDIO_TYPE not in ['voice', 'music', 'noise', 'silence']:
            continue
        #
        for audio_file in files:
            #print(AUDIO_TYPE, audio_file)
            sigs, s_rate = librosa.load(dirname + '/' + audio_file, sr=SAMPLE_RATE)
            sigs = np.abs(sigs)
            sigs_norm = sigs / np.max(sigs)
            #
            if True:
                # 0 제거 후 Re-structuring
                sigs_norm[sigs_norm < 0.05] = 0
                sigs_norm = (sigs_norm - np.min(sigs_norm)) / (np.max(sigs_norm) - np.min(sigs_norm))
                #
                sigs_no_zero = []
                for s in range(len(sigs_norm)):
                    sig = sigs_norm[s]
                    if sig != 0:
                        sigs_no_zero.append(sig)
                #
                sigs_restruct = []
                for i in range(44100):
                    sigs_restruct.append(sigs_no_zero[i % len(sigs_no_zero)])
                sigs_restruct = np.array(sigs_restruct)
            #
            # Python FFT
            freq, fft_sigs, _ = fft(sigs_restruct[: NFFT], FREQ_LIMIT, SAMPLE_RATE, NFFT)
            fft_sigs = fft_sigs.reshape(fft_sigs.shape[1])[: FREQ_LIMIT]
            fft_norm = fft_sigs / np.max(fft_sigs)
            fft_norm[:10] = 0
            #
            res_dic[AUDIO_TYPE]['ori_area'].append( np.sum(sigs) )
            res_dic[AUDIO_TYPE]['norm_area'].append( np.sum(sigs_norm) )
            res_dic[AUDIO_TYPE]['norm_median'].append( np.median(sigs_norm) )
            res_dic[AUDIO_TYPE]['norm_std'].append(np.std(sigs_norm))
            res_dic[AUDIO_TYPE]['re_median'].append(np.median(sigs_restruct))
            res_dic[AUDIO_TYPE]['re_std'].append(np.std(sigs_restruct))
            #
            print(AUDIO_TYPE, audio_file,
                  'ori_area:', np.sum(sigs),
                  'norm_area:', np.sum(sigs_norm),
                  'norm_median:', np.median(sigs_norm),
                  'norm_std:', np.std(sigs_norm),
                  'norm_median:', np.median(sigs_restruct),
                  'norm_std:', np.std(sigs_restruct)
                  )



# Box Plot of mean, median, std
if True:
    key_names = ['voice', 'music', 'noise', 'silence']
    feat_names = ['ori_area', 'norm_area', 'norm_median', 'norm_std', 're_median', 're_std']
    colors = ['red', 'blue', 'green', 'orange', 'skyblue', 'yellowgreen']
    axis_num = []
    axis = []
    boxes = []
    cnt = 0
    for key, feat_dic in res_dic.items():
        key_index = key_names.index(key)
        for feat, vals in feat_dic.items():
            if feat in feat_names:
                feat_index = feat_names.index(feat)
                cnt += 1
                axis_num.append(cnt)
                axis.append('{}_{}'.format(key[0].upper(), feat.split('_')[1]))
                if feat in ['ori_area', 'norm_area']:
                    vals_adj = []
                    for val in vals:
                        vals_adj.append(val / 10000)
                    boxes.append(vals_adj)
                else:
                    boxes.append(vals)
                #for val in vals:
                    #temp = plt.scatter(4 * key_index + feat_index, val * 1e4, s=10, c=colors[feat_index])
                    #temp = plt.scatter(4 * key_index + feat_index, val, s=10, c=colors[feat_index])
    plt.boxplot(boxes)
    plt.xticks(axis_num, axis)
    plt.show()




# ------------------------------------------------------------------------------------------------
# Read wav and plot signal and fft
# ori_area: 0.02 * 10000 이하 제거 -> Noise 제거, 짧은 음성 제거
# norm_area: 0.55 * 10000 이상 제거 -> Music제거
# norm_median: 0.1 이상 제거
if True:
    AUDIO_PATH = 'dataset/valid/'
    SAMPLE_RATE = 22050
    NFFT = 32768            # 16384
    FREQ_LIMIT = 512
    #
    cnt_correct = 0
    cnt_total = 0
    cnt_correct_voice = 0
    cnt_total_voice = 0
    for dirname, subdirs, files in os.walk(AUDIO_PATH):
        AUDIO_TYPE = dirname.replace('\\', '/').split('/')[-1]
        if AUDIO_TYPE not in ['voice', 'music', 'noise', 'silence']:
            continue
        #
        for audio_file in files:
            #print(AUDIO_TYPE, audio_file)
            sigs, s_rate = librosa.load(dirname + '/' + audio_file, sr=SAMPLE_RATE)
            sigs = np.abs(sigs)
            sigs_norm = sigs / np.max(sigs)
            #
            if True:
                # 0 제거 후 Re-structuring
                sigs_norm[sigs_norm < 0.05] = 0
                sigs_norm = (sigs_norm - np.min(sigs_norm)) / (np.max(sigs_norm) - np.min(sigs_norm))
                sigs_no_zero = []
                for s in range(len(sigs_norm)):
                    sig = sigs_norm[s]
                    if sig != 0:
                        sigs_no_zero.append(sig)
                #
                sigs_restruct = []
                for i in range(44100):
                    sigs_restruct.append(sigs_no_zero[i % len(sigs_no_zero)])
                sigs_restruct = np.array(sigs_restruct)
            #
            pred = 'others'
            # np.sum(sigs_norm) >=5000 and np.sum(sigs_norm) <=10000:
            if np.median(sigs_norm) >= 0.00 and np.median(sigs_norm) <= 0.12 \
                    and np.std(sigs_norm) >= 0.05 and np.std(sigs_norm) <= 0.15 \
                    and np.sum(sigs) >= 200 and np.sum(sigs_norm) <= 5500:
                pred = 'voice'
            #
            #print(audio_file, 'type:', AUDIO_TYPE, ', pred:', pred)
            cnt_total += 1
            if AUDIO_TYPE == 'voice':
                cnt_total_voice += 1
                if AUDIO_TYPE == pred:
                    cnt_correct += 1
                    cnt_correct_voice += 1
            else:
                if pred == 'others':
                    cnt_correct += 1
    #
    print('total:', cnt_correct, cnt_total, cnt_correct / cnt_total)
    print('voice:', cnt_correct_voice, cnt_total_voice, cnt_correct_voice / cnt_total_voice)



