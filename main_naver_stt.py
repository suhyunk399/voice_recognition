


import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import requests


client_id = "krc0aufydj"
client_secret = "3JdVK6FeBz3I9hRmoiBSBmfjUW3P83jboDY6UxJt"
lang = "Kor" # 언어 코드 ( Kor, Jpn, Eng, Chn )
url = "https://naveropenapi.apigw.ntruss.com/recog/v1/stt?lang=" + lang



# ------------------------------------------------------------------------------------------------
# Read wav and plot signal and fft
# ori_area: 0.02 * 10000 이하 제거 -> Noise 제거, 짧은 음성 제거
# norm_area: 0.55 * 10000 이상 제거 -> Music제거
# norm_median: 0.1 이상 제거
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
                # ------------------------------------------------------------------
                # 음성인식
                try:
                    data = open(dirname + '/' + audio_file, 'rb')
                    headers = {
                        "X-NCP-APIGW-API-KEY-ID": client_id,
                        "X-NCP-APIGW-API-KEY": client_secret,
                        "Content-Type": "application/octet-stream"
                    }
                    response = requests.post(url, data=data, headers=headers)
                    rescode = response.status_code
                    if (rescode == 200):
                        print()
                        print(audio_file, response.text)
                    else:
                        print("Error : " + response.text)
                except Exception as ex:
                    print(str(ex))
                # ------------------------------------------------------------------
        else:
            if pred == 'others':
                cnt_correct += 1



print('total:', cnt_correct, cnt_total, cnt_correct / cnt_total)
print('voice:', cnt_correct_voice, cnt_total_voice, cnt_correct_voice / cnt_total_voice)


