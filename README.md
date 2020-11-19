# 음성, 음악, 소음구별 및 NAVER음성인식API 연결 예제 <br>
<br>

## 목차 <br>
- 개요 <br>
- 프로세스 <br>
- 데이터수집(Youtube, Recording) <br>
- 학습하기 <br>
- Rule-base로 Voice구별하여 음성인식하기 <br>
- 실행순서 <br>
- Prerequisites <br>
<br>

## 개요 <br>
Siri야~ NUGU야~ 지니야~ 음성인식 엔진들을 보면 신기하기만 하고 이를 활용하면 보다 편리한 UI구성이 가능할 것 같은데, 간편하게 활용해볼 수 있는 방법이 없을까? <br>
<br>
일상 생활에선 깨끗한 음성(Voice)만 있는 것이 아니라, 음악(Music), 소음(Noise) 또, 무음상태도 있을 텐데, 어떻게 구분해서 음성인식을 할 수 있을까? <br>
<br>
간단한 Sound Classification 및 Voice Recognition 모듈을 만들어보도록 하자. <br>
<br>

## 프로세스 <br>
1. 데이터수집(Youtube, recording) <br>
2. Sound Classification (AI, Rule-base) <br>
3. Voice Recognition (NAVER API) <br>
4. 통합프로그램 <br>
<br>

## 실행순서 <br>
- AI학습하기: python train.py <br>
- Rule-base + NAVER STT실행: python main_naver_stt.py <br>
  ※ NAVER API는 1일 이용건수 제한이 있으며, 개인키 발급이 필요합니다.
  
## Prerequisites <br>
- tensorflow1.15 <br>
- librosa <br>
- requests <br>
<br>


