## 파일 경로
* tools/train/train_kfold.py 에 넣어주세요!
* http://boostcamp.stages.ai/competitions/28/discussion/post/292 에서 json 파일 다운 후 input/train_data0.json 형식으로 넣어주세요
* 기존 validation 파일명이 valid_data0.json으로 되어 있는데 이를 val_data0.json 형식으로 바꿔 주세요

## 실행 방법
* python tools/train_kfold --config [config 파일] (혹시 모르니 절대경로로 하는걸 추천합니다. )
* 파일을 실행하시면 config 폴더내에 파일명0~4.py이 5개가 생성 됩니다. 실행 끝나고 지우셔도 됩니다.
* 한 폴드가 시작될 때마다 work_dirs/파일명_kfold/파일명0/ 폴더가 생기고 그 폴더안에 로그랑 pth 파일들 정보가 저장됩니다.


### 버그 있으면 언제든 재혁님과 소현님한테 연락 주세요~