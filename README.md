# AlignScan ♡

Windows용 영수증·카드 다중 스캔 분리 및 원근 보정 앱입니다.

## 기능
- 한 스캔 이미지에서 여러 영수증/카드의 사각 윤곽 자동 검출
- 문서별 기울기 및 원근 보정, 수동 4점 지정과 모서리 드래그 수정
- 밝기·대비·채도 조절
- 강도 조절식 노이즈 제거와 작은 먼지 제거
- PNG 무손실 저장, JPG 품질 40~100% 지정 저장
- 귀여운 영수증 캐릭터 애니메이션 스플래시
- 선택형 ONNX 문서 검출 모델 및 DirectML 우선 / CPU 대체 실행
- GitHub Actions Windows x64 EXE 자동 빌드

## Windows 사용
Python 3.11 64-bit 설치 후 `run_windows.bat` 실행.
EXE 직접 빌드는 `build_windows.bat` 실행.
GitHub Actions 탭의 **Build Windows EXE** 실행 결과에서 `AlignScan.exe` 아티팩트를 다운로드할 수 있습니다.

## AI / Ryzen 7 4800U
기본 검출과 보정은 로컬 OpenCV CPU 처리입니다. 별도의 문서/카드 학습 YOLOv8 Detection ONNX 모델을 선택하면 ONNX Runtime DirectML 공급자를 먼저 시도하고, 불가능한 경우 CPU를 사용합니다. 실제 GPU 연산은 드라이버와 모델 연산자 호환성에 좌우됩니다.

## 주의
자동 검출은 스캔 배경과 종이 테두리 대비에 영향을 받습니다. 결과가 정확하지 않을 때에는 네 모서리 직접 지정 기능으로 수정하세요. 먼지 제거 강도가 높으면 작은 인쇄점까지 사라질 수 있으니 미리보기를 확인하세요.

MIT License.
