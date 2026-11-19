# Scene Builder

Scene Builder는 TFRecord에 담긴 TFExample 주행기록을 장면 데이터셋으로 변환한다.
구축한 데이터셋에서 시나리오를 분류하고, 모델 기반으로 장면 난이도를 평가할 수 있다.

## 설치

프로젝트 폴더에서 `uv sync`를 실행해 Python 3.12 환경과 의존성을 설치한다.

## 1. 데이터 변환

91프레임의 참여자 궤적과 신호 상태, 경로 후보, 도로 점을 읽어 고정 크기 슬롯의
100프레임 장면을 만든다. 도로 점에서 차로·차선·지물을 복원하고, 0.5 m 간격으로
다시 표본화한 도로 조각에 진입·진출, 좌우 인접, 경계 관계를 연결한다. 경로 후보에서
루트를 선택하고 각 경로 점에 최근접 링크를 기록한다.

위치 관측에는 칼만 필터와 RTS 평활을 적용해 속도와 가속도를 추정한다. 자차와 가까운
정보를 우선해 슬롯을 채우고 규격 정보가 붙은 샤드로 저장한다. 장면은 NumPy로 읽을 수 있다.

```sh
uv run scene-builder count -i tfrecords
uv run scene-builder build -i tfrecords -o scenes
```

입력은 TFRecord 파일이나 폴더다. `count`는 총 레코드 수와 용량을, `-v`는 파일별 레코드 수도 표시한다.
`build`는 파일 단위로 병렬 처리하며, 다시 실행하면 이미 만들어진 샤드를 건너뛴다.
`--workers`로 병렬 수를, `--limit`로 처리할 입력 샤드 수를 지정할 수 있다.

| 명령 | 용도 |
| --- | --- |
| `info -i scenes` | 장면·샤드 수와 데이터 규격 확인 |
| `dump -i scenes --index 0` | 장면 하나의 필드, 형상, 값 범위 확인 |
| `check -i scenes --sample 1000` | 데이터 규격, 참조, 궤적 등 유효성 점검 |
| `view -i scenes -o figures --index 0` | 장면 재생 HTML 생성. `--png`와 `--grid`로 정지 이미지도 생성 가능 |

## 2. 시나리오 분류

변환된 장면의 참여자, 도로 구조, 자차 경로와 움직임을 24개 태그 값으로 요약한다.
보행자가 많이 등장하는 장면, 교차로 회전이 있는 장면, 자차 가속이 큰 장면 등을
태그 값으로 골라낼 수 있다.

```sh
uv run scene-builder tag -i scenes
```

| 태그군 | 기록하는 값 |
| --- | --- |
| `scene` | 차량·보행자·자전거·신호등의 종류별 최대 동시 등장 수(`Vehicles`, `Pedestrians`, `Cyclists`, `Traffics`), 자차가 지도 위에 있거나 차선에 인접한 프레임의 비율(`Links`, `Lanes`), 교차로·고속도로 링크 조각 수(`Intersections`, `Freeways`) |
| `route` | 자차 유효 프레임 중 직진·좌회전·우회전·유턴·후진을 동반한 방향 전환·지도 밖·루트 없는 주행에 해당하는 비율(`GoStraight`, `TurnLeft`, `TurnRight`, `UTurn`, `Pivot`, `Mapless`, `NoRoute`) |
| `ego` | 중앙 유효 프레임의 속력·전후 가속도·회전율(`vel`, `acc`, `dir_rate`)과 장면 양끝 약 1초를 제외한 구간의 변화량(`Δvel`, `Δacc`, `Δdir`, `Δpos`, `Δpos_x`, `Δpos_y`) |

새 데이터셋과 기존 데이터셋에 같은 명령을 사용한다. 기본 출력은 데이터셋 폴더의
`tags.jsonl`이며, 장면마다 `index`와 `scene_id`로 식별되는 한 줄을 기록한다. 판단할 근거가
부족한 경로·운동 값은 `null`로 표시한다. `-o`로 출력 파일을 지정할 수 있다.

## 3. 장면 난이도 분류

내장 주행 모델이 각 장면에서 경로를 반복해서 계획하고 자차를 폐루프로 주행시킨다.
주행 결과를 평가한 종합점수가 `Score.EPDMS`다. 낮은 점수는 모델이 해당 장면에서
주행에 어려움을 겪었음을 나타낸다. 점수순으로 정렬해 분석할 장면을 고를 수 있다.

```sh
uv run --extra score scene-builder tag -i scenes --score -o scored-tags.jsonl
```

결과에는 시나리오 태그와 `Score`가 함께 들어간다. `Score`에는 종합점수와 충돌·주행
가능 영역·주행 방향·신호 준수·진행·충돌 여유·차로 유지·승차감 세부 점수(`NC`, `DAC`,
`DDC`, `TLC`, `EP`, `TTC`, `LK`, `HC`, `EC`)를 기록한다. 자차나 유효 경로가 부족해
채점할 수 없는 값과 게시 제한속도가 필요한 `SLC`는 `null`이다. 주행 모델은 10 Hz에서
이력 20프레임과 미래 최대 40프레임을 사용하며 마지막 유효 자차 프레임까지만 채점한다.

처음 채점할 때 PyTorch가 동봉된 C++ 실행 모듈을 빌드한다. 이후 실행은 빌드 캐시를 재사용한다.
Windows에서는 Visual Studio 2022의 **x64 Native Tools Command Prompt**에서 실행한다.
Linux에서는 설치된 PyTorch와 호환되는 C++ 컴파일러가 필요하다. 기본 채점 장치는 CPU다.

각 명령의 전체 옵션은 `uv run scene-builder <명령> -h`로 확인할 수 있다.
