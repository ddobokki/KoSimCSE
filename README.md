# KoSimCSE
Korean SimCSE using PLM in huggingface hub

- sentence-transformers 라이브러리를 이용해 1차 작업중
  - 해당 라이브러리는 Multi Gpu를 지원하지 않아(혹시 방법을 아시면 알려주시면 감사하겠습니다.) 최종적으로는 transformers만 이용할 예정

- Sim CSE를 이용한 한국어 문장 임베딩 모델입니다. https://huggingface.co/ddobokki/unsup-simcse-klue-roberta-small 에 공개가 되어 있습니다.

## Scenario
- 특정 도메인 관련 NLP 작업시 label이 없는 대량의 데이터를 받았을 경우를 상정함
- 분석 및 각종 Task에 기초가 되는 임베딩 벡터를 구할 수 있는가? 에 관한 실험
- 기존에 구할 수 있는 corpus(한국어 위키) + 받은 데이터(KorSTS, Klue-sts)가 내가 가지고 있는 total data라고 가정
  - sts 데이터는 특정 도메인이 관련된 데이터가 아니기 때문에 추가적인 데이터가 필요(ex 뉴스 데이터)
  - 최종적으로 한국어 위키데이터 + sts가 아닌 특정 도메인 데이터로 학습
  - 그렇게 진행 했을 경우에 sts 성능이 좋게 나오면 좋은 임베딩을 얻었다고 가정함

## train strategy
- unspervised 학습
- train: 한국어 위키 + KLUE sts-train + Kor sts-train (sentence pair를 통합)
- dev: Kor sts-dev
- test: Kor sts-test

## Performance
- KorSTS

| Model                  | Cosine Pearson | Cosine Spearman | Euclidean Pearson | Euclidean Spearman | Manhattan Pearson | Manhattan Spearman | Dot Pearson | Dot Spearman |
|------------------------|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| KoSRoBERTa<sup>small</sup>    | 84.27 | 84.17 | 83.33 | 83.65 | 83.34 | 83.65 | 82.10 | 81.38 |
| Unsup-SimCSE-RoBERTa<sup>small</sup>| 70.73 | 70.40 | 70.71 | 71.05 | 70.81 | 71.19 | 67.13 | 66.48 |

## References
```bibtex
@article{ham2020kornli,
  title={KorNLI and KorSTS: New Benchmark Datasets for Korean Natural Language Understanding},
  author={Ham, Jiyeon and Choe, Yo Joong and Park, Kyubyong and Choi, Ilji and Soh, Hyungjoon},
  journal={arXiv preprint arXiv:2004.03289},
  year={2020}
}
```
```bibtex
@misc{park2021klue,
      title={KLUE: Korean Language Understanding Evaluation},
      author={Sungjoon Park and Jihyung Moon and Sungdong Kim and Won Ik Cho and Jiyoon Han and Jangwon Park and Chisung Song and Junseong Kim and Yongsook Song and Taehwan Oh and Joohong Lee and Juhyun Oh and Sungwon Lyu and Younghoon Jeong and Inkwon Lee and Sangwoo Seo and Dongjun Lee and Hyunwoo Kim and Myeonghwa Lee and Seongbo Jang and Seungwon Do and Sunkyoung Kim and Kyungtae Lim and Jongwon Lee and Kyumin Park and Jamin Shin and Seonghyun Kim and Lucy Park and Alice Oh and Jungwoo Ha and Kyunghyun Cho},
      year={2021},
      eprint={2105.09680},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
