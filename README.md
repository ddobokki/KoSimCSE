# KoSimCSE
Korean SimCSE using PLM in huggingface hub

- sentence-transformers 라이브러리를 이용해 1차 작업중
  - 해당 라이브러리는 Multi Gpu를 지원하지 않아(혹시 방법을 아시면 알려주시면 감사하겠습니다.) 최종적으로는 transformers만 이용할 예정
- train strategy
  - unspervised 학습
  - train: 한국어 위키 + KLUE sts-train + Kor sts-train (sentence pair를 통합)
  - dev: Kor sts-dev
  - test: Kor sts-test

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
