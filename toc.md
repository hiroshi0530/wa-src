# 記事案

人と同じ事をやっても意味がない！！！！

おれの記事でも閲覧が多いのは、ガンマポアソンリーセンシーで、他の汎用的な記事は閲覧はない！！！

珍しい題材の記事を作る必要がある。そうしないと記事を作っている意味がない！！！

これはJINさんも言っている！人が既にやっていることをやってもそこに付加価値はないのだ！！！

記事を書くにもどのような記事が必要かを考える
・アクセスがなくても、このサイトを見に来てくれた人がこの人は何が出来るのかを確認出来るもの
・SEOに引っかかる付加価値が高い記事

## 2024/3/31 までに書くブログ

- 研究の話やグラフの話など書けそうな部分からどんどん書いていく！

- Graphの話
    - [rec/graph/01] [グラフ理論]  評価行列、隣接行列、次数行列、ラプラシアン行列 (済)
    - [rec/graph/02] [グラフ理論]  ベクトル場とグラフの演算子の対比
        - http://gabarro.org/ccn/algebraic_graph_calculus.html
    - [rec/graph/03] [グラフ理論]  グラフラプラシアンと固有値、GFT
    - [rec/graph/04] [スペクトラルグラフ理論] ローパスフィルタの話
    - [rec/graph/05] [スペクトラルグラフ理論] チーガーの不等式の話
        - https://mathweb.ucsd.edu/~fan/research/revised.html
    - [rec/graph/06] [スペクトラルグラフ理論] 最短カットの話

- 線型代数の話
    - [rec/linalg/base] 低ランク近似 (済)
    - [rec/linalg/base2] 疑似逆行列 (済)
    - [rec/linalg/base3] 特異値分解と主成分分析
    - [rec/linalg/base4] レイリー商の話
    - [rec/linalg/base5] 余因子と余因子展開の話
    - [rec/linalg/base6] スペクトル分解の話 (済)
    - [rec/linalg/base7] 便利な公式の話
        - Matrix CookBook https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf


- Graph Convolutional Network (GCN) の話
    - https://www.slideshare.net/yukihirodomae/graph-convolution
    - 空間上の畳み込みとスペクトル上での畳み込みの話
    - [rec/gcn/01]
    - [rec/gcn/02]

- 推薦システムの評価手法と実装の話
    - [rec/gr/03] メモリベース協調フィルタリング
    - [rec/gr/04] 内容ベースフィルタリング
    - [rec/gr/05] 行列分解の話
    - [rec/gr/06] 評価手法の話

- 研究の話
    - [rec/qwalk/01] 量子ランダムウォークと隣接行列の話
    - [rec/qwalk/02] 2重確率行列の性質の話

- マーケティングの話
    - [rec/mkt/04] RFMとCPM分析の話 (済)
    - [rec/mkt/05] アソシエーション分析の話 (済)
      - mlxtendでのaprioriとfpgtree


- 数値計算の話
  - 経済のシミュレーションの話もしたい
  - 物理シミュレーション


- Simpyの話
    - [rec/simpy/01] 離散シミュレーションの基礎





- GNN
- SQL
- J-Quants
- 統計
  - ガンマ関数とベータ関数のプロット
- MMM
- 時系列
- ベイズ



## 忘れないようにメモ
- 211007
- rec
  - 量子ウォーク
  - 量子低ランク近似推定
  - 量子プロジェクション
  - prakashの博士論文
  - データ木構造を使った振幅エンコーディング(量子RAM)
    - prakashや教科書から


- 211007
  - ガンマ関数とベータ関数のプロット
  - sklearnのmake_datasetsのclassificationのアルゴリズム解説
  - 確率密度関数の相関図
  - pandasの更新
    - df.apply()
    - df.diff()
    - df.rolling()
    - df.pct_change()
    - df.pivot()
    - pd.get_dummies()
  - pandasやnumpyのビューとコピーについて
    - XGBoost
    - LightGBM
    - RandomForest

- デバイス
  - 縦軸に研究、開発、生産、横軸に、デバイスの大きさの表
  - デバイスの抽象化の話（適用できる物理方程式がデバイスの大きさによって変化する）
  - 電子、トランジスタ、ADCなどなど、電子デバイスはそれぞれ抽象化されていて、上の階層の人が下の部分を知らなくてもいいように出来ている⇒ソフトウェアも同じ
  - ハードウェア、OS、ソフトも同じ
  - ソフトのクラスの階層構造も同じ

- CPUチップの歴史
  - MACのb1、M1などのシリーズを解説

- 量子コンピュータ業界の現状
  - 方式など

- 古典の論理ゲートとの差異を追記
  - https://whyitsso.net/physics/quantum_mechanics/QI2.html


- 半導体業界、IT業界の業界地図みたいなものを書きたい
- 液晶業界
- IT業界

1. マスク製造工程
2. 前行程
3. 後工程
4. 検査工程
5. EDAツール
6. 装置

- lib
  - python
  - numpy
    - 基本 : OK
    - 三角関数 : OK
    - 指数対数 : OK
    - 統計 : OK
    - 線形代数 : OK
    - サンプリング : OK
    - その他
      - 転置行列
  - pandas : OK
  - matplotlib
    - 二次元 : OK
    - 三次元 : OK
  - scikit-learn
    - datasets : OK
    - make datasets
    - linear regression : OK
    - logistic regression : OK
    - Random forest
    - XGBoost regression
    - Light XGBoost regression
  - bash
    - other : OK
  - tensorflow
    - チュートリアル
    - tensorflow2.0
      - 実施例
      - 利用方法解説
    - tensorflow quantum
    - tensorflow probablity
  - pytorch

- cloud
  - gcp
  - aws : OK
    - server : OK
    - serverless : OK
    - sagemaker : OK
  - terraform
  - システム設計
    - DB
      - CAPの定理
      - 冗長構成
    - Infrastructure as a code
    - Blue Green Switch
    - CI/CD

- ML
  - 深層学習
    - 自然言語処理
      - word2vec
      - BERT
      - XLNet
      - ALBERT
      - RNN
      - LSTM
      - BERT
      - Transform
  - ベイズ統計
    - R + Stan
    - 統計モデリング
      - https://www.bigdata-navi.com/aidrops/2925/
    - tensorflow probability
    - ガウス過程
  - XGBT
  - トピックモデル

- QC
  - ITエンジニアのための量子コンピューター
    - 量子力学の入門
  - 湊さんのyoutubeと記事を全部やってみた
  - Bluecat, Cirq, Qiskitの比較
    - https://sjsy.hatenablog.com/entry/20190929/1569762456

- 論文解説
  - BERT
  - Transform

- julia入門
  - 須山さんの本の例題がjuliaで書かれているので、それをやってみる。
  - https://github.com/sammy-suyama/BayesBook

- 線形代数
  - http://www.nct9.ne.jp/m_hiroi/light/julia.html

- GA（遺伝的アルゴリズム
