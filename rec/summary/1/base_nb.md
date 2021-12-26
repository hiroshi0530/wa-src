## 推薦システムアルゴリズムのまとめ 1


### github
- jupyter notebook形式のファイルは[こちら](https://github.com/hiroshi0530/wa-src/tree/master/rec/summary/1/base_nb.ipynb)

### google colaboratory
- google colaboratory で実行する場合は[こちら](https://colab.research.google.com/github/hiroshi0530/wa-src/tree/master/rec/summary/1/base_nb.ipynb)


最近仕事で推薦システムの開発に携わることがあり、その概要を調べたのでまとめてみる。

推薦システムは人によって呼び方やその定義が異なる事が多いため、[MicrosoftのGithub](https://github.com/microsoft/recommenders)に従って、まとめる。

### 協調フィルタリング

### コンテンツベースフィルタリング

### ハイブリッドフィルタリング


協調フィルタリングとコンテンツベースフィルタリングは補完関係にあるので、ハイブリッドで利用する


### 手法の説明

- matrix factrization
- factrization machine
- lightFM

MF・FMとの違い
ユーザー・アイテムembeddingを、ユーザー・アイテム特徴量から作成しない場合、つまりそれぞれにembeddingを用意してinteractionをとった場合、Matrix Factorizationになります。

また、Factorization Machinesは、全ての特徴量間のinteractionを考慮しますが、LightFMが考慮するのはuser, item間のinteractionのみです。

また、Factorization Machinesとは違い、ユーザー・アイテム特徴量からユーザー・アイテムembeddingを作成するという処理が行われています。

最後に
FMに関しても同じだが、embeddingの作成方法に関しても工夫が効く
sequencialな特徴量をpoolingして使うだったり転移学習したりできる

https://nnkkmto.hatenablog.com/entry/2020/12/20/000000

- node2vec
  - deep walk

## 推薦システムの評価手法



## 参考資料
- https://github.com/microsoft/recommenders
- https://techblog.zozo.com/entry/deep-learning-recommendation-improvement

