# SVDのアルゴリズムを用いる
from surprise import SVD
# モデル評価用
from surprise.model_selection import cross_validate

# データセット
# よくある MovieLense
from surprise import Dataset
data = Dataset.load_builtin('ml-100k')

# モデルをつくる
svd = SVD(n_factors=16, n_epochs=50, biased=False)

# モデルの検証(rmse,maeで5クロスバリデーション
cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# fit
svd.fit(data)
