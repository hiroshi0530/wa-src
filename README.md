# wayama.io — 記事のサンプルコード / Article examples

[日本語](#japanese) | [English](#english)

<a id="japanese"></a>

## 日本語

[wayama.io](https://wayama.io/) で公開している記事のサンプルコード集です。機械学習、数値計算、統計、量子計算などのコードを、主に Jupyter Notebook 形式で収録しています。

### 環境構築

このリポジトリでは、Python 環境の管理に **uv** を利用します。

Python 3.14.6 の自動取得を含む初回構築には、CI と同じ **uv 0.13.7** を使用してください。

- [uv 公式サイト](https://docs.astral.sh/uv/)
- [uv のインストール手順](https://docs.astral.sh/uv/getting-started/installation/)

#### Python 環境

共通環境では **Python 3.14.6** を使います。設定ファイルはリポジトリのルートに置いています。

| ファイル | 役割 |
| --- | --- |
| [`.python-version`](.python-version) | 使用する Python のバージョン |
| [`pyproject.toml`](pyproject.toml) | Python の対応範囲と、共通ライブラリの定義 |
| [`uv.lock`](uv.lock) | 依存ライブラリの解決済みバージョン |

共通環境には以下のライブラリが含まれます。

| 用途 | 主なライブラリ |
| --- | --- |
| Notebook の実行・環境確認 | JupyterLab、IPython、ipykernel、ipywidgets、watermark |
| 数値計算・データ処理 | NumPy、SciPy、pandas、Polars |
| 可視化 | Matplotlib、seaborn |
| 数式処理・機械学習・グラフ | SymPy、scikit-learn、NetworkX |

共通環境で実行確認した Notebook は、後述の「Notebook 一覧」に記載しています。追加ライブラリや過去の API を使用する例は、「個別の環境が必要な例」を参照してください。

#### 初回の準備

Git と uv をインストールした後、リポジトリを取得します。

```bash
git clone https://github.com/hiroshi0530/wa-src.git
cd wa-src
```

リポジトリのルートで、共通環境を作成します。

```bash
uv sync --locked
```

ルートに `.venv/` が作成され、必要なライブラリがインストールされます。指定された Python が見つからない場合は、uv が取得します。`--locked` は設定と `uv.lock` が一致しない場合にエラーにし、ロックファイルを自動更新しないための指定です。

以降の `uv` コマンドもリポジトリのルートで実行してください。仮想環境の手動有効化は不要です。

#### VS Code を利用する場合

1. Microsoft の **Python** 拡張機能と **Jupyter** 拡張機能をインストールします。
2. リポジトリのルートフォルダーを VS Code で開きます。
3. `.ipynb` ファイルを開き、右上のカーネル選択から **Python Environments** を選び、ルートの `.venv/` の Python を指定します。
4. [main_314.ipynb](main_314.ipynb) を上から順に実行し、Python のパスとライブラリのバージョンを確認します。

`.venv/` を直接選ぶ場合、カーネル登録や JupyterLab の起動は不要です。Python ファイルを実行する場合も、`Python: Select Interpreter` で同じ環境を選びます。

名前でカーネルを選びたい場合は、次の「JupyterLab を利用する場合」の登録コマンドを実行し、**Jupyter Kernels → wayama.io (Python 3.14)** を選んでください。

#### JupyterLab を利用する場合

初回に、共通環境のカーネルを登録します。

```bash
uv run --locked python -m ipykernel install --user --name wa-src --display-name "wayama.io (Python 3.14)"
```

JupyterLab を起動します。

```bash
uv run --locked jupyter lab
```

ブラウザーで [main_314.ipynb](main_314.ipynb) を開き、カーネルに **wayama.io (Python 3.14)**（登録名: `wa-src`）を選びます。共通環境を使うほかの Notebook でも同じカーネルを選び、セルを上から順に実行してください。

終了するときは、起動したターミナルで `Ctrl+C` を押し、終了確認に従います。リポジトリを移動したり `.venv/` を作り直したりした場合は、カーネルを再登録してください。

### 利用するデータセット

環境確認・SymPy・統計・線形代数の動作確認用 Notebook は外部データを必要としません。その他の例で使う主なデータと保存先は以下のとおりです。

| 対象 Notebook | 入手先・データ | 保存先と準備 |
| --- | --- | --- |
| [pandas の基本操作](article/library/pandas/01/pandas_nb.ipynb) | [pandas for everyone](https://github.com/chendaniely/pandas_for_everyone/blob/master/data/country_timeseries.csv) の `country_timeseries.csv` | `article/library/pandas/01/country_timeseries.csv` に同梱 |
| [言語処理 100 本ノック 第2章](article/ml/nlp100/02/02_nb.ipynb) | [2020年版 第2章](https://nlp100.github.io/2020/ja/ch02.html) の `popular-names.txt` | `article/ml/nlp100/02/popular-names.txt` に同梱 |
| [言語処理 100 本ノック 第3章](article/ml/nlp100/03/03_nb.ipynb) | [2020年版 第3章](https://nlp100.github.io/2020/ja/ch03.html) の `jawiki-country.json.gz` | 別途取得して展開し、`article/ml/nlp100/03/wiki.json` として保存 |
| [言語処理 100 本ノック 第4章](article/ml/nlp100/04/04_nb.ipynb) | [2020年版 第4章](https://nlp100.github.io/2020/ja/ch04.html) の `neko.txt` | `article/ml/nlp100/04/` に `neko.txt` と `neko.txt.mecab` を同梱。解析の再実行には MeCab と辞書などが必要 |

第3章では、配布元の圧縮ファイルを展開した後のファイル名を `wiki.json` に変更してください。Notebook 内の古いダウンロード先が使えない場合は、上の配布ページから取得し、ダウンロード用セルをスキップしてください。

その他の CSV・画像・データベースは、各記事・Notebook の読み込みパスに合わせて配置します。外部データには配布元の利用条件が適用されます。

### Notebook 一覧

#### 環境確認と実行確認済みの例

以下は、2026-11-11 に **macOS 26.5.2 / Python 3.14.6 / uv 0.11.6** と、このリポジトリの `uv.lock` を使って全コードセルを実行確認したものです。

| 分野 | Notebook | 内容・ブログ記事 | 実行環境 |
| --- | --- | --- | --- |
| 環境確認 | [main_314.ipynb](main_314.ipynb) | Python のパス、共通ライブラリのインポート・バージョン、計算とグラフ表示 | 共通環境 |
| 数式処理 | [article/library/sympy/01/01_nb.ipynb](article/library/sympy/01/01_nb.ipynb) | [SymPy による数式処理](https://wayama.io/article/library/sympy/01/) | 共通環境 |
| 統計 | [rec/stats/01/01_nb.ipynb](rec/stats/01/01_nb.ipynb) | [ガンベル分布](https://wayama.io/rec/stats/01/) | 共通環境 |
| 線形代数 | [rec/linalg/base/base_nb.ipynb](rec/linalg/base/base_nb.ipynb) | [特異値分解と低ランク近似](https://wayama.io/rec/linalg/base/) | 共通環境 |

#### 分野から探す

[`article/`](article/) はライブラリの使い方や開発・実装の例、[`rec/`](rec/) は数理・アルゴリズムの解説を中心に収録しています。各ディレクトリ内の全 Notebook を共通環境で検証したものではありません。

| 分野 | ディレクトリ | 主な内容 |
| --- | --- | --- |
| ライブラリ・開発の基礎 | [`article/library/`](article/library/) | NumPy、pandas、SymPy、SQL、Bash など |
| 機械学習 | [`article/ml/`](article/ml/) | 自然言語処理、ベイズ推論など |
| 線形代数 | [`rec/linalg/`](rec/linalg/) | 行列、特異値分解、低ランク近似など |
| 統計 | [`rec/stats/`](rec/stats/) | 確率分布と統計の実装例 |
| グラフ・ランキング | [`rec/graph/`](rec/graph/)、[`rec/gcn/`](rec/gcn/)、[`rec/pagerank/`](rec/pagerank/) | グラフ、グラフ畳み込み、PageRank |
| 量子計算 | [`rec/qiskit/`](rec/qiskit/)、[`rec/qwalk/`](rec/qwalk/) | 量子回路、量子アルゴリズム、量子ウォーク |
| マーケティング | [`article/mkt/`](article/mkt/)、[`rec/mkt/`](rec/mkt/) | マーケティングに関する数理・分析例 |

[NumPy の基本的な演算](https://wayama.io/article/library/numpy/base/)の [Notebook](article/library/numpy/base/base_nb.ipynb) など、過去の記事の例も収録しています。保存された出力やメタデータのバージョンは、記事作成時の環境を示します。

#### 個別の環境が必要な例

次のライブラリは共通環境に含めていません。各記事の環境と API を確認し、対応する Python・ライブラリの組み合わせを用意してください。

| 対象 | 追加で確認する内容 |
| --- | --- |
| [TensorFlow](article/library/tf/) | TensorFlow、TensorFlow Datasets など。記事によって使用するデータセットや API が異なる |
| [PyTorch](article/library/torch/)・[深層学習の例](article/ml/lec/text/) | PyTorch に加え、Transformers などの追加ライブラリが必要な例がある |
| [Qiskit](rec/qiskit/) | `from qiskit import execute, Aer` や `qiskit_textbook` を使う旧 API の例を含む |
| [Stan](article/ml/bayse/pystan/base/base_nb.ipynb) | `import pystan` と `pystan.stan(...)` を使う旧 API の例 |
| [OpenCV](article/library/opencv/)・[形態素解析](article/ml/nlp100/04/04_nb.ipynb) | OpenCV、MeCab・辞書、日本語表示用ライブラリなど |

これらの例については、Python 3.14 の共通環境での実行を確認していません。古い依存関係が必要な場合は、共通環境とは別の uv プロジェクトで Python と依存ライブラリを固定してください。

### 実行時の補足

- **作業ディレクトリ：** JupyterLab は通常、Notebook のあるフォルダーを作業ディレクトリとして実行します。`FileNotFoundError` が出る場合は、ファイルの配置と相対パスを確認してください。
- **OS の違い：** `!sw_vers` は macOS の情報表示用です。Linux / Windows ではそのセルをスキップしてください。Bash や言語処理の演習で使う `wget`・`sed` などの外部コマンドは uv の管理対象外です。
- **追加・更新：** 共通環境のライブラリを追加するときは、ルートで `uv add ライブラリ名` を使い、`pyproject.toml` と `uv.lock` を一緒に更新してください。Notebook 内の `!pip install` や `%pip install` ではロックファイルが更新されません。
- **動作確認の範囲：** 上記 4 本は macOS と Linux ARM64 で実行確認済みです。Windows での実行、追加ライブラリが必要な例、全 Notebook の互換性は未検証です。

### CI とローカルでの検証

[GitHub Actions](.github/workflows/ci.yml) は、`develop`・`master` への push、Pull Request、手動実行で動作します。Ubuntu 上で `.python-version` と `uv.lock` に従って環境を作成し、[tests/test_notebooks.py](tests/test_notebooks.py) に定義した環境確認・SymPy・統計・線形代数の 4 本を個別に実行します。

2026-11-11 に macOS と Linux ARM64 コンテナで 4 テストの成功を確認しました。Linux では uv 0.13.7 による Python 3.14.6 の取得から検証しています。

ローカルでも同じテストをまとめて実行できます。

```bash
uv sync --locked
uv run --locked pytest -q tests --basetemp .pytest-notebooks
```

テストは一時ディレクトリで Notebook を実行し、元のファイルを変更しません。macOS 以外では `!sw_vers` だけを含む情報表示セルをスキップし、計算中のエラーやセルごとの 180 秒のタイムアウトで失敗します。カーネルはテスト用に準備するため、手動登録は不要です。

実行結果は `.pytest-notebooks/` に保存されます。GitHub Actions では、失敗した場合も実行済み Notebook を成果物として 7 日間保存します。テスト対象を追加するときは、テストファイルの `NOTEBOOKS` と CI の `matrix.notebook` を更新してください。

### 不具合の報告

コードや説明の不具合は [GitHub Issues](https://github.com/hiroshi0530/wa-src/issues) に報告してください。対象の Notebook のパス、再現手順、OS、`uv --version`、[main_314.ipynb](main_314.ipynb) の出力、エラーメッセージがあると確認しやすくなります。

### ライセンス

本リポジトリのうち、著者が権利を有するコードおよび文章は、[CC0 1.0 Universal](LICENSE) のもとで公開しています。商用・非商用を問わず、自由に利用・複製・改変・再配布できます。著作権表示や利用の連絡は不要です。無保証で提供します。

第三者が権利を有するコード・画像・データセットなどには、それぞれの利用条件が適用されます。正式な本文は [LICENSE](LICENSE) を参照してください。

<a id="english"></a>

## English

This repository contains code examples for articles published on [wayama.io](https://wayama.io/en/), primarily as Jupyter Notebooks. Topics include machine learning, numerical computing, statistics, and quantum computing.

### Environment setup

This repository uses **uv** to manage its Python environment.

For an initial setup that also downloads Python 3.14.6, use **uv 0.13.7**, matching CI.

- [uv documentation](https://docs.astral.sh/uv/)
- [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/)

#### Python environment

The shared environment uses **Python 3.14.6**. Its configuration files are at the repository root.

| File | Purpose |
| --- | --- |
| [`.python-version`](.python-version) | Selected Python version |
| [`pyproject.toml`](pyproject.toml) | Supported Python versions and shared dependencies |
| [`uv.lock`](uv.lock) | Resolved dependency versions |

The shared environment includes the following libraries.

| Purpose | Main libraries |
| --- | --- |
| Notebook execution and environment checks | JupyterLab, IPython, ipykernel, ipywidgets, watermark |
| Numerical computing and data processing | NumPy, SciPy, pandas, Polars |
| Visualization | Matplotlib, seaborn |
| Symbolic mathematics, machine learning, and graphs | SymPy, scikit-learn, NetworkX |

Notebooks verified in this environment are listed below. See “Examples requiring a separate environment” for examples using additional libraries or historical APIs.

#### Initial setup

Install Git and uv, then clone the repository.

```bash
git clone https://github.com/hiroshi0530/wa-src.git
cd wa-src
```

Create the shared environment from the repository root.

```bash
uv sync --locked
```

This creates `.venv/` at the root and installs the required libraries. uv downloads the selected Python version if it is not available locally. `--locked` reports an error if the configuration and `uv.lock` disagree, instead of updating the lockfile automatically.

Run the following `uv` commands from the repository root as well. Manual environment activation is not required.

#### Using VS Code

1. Install Microsoft's **Python** and **Jupyter** extensions.
2. Open the repository root folder in VS Code.
3. Open an `.ipynb` file, choose **Python Environments** in the kernel picker, and select the Python interpreter in the root `.venv/`.
4. Run [main_314.ipynb](main_314.ipynb) from top to bottom to check the Python path and library versions.

When selecting `.venv/` directly, kernel registration and a running JupyterLab server are unnecessary. For Python files, select the same environment with `Python: Select Interpreter`.

To select a named kernel instead, run the registration command in the next section and choose **Jupyter Kernels → wayama.io (Python 3.14)**.

#### Using JupyterLab

Register the shared environment's kernel once.

```bash
uv run --locked python -m ipykernel install --user --name wa-src --display-name "wayama.io (Python 3.14)"
```

Start JupyterLab.

```bash
uv run --locked jupyter lab
```

Open [main_314.ipynb](main_314.ipynb) in the browser and select **wayama.io (Python 3.14)** (kernel name: `wa-src`). Select the same kernel for other Notebooks using the shared environment, then run their cells from top to bottom.

To stop JupyterLab, press `Ctrl+C` in the terminal where it was started and follow the shutdown prompt. Register the kernel again if you move the repository or recreate `.venv/`.

### Datasets

The environment check and the verified SymPy, statistics, and linear algebra examples need no external data. Some other examples use the following datasets.

| Notebook | Source and data | Location and preparation |
| --- | --- | --- |
| [pandas basics](article/library/pandas/01/pandas_nb.ipynb) | `country_timeseries.csv` from [pandas for everyone](https://github.com/chendaniely/pandas_for_everyone/blob/master/data/country_timeseries.csv) | Included at `article/library/pandas/01/country_timeseries.csv` |
| [NLP 100, chapter 2](article/ml/nlp100/02/02_nb.ipynb) | `popular-names.txt` from [chapter 2 of the 2020 edition](https://nlp100.github.io/2020/ja/ch02.html) | Included at `article/ml/nlp100/02/popular-names.txt` |
| [NLP 100, chapter 3](article/ml/nlp100/03/03_nb.ipynb) | `jawiki-country.json.gz` from [chapter 3 of the 2020 edition](https://nlp100.github.io/2020/ja/ch03.html) | Download, decompress, and save as `article/ml/nlp100/03/wiki.json` |
| [NLP 100, chapter 4](article/ml/nlp100/04/04_nb.ipynb) | `neko.txt` from [chapter 4 of the 2020 edition](https://nlp100.github.io/2020/ja/ch04.html) | `neko.txt` and `neko.txt.mecab` are included in `article/ml/nlp100/04/`; rerunning the analysis also requires MeCab, a dictionary, and related libraries |

For chapter 3, rename the decompressed file to `wiki.json`. If an old download URL in a Notebook no longer works, use the source page above and skip the download cell.

Place other CSV files, images, and databases at the paths used by the relevant article or Notebook. External datasets remain subject to their providers' terms.

### Notebook index

#### Environment check and verified examples

All code cells in the following Notebooks were executed on **2026-11-11 using macOS 26.5.2, Python 3.14.6, uv 0.11.6**, and this repository's `uv.lock`.

| Topic | Notebook | Contents and article | Environment |
| --- | --- | --- | --- |
| Environment check | [main_314.ipynb](main_314.ipynb) | Python path, shared library imports and versions, calculation, and plotting | Shared |
| Symbolic mathematics | [article/library/sympy/01/01_nb.ipynb](article/library/sympy/01/01_nb.ipynb) | [Symbolic Mathematics with SymPy](https://wayama.io/en/article/library/sympy/01/) | Shared |
| Statistics | [rec/stats/01/01_nb.ipynb](rec/stats/01/01_nb.ipynb) | [Gumbel Distribution](https://wayama.io/en/rec/stats/01/) | Shared |
| Linear algebra | [rec/linalg/base/base_nb.ipynb](rec/linalg/base/base_nb.ipynb) | [Singular Value Decomposition and Low Rank Approximation](https://wayama.io/en/rec/linalg/base/) | Shared |

Notebook text may be in Japanese even when an English article is available.

#### Browse by topic

[`article/`](article/) mainly contains library usage and development examples. [`rec/`](rec/) focuses on mathematics and algorithms. Not every Notebook in these directories has been tested in the shared environment.

| Topic | Directory | Contents |
| --- | --- | --- |
| Libraries and development basics | [`article/library/`](article/library/) | NumPy, pandas, SymPy, SQL, Bash, and more |
| Machine learning | [`article/ml/`](article/ml/) | Natural language processing, Bayesian inference, and more |
| Linear algebra | [`rec/linalg/`](rec/linalg/) | Matrices, singular value decomposition, low-rank approximation, and more |
| Statistics | [`rec/stats/`](rec/stats/) | Probability distributions and statistical examples |
| Graphs and ranking | [`rec/graph/`](rec/graph/), [`rec/gcn/`](rec/gcn/), [`rec/pagerank/`](rec/pagerank/) | Graphs, graph convolutions, and PageRank |
| Quantum computing | [`rec/qiskit/`](rec/qiskit/), [`rec/qwalk/`](rec/qwalk/) | Quantum circuits, algorithms, and quantum walks |
| Marketing | [`article/mkt/`](article/mkt/), [`rec/mkt/`](rec/mkt/) | Mathematical models and analysis for marketing |

Older examples include the [basic NumPy operations article (Japanese)](https://wayama.io/article/library/numpy/base/) and its [Notebook](article/library/numpy/base/base_nb.ipynb). Versions in saved outputs and metadata describe the environment used when the article was written.

#### Examples requiring a separate environment

The following libraries are not included in the shared environment. Check each article's environment and APIs to determine the required Python and library versions.

| Examples | Additional requirements to check |
| --- | --- |
| [TensorFlow](article/library/tf/) | TensorFlow, TensorFlow Datasets, and other packages; datasets and APIs vary by article |
| [PyTorch](article/library/torch/) and [deep learning](article/ml/lec/text/) | PyTorch; some examples also require Transformers and other libraries |
| [Qiskit](rec/qiskit/) | Includes historical APIs such as `from qiskit import execute, Aer` and `qiskit_textbook` |
| [Stan](article/ml/bayse/pystan/base/base_nb.ipynb) | Uses the historical `import pystan` and `pystan.stan(...)` APIs |
| [OpenCV](article/library/opencv/) and [morphological analysis](article/ml/nlp100/04/04_nb.ipynb) | OpenCV, MeCab and a dictionary, Japanese font support, and related packages |

These examples have not been verified in the shared Python 3.14 environment. When historical dependencies are needed, pin Python and library versions in a separate uv project.

### Execution notes

- **Working directory:** JupyterLab normally uses the Notebook's folder as its working directory. If you get `FileNotFoundError`, check the file location and relative path.
- **OS differences:** `!sw_vers` displays macOS information; skip that cell on Linux / Windows. External commands such as `wget` and `sed`, used in shell and NLP exercises, are not managed by uv.
- **Adding or updating libraries:** Use `uv add PACKAGE` from the root when adding a shared dependency, and update `pyproject.toml` and `uv.lock` together. Running `!pip install` or `%pip install` inside a Notebook does not update the lockfile.
- **Verification scope:** The four examples above have been verified on macOS and Linux ARM64. Windows execution, examples requiring additional libraries, and compatibility of every Notebook have not been verified.

### CI and local verification

[GitHub Actions](.github/workflows/ci.yml) runs on pushes to `develop` and `master`, pull requests, and manual dispatch. On Ubuntu, it creates the environment from `.python-version` and `uv.lock`, then independently executes the environment check, SymPy, statistics, and linear algebra Notebooks defined in [tests/test_notebooks.py](tests/test_notebooks.py).

All four tests passed on macOS and in a Linux ARM64 container on 2026-11-11. The Linux verification included downloading Python 3.14.6 with uv 0.13.7.

Run the same tests together locally with:

```bash
uv sync --locked
uv run --locked pytest -q tests --basetemp .pytest-notebooks
```

The tests execute Notebooks in temporary directories without modifying the originals. On non-macOS systems, only standalone `!sw_vers` information cells are skipped. Execution errors and a per-cell timeout of 180 seconds fail the test. A private test kernel is created automatically; manual kernel registration is unnecessary.

Executed Notebooks are saved under `.pytest-notebooks/`. GitHub Actions uploads these outputs even on failure and retains them for 7 days. To add a test case, update `NOTEBOOKS` in the test file and `matrix.notebook` in the workflow.

### Reporting issues

Report problems with code or explanations through [GitHub Issues](https://github.com/hiroshi0530/wa-src/issues). Include the Notebook path, steps to reproduce, OS, `uv --version`, the output of [main_314.ipynb](main_314.ipynb), and any error messages.

### License

The code and text in this repository for which the author holds the rights are released under [CC0 1.0 Universal](LICENSE). You may freely use, copy, modify, and redistribute this material for commercial or non-commercial purposes. No attribution or notification is required. The material is provided without warranty.

Third-party code, images, datasets, and other materials remain subject to their respective terms. See [LICENSE](LICENSE) for the full legal text.
