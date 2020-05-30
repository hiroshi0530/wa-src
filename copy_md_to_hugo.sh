#!/bin/bash

author='whiroshi0530'
weight='1'
yyyy_mm_dd=`date -u +%Y-%m-%d`

_temp_raw_file_name="_temp_raw_index.md"
_temp_orig_file_name="_temp_orig_index.md"

copy_file () {
  
  top=$1
  title=$2
  file_name=$3
  src_sub_dir=$4
  categories=$5
  tags=$6
  keywords=$7
  weight=$8

  # 汎用的に使いたい
  static_top_dir="../wa/static/content/"${top}"/"${src_sub_dir}

  # src_sub_dirがない場合はret
  if [ ! -f "./"${top}"/"${src_sub_dir}${file_name}"_nb.md" ]; then return 0; fi

  # static ディレクトリがない場合作成し、仮のindex.mdファイルを作成 
  if [ ! -d ${static_top_dir} ]; then mkdir -p ${static_top_dir} && touch ${static_top_dir}index.md; fi

  png_dir="${file_name}"_nb_files
  png_dir_local="${file_name}"_nb_files_local

  md_file="./"${top}"/"${src_sub_dir}${file_name}"_nb.md"
  png_files="./"${top}"/"${src_sub_dir}${png_dir}"/*.png"
  svg_files="./"${top}"/"${src_sub_dir}${png_dir}"/*.svg"
  png_files_local="./"${top}"/"${src_sub_dir}${png_dir_local}"/*.png"
  svg_files_local="./"${top}"/"${src_sub_dir}${png_dir_local}"/*.svg"
  prefix="./"${top}"/"${src_sub_dir}${file_name}"_nb.prefix"

  # 1. prefixの作成
  cp ./prefix_template ${prefix}

  sed -i -E 's/__title__/'${title}'/g' ${prefix}
  sed -i -E 's/__yyyy-mm-dd__/'${yyyy_mm_dd}'/g' ${prefix}
  sed -i -E 's/__author__/'${author}'/g' ${prefix}
  sed -i -E 's/__weight__/'${weight}'/g' ${prefix}
  sed -i -E 's/__categories__/'${categories}'/g' ${prefix}
  sed -i -E 's/__tags__/'${tags}'/g' ${prefix}
  sed -i -E 's/__keywords__/'${keywords}'/g' ${prefix}
  sed -i -E 's/__weight__/'${weight}'/g' ${prefix}

  # 2. prefixと本体の結合
  cat ${prefix} ${md_file} > ${md_file}_cat

  # 3. Markdownのファイル読み込み部の置換
  # 3-1. matplolibなどで自動的に作成した画像ファイル
  sed -i"" -E 's/^\!\[png\]('${file_name}'_nb_files\/\(.*\).png)$/{{<figure src="\1.png" class="center">}}/' ${md_file}_cat
  sed -i"" -E 's/^\!\[png\]('${file_name}'_nb_files\/\(.*\).svg)$/{{<figure src="\1.svg" class="center">}}/' ${md_file}_cat
  sed -i"" -E 's/^\!\[svg\]('${file_name}'_nb_files\/\(.*\).png)$/{{<figure src="\1.png" class="center">}}/' ${md_file}_cat
  sed -i"" -E 's/^\!\[svg\]('${file_name}'_nb_files\/\(.*\).svg)$/{{<figure src="\1.svg" class="center">}}/' ${md_file}_cat

  # 3-2. localで作成した画像ファイル
  sed -i"" -E 's/^\!\[png\]('${file_name}'_nb_files_local\/\(.*\).png)$/{{<figure src="\1.png" class="center">}}/' ${md_file}_cat
  sed -i"" -E 's/^\!\[png\]('${file_name}'_nb_files_local\/\(.*\).svg)$/{{<figure src="\1.svg" class="center">}}/' ${md_file}_cat
  sed -i"" -E 's/^\!\[svg\]('${file_name}'_nb_files_local\/\(.*\).png)$/{{<figure src="\1.png" class="center">}}/' ${md_file}_cat
  sed -i"" -E 's/^\!\[svg\]('${file_name}'_nb_files_local\/\(.*\).svg)$/{{<figure src="\1.svg" class="center">}}/' ${md_file}_cat

  # ipynbファイルはpngを、mdはsvgファイルにするために、ipynbからコピーする部分はsvgファイルにする
  sed -i"" -E 's/\.png/\.svg/g' ${md_file}_cat

  # 4. 差分がある場合のみコピーしたいので、差分をチェック
  cp ${md_file}_cat ${_temp_raw_file_name}   # コピーしようとするファイル
  cp ${static_top_dir}index.md ${_temp_orig_file_name}  # 上書きしようとするファイル

  # 上から11行を削除
  sed -i -E '1,11d' ${_temp_raw_file_name}
  sed -i -E '1,11d' ${_temp_orig_file_name}

  _temp_raw_hash=`md5 ${_temp_raw_file_name} | awk '{print $4}'`
  _temp_orig_hash=`md5 ${_temp_orig_file_name} | awk '{print $4}'`

  # 5. hashが一致した場合のみstatic site部へのコピー
  if [[ $_temp_raw_hash != $_temp_orig_hash ]]; then
    echo 'modified file : '$2
    rm -f ${static_top_dir}*.png
    rm -f ${static_top_dir}*.svg
    
    # 空白対策で入れている@をここで空白に変換
    sed -i -E 's/__@__/ /g' ${md_file}_cat 

    cp ${md_file}_cat ${static_top_dir}index.md 2> /dev/null
    cp ${png_files} ${static_top_dir} 2> /dev/null
    cp ${svg_files} ${static_top_dir} 2> /dev/null
    cp ${png_files_local} ${static_top_dir} 2> /dev/null
    cp ${svg_files_local} ${static_top_dir} 2> /dev/null
  fi

  # 6. 余計なファイルの削除
  rm -rf ${md_file}_cat
  rm -rf ${md_file}_cat-E
  rm -rf ${prefix}
  rm -rf ${prefix}-E
  rm -rf ${_temp_raw_file_name}
  rm -rf ${_temp_raw_file_name}-E
  rm -rf ${_temp_orig_file_name}
  rm -rf ${_temp_orig_file_name}-E

}


main () {

  # argument
  # top=$1
  # title=$2
  # file_name=$3
  # src_sub_dir=$4
  # categories=$5
  # tags=$6
  # keywords=$7

  ########################################################################
  # pandas
  copy_file 'article/library/pandas' '[pandas]__@__snipet' 'pandas' '' '["library","pandas"]' '["pandas"]' '["pandas","入門","使い方"]' '60'


  ########################################################################
  # sklearn
  copy_file 'article/library/sklearn' '[scikit-learn]__@__1.__@__datasets' 'ds' 'datasets/' '["library","scikit-learn"]' '["scikit-learn","sklearn","linear_regression","線形回帰"]' '["scikit-learn","sklearn","linear_regression","線形回帰","入門","使い方"]' '130'
  copy_file 'article/library/sklearn' '[scikit-learn]__@__2.__@__make__@__datasets' 'md' 'makedatas/' '["library","scikit-learn"]' '["scikit-learn","sklearn","linear_regression","データ作成"]' '["scikit-learn","sklearn","linear_regression","データ","作成","入門","使い方"]' '131'
  copy_file 'article/library/sklearn' '[scikit-learn]__@__3.__@__線形回帰' 'lr' 'linear_regression/' '["library","scikit-learn"]' '["scikit-learn","sklearn","linear_regression","線形回帰"]' '["scikit-learn","sklearn","linear_regression","線形回帰","入門","使い方"]' '132'
  copy_file 'article/library/sklearn' '[scikit-learn]__@__4.__@__ロジスティック回帰' 'lr' 'logistic_regression/' '["library","scikit-learn"]' '["scikit-learn","sklearn","logistic_regression","ロジスティック回帰"]' '["scikit-learn","sklearn","logistic_regression","ロジスティック回帰","入門","使い方"]' '133'



  ########################################################################
  # scipy
  copy_file 'article/library/scipy' '[scipy]__@__1.__@__distributions' 'dist' 'dist/' '["library","scipy"]' '["scipy"]' '["scipy","distributions","確率分布"]' '150'

  ########################################################################
  # numpy
  copy_file 'article/library/numpy' '[numpy]__@__1.__@__基本的な演算' 'base' 'base/' '["library","numpy"]' '["numpy"]' '["numpy","入門","使い方"]' '120'
  copy_file 'article/library/numpy' '[numpy]__@__2.__@__三角関数' 'trigonometric' 'trigonometric/' '["library","numpy"]' '["numpy","三角関数"]' '["numpy","入門","使い方","行列","三角関数"]' '121'
  copy_file 'article/library/numpy' '[numpy]__@__3.__@__指数・対数' 'explog' 'explog/' '["library","numpy"]' '["numpy","指数対数"]' '["numpy","入門","使い方","行列","指数","対数"]' '122'
  copy_file 'article/library/numpy' '[numpy]__@__4.__@__統計関数' 'statistics' 'statistics/' '["library","numpy"]' '["numpy","統計"]' '["numpy","入門","使い方","行列","統計"]' '123'
  copy_file 'article/library/numpy' '[numpy]__@__5.__@__線形代数' 'matrix' 'matrix/' '["library","numpy"]' '["numpy","行列","線形"]' '["numpy","入門","使い方","行列","線形代数"]' '124'
  copy_file 'article/library/numpy' '[numpy]__@__6.__@__サンプリング' 'sampling' 'sampling/' '["library","numpy"]' '["numpy","sampling"]' '["numpy","入門","使い方"]' '125'
  copy_file 'article/library/numpy' '[numpy]__@__7.__@__その他' 'misc' 'misc/' '["library","numpy"]' '["numpy"]' '["numpy","入門","使い方"]' '126'


  ########################################################################
  # tf
  copy_file 'article/library/tf' '[tensorflow__@__tutorilas]__@__1.__@__分類問題の初歩' '01' 'tutorials/01/' '["library","機械学習","tensorflow"]' '["tensorflow"]' '["tensorflow","tf","tutorial","チュートリアル"]' '1'
  copy_file 'article/library/tf' '[tensorflow__@__api]__@__1.__@__dataset' 'ds' 'api/dataset/' '["library","機械学習","tensorflow"]' '["tensorflow"]' '["tensorflow","tf","dataset"]' '10'


  ########################################################################
  # distribution functions
  copy_file 'ml/functions/dist' '確率分布一覧' 'dist' '' '["library","確率分布","機械学習"]' '["確率分布"]' '["確率分布","機械学習"]' '200'


  ########################################################################
  # special functions
  copy_file 'ml/functions/spec' '特殊関数' 'spec' '' '["library","特殊関数","機械学習"]' '["特殊関数"]' '["特殊関数","機械学習"]' '300'
  

  ########################################################################
  # NLP 100本ノック
  copy_file 'ml/nlp100/01' '[自然言語処理100本ノック]__@__第1章' '01' '' '["library","自然言語処理100本ノック","機械学習"]' '["自然言語処理100本ノック","第1章"]' '["自然言語処理100本ノック","機械学習","第1章"]' '401'
  copy_file 'ml/nlp100/02' '[自然言語処理100本ノック]__@__第2章' '02' '' '["library","自然言語処理100本ノック","機械学習"]' '["自然言語処理100本ノック","第2章"]' '["自然言語処理100本ノック","機械学習","第2章"]' '402'
  copy_file 'ml/nlp100/03' '[自然言語処理100本ノック]__@__第3章' '03' '' '["library","自然言語処理100本ノック","機械学習"]' '["自然言語処理100本ノック","第3章"]' '["自然言語処理100本ノック","機械学習","第3章"]' '403'
  copy_file 'ml/nlp100/04' '[自然言語処理100本ノック]__@__第4章' '04' '' '["library","自然言語処理100本ノック","機械学習"]' '["自然言語処理100本ノック","第4章"]' '["自然言語処理100本ノック","機械学習","第4章"]' '404'
  copy_file 'ml/nlp100/05' '[自然言語処理100本ノック]__@__第5章' '05' '' '["library","自然言語処理100本ノック","機械学習"]' '["自然言語処理100本ノック","第5章"]' '["自然言語処理100本ノック","機械学習","第5章"]' '405'
  copy_file 'ml/nlp100/06' '[自然言語処理100本ノック]__@__第6章' '06' '' '["library","自然言語処理100本ノック","機械学習"]' '["自然言語処理100本ノック","第6章"]' '["自然言語処理100本ノック","機械学習","第6章"]' '406'
  copy_file 'ml/nlp100/07' '[自然言語処理100本ノック]__@__第7章' '07' '' '["library","自然言語処理100本ノック","機械学習"]' '["自然言語処理100本ノック","第7章"]' '["自然言語処理100本ノック","機械学習","第7章"]' '407'
  copy_file 'ml/nlp100/08' '[自然言語処理100本ノック]__@__第8章' '08' '' '["library","自然言語処理100本ノック","機械学習"]' '["自然言語処理100本ノック","第8章"]' '["自然言語処理100本ノック","機械学習","第8章"]' '408'
  copy_file 'ml/nlp100/09' '[自然言語処理100本ノック]__@__第9章' '09' '' '["library","自然言語処理100本ノック","機械学習"]' '["自然言語処理100本ノック","第9章"]' '["自然言語処理100本ノック","機械学習","第9章"]' '409'
  copy_file 'ml/nlp100/10' '[自然言語処理100本ノック]__@__第10章' '10' '' '["library","自然言語処理100本ノック","機械学習"]' '["自然言語処理100本ノック","第10章"]' '["自然言語処理100本ノック","機械学習","第10章"]' '410'

  ########################################################################
  # NLP 100本ノック
  copy_file 'ml/data100/01' '[データ分析100本ノック]__@__第1章' '01' '' '["library","データ分析100本ノック","機械学習"]' '["データ分析100本ノック","第1章"]' '["データ分析100本ノック","機械学習","第1章"]' '501'
  copy_file 'ml/data100/02' '[データ分析100本ノック]__@__第2章' '02' '' '["library","データ分析100本ノック","機械学習"]' '["データ分析100本ノック","第2章"]' '["データ分析100本ノック","機械学習","第2章"]' '502'
  copy_file 'ml/data100/03' '[データ分析100本ノック]__@__第3章' '03' '' '["library","データ分析100本ノック","機械学習"]' '["データ分析100本ノック","第3章"]' '["データ分析100本ノック","機械学習","第3章"]' '503'
  copy_file 'ml/data100/04' '[データ分析100本ノック]__@__第4章' '04' '' '["library","データ分析100本ノック","機械学習"]' '["データ分析100本ノック","第4章"]' '["データ分析100本ノック","機械学習","第4章"]' '504'
  copy_file 'ml/data100/05' '[データ分析100本ノック]__@__第5章' '05' '' '["library","データ分析100本ノック","機械学習"]' '["データ分析100本ノック","第5章"]' '["データ分析100本ノック","機械学習","第5章"]' '505'
  copy_file 'ml/data100/06' '[データ分析100本ノック]__@__第6章' '06' '' '["library","データ分析100本ノック","機械学習"]' '["データ分析100本ノック","第6章"]' '["データ分析100本ノック","機械学習","第6章"]' '506'
  copy_file 'ml/data100/07' '[データ分析100本ノック]__@__第7章' '07' '' '["library","データ分析100本ノック","機械学習"]' '["データ分析100本ノック","第7章"]' '["データ分析100本ノック","機械学習","第7章"]' '507'
  copy_file 'ml/data100/08' '[データ分析100本ノック]__@__第8章' '08' '' '["library","データ分析100本ノック","機械学習"]' '["データ分析100本ノック","第8章"]' '["データ分析100本ノック","機械学習","第8章"]' '508'
  copy_file 'ml/data100/09' '[データ分析100本ノック]__@__第9章' '09' '' '["library","データ分析100本ノック","機械学習"]' '["データ分析100本ノック","第9章"]' '["データ分析100本ノック","機械学習","第9章"]' '509'
  copy_file 'ml/data100/10' '[データ分析100本ノック]__@__第10章' '10' '' '["library","データ分析100本ノック","機械学習"]' '["データ分析100本ノック","第10章"]' '["データ分析100本ノック","機械学習","第10章"]' '510'

  ########################################################################
  # bash
  copy_file 'article/library/bash' '[bash]__@__seq' 'seq' 'seq/' '["library","bash"]' '["bash","seq"]' '["bash","seq","入門","使い方"]' '6001'
  copy_file 'article/library/bash' '[bash]__@__expand' 'expand' 'expand/' '["library","bash"]' '["bash","expand"]' '["bash","expand","入門","使い方"]' '6002'
  copy_file 'article/library/bash' '[bash]__@__cut' 'cut' 'cut/' '["library","bash"]' '["bash","cut"]' '["bash","cut","入門","使い方"]' '6003'
  copy_file 'article/library/bash' '[bash]__@__echo' 'echo' 'echo/' '["library","bash"]' '["bash","echo"]' '["bash","echo","入門","使い方"]' '6004'
  copy_file 'article/library/bash' '[bash]__@__cat' 'cat' 'cat/' '["library","bash"]' '["bash","cat"]' '["bash","cat","入門","使い方"]' '6005'
  copy_file 'article/library/bash' '[bash]__@__unexpand' 'unexpand' 'unexpand/' '["library","bash"]' '["bash","unexpand"]' '["bash","unexpand","入門","使い方"]' '6006'
  copy_file 'article/library/bash' '[bash]__@__ls' 'ls' 'ls/' '["library","bash"]' '["bash","ls"]' '["bash","ls","入門","使い方"]' '6007'
  copy_file 'article/library/bash' '[bash]__@__tr' 'tr' 'tr/' '["library","bash"]' '["bash","tr"]' '["bash","tr","入門","使い方"]' '6008'


  copy_file 'article/library/bash' '[bash]__@__paste' '004' '004/' '["library","bash"]' '["bash","paste"]' '["bash","paste","入門","使い方"]' '6004'
  copy_file 'article/library/bash' '[bash]__@__split' '005' '005/' '["library","bash"]' '["bash","split"]' '["bash","split","入門","使い方"]' '6005'
  copy_file 'article/library/bash' '[bash]__@__cat' '006' '006/' '["library","bash"]' '["bash","cat"]' '["bash","cat","入門","使い方"]' '6006'
  copy_file 'article/library/bash' '[bash]__@__ls' '007' '007/' '["library","bash"]' '["bash","ls"]' '["bash","ls","入門","使い方"]' '6007'
  copy_file 'article/library/bash' '[bash]__@__sort' '008' '008/' '["library","bash"]' '["bash","sort"]' '["bash","sort","入門","使い方"]' '6008'
  copy_file 'article/library/bash' '[bash]__@__tar' '009' '009/' '["library","bash"]' '["bash","tar"]' '["bash","tar","入門","使い方"]' '6009'
  copy_file 'article/library/bash' '[bash]__@__sed' '010' '010/' '["library","bash"]' '["bash","sed"]' '["bash","sed","入門","使い方"]' '6010'
  copy_file 'article/library/bash' '[bash]__@__xarg' '011' '011/' '["library","bash"]' '["bash","xarg"]' '["bash","xarg","入門","使い方"]' '6011'
  copy_file 'article/library/bash' '[bash]__@__git' '012' '012/' '["library","bash"]' '["bash","git"]' '["bash","git","入門","使い方"]' '6012'
  copy_file 'article/library/bash' '[bash]__@__awk' '013' '013/' '["library","bash"]' '["bash","awk"]' '["bash","awk","入門","使い方"]' '6013'
  copy_file 'article/library/bash' '[bash]__@__wget' '014' '014/' '["library","bash"]' '["bash","wget"]' '["bash","wget","入門","使い方"]' '6014'

  copy_file 'article/library/bash' '[bash]__@__head' '015' '015/' '["library","bash"]' '["bash","head"]' '["bash","head","入門","使い方"]' '6015'
  copy_file 'article/library/bash' '[bash]__@__tail' '016' '016/' '["library","bash"]' '["bash","tail"]' '["bash","tail","入門","使い方"]' '6016'

  copy_file 'article/library/bash' '[bash]__@__uniq' '017' '017/' '["library","bash"]' '["bash","uniq"]' '["bash","uniq","入門","使い方"]' '6017'
  copy_file 'article/library/bash' '[bash]__@__sort' '018' '018/' '["library","bash"]' '["bash","sort"]' '["bash","sort","入門","使い方"]' '6018'


  ########################################################################
  # python
  copy_file 'article/library/python' '[python]__@__snipet' '001' '001/' '["library","python"]' '["python"]' '["python","入門","使い方"]' '5001'
  copy_file 'article/library/python' '[python]__@__オブジェクトの使用メモリの確認' '002' '002/' '["library","python"]' '["python","メモリ"]' '["python","入門","使い方"]' '5002'
  copy_file 'article/library/python' '[python]__@__snipet' '003' '003/' '["library","python"]' '["python","メモリ"]' '["python","入門","使い方"]' '5003'
  copy_file 'article/library/python' '[python]__@__re__@__正規表現' '004' '004/' '["library","python"]' '["python","正規表現"]' '["python","入門","使い方"]' '5004'
  copy_file 'article/library/python' '[python]__@__snipet' '005' '005/' '["library","python"]' '["python","メモリ"]' '["python","入門","使い方"]' '5005'
  copy_file 'article/library/python' '[python]__@__snipet' '006' '006/' '["library","python"]' '["python","メモリ"]' '["python","入門","使い方"]' '5006'
  copy_file 'article/library/python' '[python]__@__snipet' '007' '007/' '["library","python"]' '["python","メモリ"]' '["python","入門","使い方"]' '5007'
  copy_file 'article/library/python' '[python]__@__snipet' '008' '008/' '["library","python"]' '["python","メモリ"]' '["python","入門","使い方"]' '5008'
  copy_file 'article/library/python' '[python]__@__snipet' '009' '009/' '["library","python"]' '["python","メモリ"]' '["python","入門","使い方"]' '5009'
  copy_file 'article/library/python' '[python]__@__snipet' '010' '010/' '["library","python"]' '["python","メモリ"]' '["python","入門","使い方"]' '5010'


}

main

