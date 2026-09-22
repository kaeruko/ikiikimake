# 透明メイクマスク抽出・転写の実装

[Towards High-Fidelity, Identity-Preserving Real-Time Makeup Transfer: Decoupling Style Generation (arXiv:2509.02445v2)](https://arxiv.org/html/2509.02445v2) の公開記述を基にした独立実装です。目・唇・頬のRGBAマスクを別々のU-Netで推定し、そのマスクを顔ランドマークに合わせて変形・合成します。

**コードの実装と動作確認用の短い学習を提供します。論文の精度を再現した学習済みモデルではありません。** 合成メイク素材、トラッカー、顔領域マスクは下記の代替実装を使います。既存のROI・Lab解析とは別の `makeup_transfer` パッケージです。

## データ

既定の入力はこのリポジトリの `datasets/` です。Windowsのjunctionもたどり、参照元画像は変更しません。

| 入力 | 確認した画像数 | 元画像サイズ | 用途 |
|---|---:|---|---|
| `datasets/ffhq-dataset/senior_female_50plus` | 3,471 | 128×128 | 合成ペアの元画像 |
| `datasets/fairface_asian_female_50plus` | 755 | 224×224 | 合成ペアの元画像 |

全4,226枚の画像ヘッダで上記サイズを確認しました。部位を256×256に拡大しても元の細部は増えません。論文のfull-face約1024相当の高精細転写を検証するには、高解像度の顔画像を用意する必要があります。

`docs/`、`metadata/`、FFHQの紹介画像などは除外します。元画像のSHA-256で重複を除き、約80%/10%/10%をtrain/val/testに分けた後、1画像あたり既定3種類のメイクを生成します。同じ画像のメイク違いは異なるsplitに入りません。別写真に写る同一人物の照合は行わないため、人物単位での分離ではありません。正規化顔形状と平均alphaはtrainだけで作成します。

生成コード・顔モデル・各学習サンプル・平均alpha・正規化顔形状のハッシュと、NumPy/OpenCVのバージョンをmanifestに記録します。読込時に記録とデータを照合するため、同じ画像・乱数seedでも合成処理や保存内容が変われば別の学習データとして識別します。

手元のデータは年齢・性別などで選別された部分集合です。元画像のすっぴん・遮蔽物の有無は確認されておらず、既存メイクが重なる可能性があります。論文の20,000枚のFFHQ選別画像、MTのメイク参照画像、評価用のMakeup-Wild/LADNとは異なります。

## セットアップ

リポジトリ直下のPowerShellで実行します。既存の `.venv` にCUDA版PyTorchを追加済みで、RTX 5070 Tiで動作確認しています。別環境では次の手順を使います。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe .\analysis\setup_roi_models.py
```

CPU環境ではPyTorch公式のCPU版を使い、各コマンドに `--device cpu` を指定できます。学習はGPUを推奨します。`inventory` と `prepare` はPyTorchなしでも実行できます。

## 小規模に一通り動かす

出力先は新しいフォルダにしてください。以下は動作確認用で、十分な学習を行う設定ではありません。

```powershell
.\.venv\Scripts\python.exe -m makeup_transfer inventory --datasets datasets

.\.venv\Scripts\python.exe -m makeup_transfer prepare `
  --datasets datasets --output outputs\makeup_transfer_trial\data `
  --max-images 32 --variants 3

.\.venv\Scripts\python.exe -m makeup_transfer train `
  --data outputs\makeup_transfer_trial\data `
  --output outputs\makeup_transfer_trial\checkpoints `
  --base-channels 8 --architecture strided --batch-size 2 `
  --epochs 1 --max-steps 3 --color-epochs 1 --color-batch-size 2 --color-max-steps 3

.\.venv\Scripts\python.exe -m makeup_transfer evaluate `
  --data outputs\makeup_transfer_trial\data `
  --checkpoints outputs\makeup_transfer_trial\checkpoints `
  --output outputs\makeup_transfer_trial\evaluation

.\.venv\Scripts\python.exe -m makeup_transfer benchmark `
  --data outputs\makeup_transfer_trial\data `
  --checkpoints outputs\makeup_transfer_trial\checkpoints `
  --output outputs\makeup_transfer_trial\benchmark --max-pairs 2
```

`data/previews/` でメイク合成の位置を確認できます。`evaluation/` は入力・正解RGBA・予測RGBAの比較、`benchmark/` は別の顔への転写比較です。短い学習ではマスクに不自然な色や透明度が残ります。

## 手元の全データを使って学習する

```powershell
.\.venv\Scripts\python.exe -m makeup_transfer prepare `
  --datasets datasets --output outputs\makeup_transfer_full\data --variants 3

.\.venv\Scripts\python.exe -m makeup_transfer train `
  --data outputs\makeup_transfer_full\data `
  --output outputs\makeup_transfer_full\checkpoints `
  --epochs 55 --color-epochs 10 --base-channels 64 `
  --architecture paper --batch-size 8 --color-batch-size 32
```

最後の設定は論文の基本値に対応しますが、付録のstride=1を使う `paper` の識別器・色回帰器は大量のGPUメモリを使います。16GB環境でこのバッチサイズが収まることは保証していません。ローカル向けには `--architecture strided --batch-size 2 --color-batch-size 4 --amp`、必要に応じて `--base-channels 32` を指定します。この変更は論文と異なる構成として保存されます。

`--sources ffhq` / `--sources fairface` で片方だけを選べます。実際のメイク参照画像がある場合は、prepareに `--real-makeup パス` を追加すると目のk-means擬似ラベルも生成・混合します。入力を本当のメイク画像として扱う指定なので、すっぴんデータをこの引数に指定しないでください。

中断後は同じtrainコマンドに `--resume` を追加します。重み、最適化器、AMP状態、処理ステップを復元します。保存は各epoch終了時または `--max-steps` 到達時です。epoch途中のcheckpointを再開する場合、そのepochを最初から繰り返します。乱数状態まで一致する厳密な再開ではありません。`--epochs` と `--max-steps` は再開後の追加数ではなく全体の上限です。再開時に指定した学習率が最適化器にも適用されます。

## 画像・動画への転写

メイク済みの参照画像と転写先を指定します。学習済み重みがない場合はエラーになります。

```powershell
.\.venv\Scripts\python.exe -m makeup_transfer transfer `
  --reference path\to\makeup_reference.jpg --target path\to\target.jpg `
  --checkpoints outputs\makeup_transfer_full\checkpoints `
  --output outputs\makeup_transfer_full\transferred.png

.\.venv\Scripts\python.exe -m makeup_transfer video `
  --reference path\to\makeup_reference.jpg --target path\to\video.mp4 `
  --checkpoints outputs\makeup_transfer_full\checkpoints `
  --output outputs\makeup_transfer_full\transferred.mp4
```

`--regions lip` などで部位を選択でき、`--strength 0.5` で合成強度を下げられます。画像では `--semantic-mask mask.png` で転写先と同じ大きさの可視領域マスクを追加できます。白/1が塗布可能、黒/0が除外です。

動画は参照マスクを最初に1回だけ抽出し、各フレームではランドマーク検出・平滑化・TPS合成を行います。顔が見つからない、または複数の顔が見つかったフレームはそのまま出力します。`--max-frames 30` で短く検証でき、`--smoothing 0` で平滑化を無効化できます。OpenCVによる動画出力には音声が含まれません。実測処理速度をJSONに保存し、リアルタイム性能は保証しません。

画像・動画には処理設定を記録したJSONと、抽出したRGBAパッチのNPZが付きます。画像は合成alpha画像も保存します。

## 論文との対応と補完点

| 論文の要素 | 実装 |
|---|---|
| 目・唇・頬を分離 | 各部位に独立した生成器・識別器 |
| U-Net-256、入力/出力4ch | RGBと平均alphaを入力、RGBAを出力。内部でpix2pixの−1〜1に変換 |
| 条件付き識別器 | 正規化顔へTPS変形した参照RGBとRGBAの7ch |
| graphics-based擬似正解 | 色・形状・透明度・光沢を変える手続き的素材とTPS合成 |
| k-means擬似ラベル | LAB色、k=6、上位s=2クラスタの肌色、式(1)のcos類似度からalpha |
| 式(3)の再構成損失 | 正解alphaを重みにしたRGBのL1。既定では画素・チャネル・batchの平均 |
| alpha損失 | 合成ペアにだけL1を適用。k-meansのalphaはRGB重みのみに利用 |
| 唇の色補助モデル | 色回帰器をSGDで事前学習し、生成器学習中はBNを含め固定。入力への勾配は維持 |
| 損失重み | 再構成100、alpha100、GAN10、唇の色50 |
| 学習 | Adam (0.5, 0.999)、G/D各2e-4、55epoch、batch8。色はSGD 5e-5、momentum0.9、10epoch、batch32 |
| 拡張 | 学習入力に平行移動・拡大縮小・ぼかし |
| 動画推論 | 参照の抽出1回、フレームごとの同じスタイルの再利用 |

以下は論文から一意に復元できない、または手元の資源に合わせて変更した箇所です。

- 専用素材ライブラリを手続き的テンプレートに置き換えています。色・形状の多様さや質感は同一ではありません。
- トラッカーはMediaPipeです。顔パーサの代わりにランドマーク由来の幾何マスクを使い、目の内部・眉・口の内部を除外します。髪・手・眼鏡などの遮蔽を意味的に判定するモデルは含みません。
- 標準の顔キャンバスは512、部位入力は256です。論文のfull-face約1024との違いがあります。`prepare --canvas-size 1024` で変更できます。TPSは128×128格子で計算した座標写像を補間する既定実装です。
- 付録の識別器はstride=1の畳み込みの後に36入力の全結合層があり、256入力からの空間サイズが整合しません。全結合層の前に6×6のadaptive average poolingを追加しています。
- 省メモリ版 `strided` は識別器・色回帰器のstrideを2に変えます。U-Netは同じ構造ですが、`base_channels` を変えた場合は幅も変わります。
- 式(3)の総和は既定で平均に正規化しています。`losses.reconstruction_loss(..., reduction='sum')` には式どおりの総和もあります。
- 式(4)/(5)はL2ノルム、本文は二乗平均と読めるため、既定MSEと `--color-loss-type l2` を選べます。色回帰の教師はRGBの3成分です。
- G/Dはミニバッチごとに交互更新します。本文のepoch単位更新の記載やTTURの別々の学習率は詳細不明のため、両方2e-4を既定とし、識別器は `--discriminator-lr` で変更できます。

## 評価とテスト

`evaluate` は未使用splitの部位マスクについて、alpha加重RGB誤差・alpha誤差・黒背景に合成したマスクのPSNRを出します。これは論文表のフル顔転写PSNRとは異なります。

`benchmark` は異なる未使用顔A/Bに同じ合成スタイルを適用し、Aから抽出したメイクをBに移した結果と、Bの合成正解を比較します。入力データと合成器が異なるため、論文表の数値と直接比較できません。少数ペアでの評価は動作確認です。

任意の知覚評価は次で追加できます。初回は各評価器の学習済み重みを取得する場合があります。

```powershell
.\.venv\Scripts\python.exe -m pip install torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install lpips==0.1.4 pytorch-fid==0.3.0
# benchmarkコマンドに --perceptual を追加
```

実メイク参照を使うFID(I)の論文実験は実行していません。転写元の人物らしさが混入しないこと、様々な肌色での性能、時間的一貫性についても、本実装を動かしただけで論文と同じ性能が得られるとは判断できません。

任意評価の `fid_identity_synthetic` は合成メイク参照による転写と元画像の分布差で、実メイク参照による論文のFID(I)とは区別しています。

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_makeup_*.py" -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

重みは `eye.pt` / `lip.pt` / `cheek.pt` と、事前学習した `color_regressor.pt` に保存します。`geometry.npz` をcheckpointと一緒に持ち運べます。学習条件は `training_config.json`、各epochの損失は `history.jsonl` に保存します。

## この環境での確認結果（2026-09-23）

- Python 3.12.10、PyTorch 2.11.0+cu128、RTX 5070 Tiで確認。
- 全体から32枚（FFHQ28/FairFace4）を固定seedで抽出。元画像25/3/4枚をtrain/val/testに分け、合計288件の部位ペアを作成。
- 最終確認モデルは幅16、`strided`、batch2で各部位10epoch・380step。色回帰器はbatch4、10epoch・190step。**小規模の動作確認モデルで、論文設定の本学習ではありません。**
- `paper`構成も幅8・batch1で各部位の学習が動くことを別途確認。
- 2組の未使用顔で転写、`cheek01.mp4` の12フレームで参照抽出1回・動画出力を確認。640×360で約11.1fps（短い実行、画像転写処理との同時実行時の実測）。30fps動作を達成した結果ではありません。
- 全145テスト成功。生成データ292ファイルのハッシュ照合と依存関係の整合性確認も成功。
- LPIPS/FIDの任意評価、全4,226枚による本学習、論文スコアの再現は未実施。

最終成果物は `outputs/makeup_transfer_final/` です。

- `checkpoints/`: 各部位・色回帰器の重み、学習設定・ログ
- `benchmark/reviews/pair_0000.png`: 参照、元画像、合成正解、学習モデルの転写を並べた画像
- `benchmark/benchmark.json`: 別の顔への転写評価
- `evaluation/metrics.json`: 部位ごとの検証指標
- `transferred.png` / `video_check.mp4`: CLIによる画像・動画出力

比較画像では目の色などに正解との差が残っています。小規模モデルの転写品質を示す確認資料として扱ってください。

## ファイル構成

```text
makeup_transfer/
  geometry.py    顔検出・位置合わせ・TPS・部位マスク
  synthesis.py   手続き的メイク素材・k-means擬似ラベル
  prepare.py     データ検索・分割・ペア生成
  data.py        学習データ読込・拡張
  models.py      U-Net・識別器・色回帰器
  losses.py      alpha・色・GAN損失
  train.py       事前学習・GAN学習・保存・再開
  inference.py   画像/動画転写
  evaluate.py    部位マスク診断
  benchmark.py   別の顔への合成転写評価
  __main__.py    コマンド入口
```

参照: [論文本文・付録](https://arxiv.org/html/2509.02445v2)、[pix2pix公式実装](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)、[PyTorch公式セットアップ](https://pytorch.org/get-started/locally/)。
