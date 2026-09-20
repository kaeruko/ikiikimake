# メイク前後の自動ROI生成

顔ランドマークを基準に、**画面左頬・画面右頬・額**のpolygonを自動生成します。初回は確認画像の保存までで止まり、人間が確認したIDを指定したときだけ、保存済みマスクを使ってLab解析します。メイクの点数は出しません。

この作業環境では `.venv` と顔・手のモデルをセットアップ済みです。

## 使う

リポジトリ直下で、PowerShellから実行します。出力先には新しいフォルダ名を指定してください。既存の結果には上書きしません。

```powershell
.\.venv\Scripts\python.exe .\analysis\run_cheek01.py `
  --before .\data\cheek01\before.png `
  --after .\data\cheek01\after.png `
  --output .\outputs\cheek01_auto_new
```

`review.html` または `review.png` を開きます。**画面左頬＝緑、画面右頬＝青、額＝黄**です。左右は本人の左右ではなく、画像上の左右です。

- 目・鼻・口・髪・字幕・道具を含まず、肌の上に乗っているか。
- 手だけでなく、その影・反射・押圧による変形も避けられているか。
- 前後で、対応する頬・額の位置になっているか。
- 顔向きや表情の違いが、比較したい変化より大きくないか。

問題なければ、画面に表示された確認IDで次を実行します。この操作が目視確認済みの指定になります。

```powershell
.\.venv\Scripts\python.exe .\analysis\run_cheek01.py `
  --output .\outputs\cheek01_auto_new `
  --approve-review 表示された確認ID
```

承認時に再検出はしません。画像・マスク・座標JSON・確認画像が生成後に変わっていれば停止します。失敗した結果は承認できません。画像を選び直し、別の出力先で生成してください。

画像1枚だけのROI生成もできます。

```powershell
.\.venv\Scripts\python.exe .\analysis\extract_face_rois.py `
  --image .\data\cheek01\before.png `
  --output .\outputs\before_roi_new
```

## 出力

```text
outputs/cheek01_auto_new/
  review.png                 前後の確認画像
  review.html                確認項目・品質情報
  review.json                状態、確認ID、入力と成果物のハッシュ
  before/                    after/ も同じ構造
    roi_overlay.png
    roi_masks.npz            left_cheek / right_cheek / forehead の H×W, uint8, 0/1
    roi_points.json          polygon座標、ランドマーク、モデル情報、判定理由
    left_cheek_mask.png
    right_cheek_mask.png
    forehead_mask.png
  lab/                       確認IDで承認した場合にのみ作成
    lab_stats.csv
    lab_deltas.csv
    analysis_summary.json
    roi_samples.png
    *_lab_hist.png
```

失敗時は診断用画像・JSONと理由を残し、不合格の画像には解析用NPZを出しません。前の画像で失敗した場合は、次の画像に進みません。

## 処理と停止条件

MediaPipe Face Landmarkerで基準点を取り、その固定インデックスから頬・額のpolygonを作り、中心へ12%縮めます。顔の向きはモデルの変換行列から推定します。Hand Landmarkerで検出した手の点を囲む領域に余白を加え、ROIとの重なりを検査します。画像の色やテクスチャを生成・補完する処理はしません。

現在の閾値は初期の実験用設定です。検証済みの採点基準や検出精度を表すものではありません。

| 停止条件 | 初期値 |
|---|---|
| 顔の数 | 0人または複数人（最大2人を検出） |
| 顔検出／顔存在の内部しきい値 | 各0.6 |
| 顔の横向き yaw／上下向き pitch／傾き roll | 絶対値25°／25°／20°を超える |
| 顔幅 | 120px未満 |
| ROI | 画面外、重複、100画素未満 |
| 検出した手の推定領域との重なり | ROIの2%を超える |
| 飽和・黒つぶれの画素 | ROIの10%を超える（8-bitのいずれかの成分が255、または全成分が2以下） |
| 前後の姿勢差 | yaw／pitch 10°、roll 12°を超える |
| 額のL*中央値の前後差 | 絶対値12を超える |

**額のL*差は露出差そのものではありません。** 塗布、影、顔向き、照明などが混ざるため、大きく変わった組を保留にする参考チェックです。露出やホワイトバランスの同一性を保証する仕組みではありません。

ランドマークごとの信頼度はこのAPIから得られないため、JSONでは `landmark_confidence: null` とします。内部しきい値と、実際に得られた信頼度を混同しません。

**手の未検出は「手なし」の保証ではありません。** 部分的な手、握った道具、髪、字幕、影などは見逃せます。今回の `data/cheek01/before.png` でも見えている指は未検出でした。全自動の安全判定には使わず、確認画像を見てください。

## Lab解析

元の `analyze_cheek_lab.py` と同じ8-bit OpenCV Lab換算と、a*の集計指標・閾値を使います。polygon外の画素を統計に含めません。

- 左右の頬と額について、前後の統計を出します。
- 頬の `after − before` と、参考値として `頬の変化 − 額の変化` を分けて出します。
- 額を差し引いた値には `unvalidated_forehead_control` と記録します。額にも塗布したケースなどでは対照として成立しないためです。
- 位置合わせをしていない画素同士の差分ヒートマップは作りません。比較するのは領域の分布と集計値です。

元の手動スクリプトと `outputs/cheek01` はそのまま残しています。新しい自動経路では `selectROI` は呼びません。

## 別環境でのセットアップ

Windows / Python 3.12.10で確認しています。別の環境ではパッケージやモデルの対応を確認してください。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe .\analysis\setup_roi_models.py
```

公式のバージョン1のモデルをダウンロードしてSHA-256を照合します。モデルは `models/` に保存され、Git管理対象外です。モデルをそろえた後の画像解析はローカルで動き、画像を外部へ送りません。

参照：[Face Landmarker公式資料](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python)、[Hand Landmarker公式資料](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python)。

## 検証

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

マスク外画素の影響、元のLab統計との一致、画面外や重複、手との重なり、姿勢、確認前の解析禁止、確認後の画像変更検出などを検査します。実画像でROIの位置を目視確認し、手で頬を覆ったフレームで停止することも確認しました。
