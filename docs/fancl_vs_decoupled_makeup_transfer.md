# FANCLのバーチャルメイクと arXiv:2509.02445v2 の比較

## 目的

この文書は、FANCL「パーソナルカラー×顔タイプ診断」に組み込まれているバーチャルメイクと、論文 **Towards High-Fidelity, Identity-Preserving Real-Time Makeup Transfer: Decoupling Style Generation**（arXiv:2509.02445v2）を、`ikiikimake` の研究・実装方針を考えるために比較したメモである。

両者はどちらも「自分の顔でメイク後の見え方を確認する」体験を提供できるが、そもそもの目的と技術構成がかなり異なる。

- FANCL: **似合うメイクの提案 + 商品試着**を主目的とする商用サービス
- arXiv:2509.02445v2: **実在する参照メイクを、本人の顔情報から分離した透明レイヤーとして抽出し、別の顔へ高忠実度に転写する**ことを主目的とする研究

したがって、単純に「AR vs 生成AI」と整理するより、**推薦、メイク表現、レンダリング、分析可能性**を分けて考える方がよい。

---

## 1. FANCL側

### ユーザー体験

FANCLの「パーソナルカラー×顔タイプ診断」では、設問への回答からパーソナルカラーと顔タイプを組み合わせたタイプを提示し、そのタイプに応じたメイク方法・カラー・商品を提案している。

公式ページでは、

- パーソナルカラー: 4タイプ
- 顔タイプ: 4タイプ
- 組み合わせ: 16パターン
- おすすめメイク
- アイシャドウ、アイブロウ、マスカラ、アイライナー、チーク、ハイライト、ルージュ等のバーチャル試着

が確認できる。

つまり大まかには、

```text
設問回答
  ↓
パーソナルカラー × 顔タイプ
  ↓
16タイプのいずれか
  ↓
タイプごとのおすすめメイク・商品
  ↓
ARバーチャルメイクで試着
```

という構成である。

### AR部分

FANCLのバーチャルメイクは **Perfect Corp.（パーフェクト社）** が提供している。Perfectの導入発表では、FANCLオンラインにメイクAR機能を提供し、診断タイプに合わせたメイクを疑似体験できるとしている。

Perfectの一般的なARバーチャルメイク製品について公開されている情報では、

- AgileFace® フェイス認識AI
- ディープラーニング
- リアルタイム顔追跡
- ARレンダリング
- 実商品の色、濃度、テクスチャ再現
- リップ、アイメイク、チーク、ハイライト、ファンデーション等
- マット、サテン、シアー、グロス、シマー等の質感表現

に対応するとされている。

ただし、**FANCL導入環境の内部ネットワーク構造、ランドマーク形式、メイクマスク表現、alpha表現、シェーダー、3D顔モデルの詳細は公開情報からは分からない**。

したがって、FANCLについて確実に言えるのは、

1. 診断・推薦ロジックがある
2. PerfectのARで推薦メイクを自分の顔に表示できる
3. 見た目の自然さを重視した商用品質の試着である

という範囲である。

---

## 2. arXiv:2509.02445v2 側

### 基本思想

論文はメイク転写を、

1. **transparent makeup mask extraction**
2. **graphics-based mask rendering**

の2段階に分離する。

顔全体を別の画像として生成するのではなく、参照メイクから **RGBAの透明メイクレイヤー** を取り出し、そのレイヤーをターゲット顔へ変形・合成する。

```text
参照メイク画像
  ↓
eye / lip / cheek の透明RGBAマスク抽出
  ↓
抽出したメイクレイヤーを保持
  ↓
ターゲット顔のlandmark / parsing
  ↓
warp
  ↓
alpha blend / graphics rendering
```

一度メイクマスクを抽出した後は、動画の各フレームで同じマスクを使い回せるため、リアルタイム性と時間的一貫性を狙っている。

### 学習データ

真の「素顔 + メイクRGBA」のペアを大量に取得することは難しいため、論文はpseudo ground truthを使う。

#### Graphics-rendered pseudo ground truth

メイクの

- shape
- color
- transparency
- finish

を制御したグラフィック素材を生成し、自然顔へTPS warp + alpha blendする。

これにより、

- メイク後画像
- 正解RGBAメイクマスク

のペアを作る。

#### 実メイク画像からのeye pseudo label

実画像の多様なアイメイクを取り込むため、

1. canonical faceへwarp
2. LAB色空間でk-means
3. 頻度上位クラスタからskin tone推定
4. skin toneとのcosine similarityからalpha推定

を行う。

論文式:

```text
Alpha(i,j) = max(0, 1 - CosSim(pixel_LAB(i,j), skinTone_LAB))
```

設定は `k=6`, `s=2`。

これは物理的な真のalphaを直接観測しているのではなく、実メイク画像から学習用のpseudo labelを得るための近似である。

### モデル

eye / lip / cheekごとに別のモデルを持つ。

- U-Netベースgenerator
- regionごとのdiscriminator
- lipにはcolor regressorを追加
- 出力はRGBAマスク

主な損失:

- alpha-weighted reconstruction loss
- alpha loss
- adversarial loss
- lip color loss

### 推論

参照画像からRGBAマスクを抽出した後、ターゲット顔ではlandmarkとface parsingを使ってメイクレイヤーをwarpし、顔領域へ適用する。

つまり、推論時の「メイクを載せる」部分はかなりgraphics/AR的であり、毎回顔全体を生成する方式ではない。

---

## 3. 比較表

| 観点 | FANCL + Perfect AR | arXiv:2509.02445v2 |
|---|---|---|
| 主目的 | 似合う商品・メイクの提案と試着 | 参照メイクの高忠実度な転写 |
| 推薦 | あり。診断タイプからおすすめを提示 | なし。どのメイクが似合うかは扱わない |
| 入力メイク | 商品/ルックとして事前登録されたメイク | 実在する参照メイク画像 |
| メイク表現 | 内部表現は非公開 | eye/lip/cheekのRGBA透明マスク |
| 顔認識 | PerfectのAI/AR技術 | lightweight face tracking + landmarks + parsing |
| 描画 | 商用ARレンダリング | warp + graphics-based mask rendering |
| リアルタイム | 商用品質のリアルタイム試着 | mask抽出後はリアルタイム描画を目的 |
| 色・質感 | 商品色、濃度、各種textureを再現可能と公表 | 参照メイクの色・形・透明度・細部を抽出 |
| 実在メイクをそのまま転写 | 公開情報からは確認できない | 中心的な目的 |
| 新しいメイクスタイル追加 | SKU/ルック登録が基本と考えられるが詳細非公開 | 参照画像を与えて抽出可能 |
| メイク特徴量の取得 | 自前実装なら可能。ただしPerfect内部値へのアクセス可否は非公開 | RGBAを直接保持するため取りやすい |
| 部位別介入 | AR側では商品・部位別に可能 | eye/lip/cheekを独立して扱う |
| 分析・研究用途 | 商品推薦には向くが内部実装はブラックボックス | レイヤーが明示的なので分析しやすい |
| 本人性の保持 | ARなので元顔を保持しやすい | identityとmakeupを分離すること自体が研究目的 |
| シニア特化 | FANCLの公開ページはシニア専用ではない | 論文自体はシニア専用ではない |
| 実装の再現可能性 | Perfect部分はプロプライエタリ | 論文記述から独立実装可能 |
| 品質の現状 | 実運用されており非常に自然 | 論文では高品質だが、自前再現には十分なデータと学習が必要 |

---

## 4. 「ARの方が上」なのか

一概には言えない。

### 見た目・プロダクト体験

現時点の商用品質ではFANCL/Perfectの方が強い可能性が高い。

理由は、

- 長期間作り込まれた顔追跡
- AR描画
- 商品ごとの色調整
- texture表現
- リアルタイム最適化
- 多数デバイスへの対応

がすでに製品化されているためである。

`ikiikimake` が独自に同じ描画品質まで到達するには、かなりの実装・調整が必要になる。

### 研究・分析

一方、arXiv方式は **「メイクそのものをRGBAレイヤーとして取り出す」** ため研究には非常に扱いやすい。

例えば抽出RGBAから、

- hue / saturation / Lab
- mean alpha
- alpha distribution
- mask area
- faceに対するarea ratio
- centroid
-左右差
- eye makeupの外側への広がり
- cheek位置
- lip境界
- texture / spatial frequency

などを定量化できる。

ARでも自前でメイクパラメータを保持していれば同様の特徴量は出せるため、**「ARだから分析できない」という違いではない**。

本質的な差は、

> メイク表現・レンダリングの内部値を自分たちが直接取得・変更できるか

である。

PerfectをブラックボックスSDKとして使う場合、その内部値をどこまで取得できるかはSDK仕様次第になる。

---

## 5. ikiikimakeで考えられる構成

FANCLと論文は競合する方式ではなく、組み合わせることもできる。

### 研究側

```text
日本の実在シニアメイク画像
  ↓
arXiv方式でRGBA抽出
  ↓
eye / lip / cheek の特徴量化
  ↓
複数の顔へ同一メイクを適用
  ↓
人間による印象評価
  ↓
顔特徴 × メイク特徴 → 印象変化を分析
```

### アプリ側

```text
ユーザー顔
  ↓
顔特徴を取得
  ↓
過去の分析データからメイク候補を推薦
  ↓
高品質なAR/graphics rendererで試着
```

この場合、

- **FANCLから学ぶもの**: ユーザー体験、推薦 → 試着の導線、AR品質
- **論文から使うもの**: 実在メイクの抽出、再利用可能なRGBA表現、分析可能性

と役割を分けられる。

---

## 6. ikiikimakeで特に研究価値がありそうな部分

FANCLは「診断タイプに合わせておすすめメイクを提示する」構成だが、`ikiikimake` では必ずしも最初からタイプを決める必要はない。

例えば同一人物に複数メイクを載せ、

```text
original
original + makeup A
original + makeup B
original + makeup C
```

を作り、人間評価で

- 健康的
- 自然
- 華やか
- やさしい
- 若々しい
- 好印象

などの変化量を取得できる。

その後、

```text
顔特徴 + メイク特徴
        ↓
印象変化
```

の関係を解析する。

これは、

```text
あなたは○○タイプ
  ↓
このメイクがおすすめ
```

という固定カテゴリ型とは異なり、

```text
この顔に、この具体的なメイク介入をした場合
どの印象がどの程度変わるか
```

を検証する方向である。

特にシニア女性に対象を絞れば、

- 加齢した肌のtexture
- wrinkles
- uneven skin tone
- lip colorの低下
- cheek color
- eye contrast

などとメイクの相互作用を分析する余地がある。

---

## 7. 現時点の整理

`ikiikimake` にとっては、どちらか一方を選ぶ必要はない。

**FANCL型のUX + 論文型のメイク表現・分析基盤** が最も自然な組み合わせである。

```text
                 ┌─ 実在メイク収集
                 │
                 ↓
        RGBA makeup extraction
                 │
                 ├─→ 特徴量・研究データ
                 │
                 ↓
顔分析 → 推薦モデル → メイク候補
                 │
                 ↓
          AR / graphics rendering
                 │
                 ↓
            ユーザー試着
```

短期的には、商用ARと同じ自然さをゼロから再現することよりも、

1. 実在シニアメイクをRGBAとして安定して抽出する
2. メイク特徴量を定義する
3. 同一顔に複数メイクを適用できる実験環境を作る
4. 印象評価データを集める
5. どの顔特徴 × メイク特徴が印象変化と関係するか調べる

ことの方が、`ikiikimake` 独自の研究価値につながる。

---

## 公開情報から分からないこと

比較にあたって、以下は推測で埋めない。

### FANCL / Perfect

- FANCL環境で使用している具体的なニューラルネットワーク
- landmark点数・3D mesh仕様
- makeup内部表現がRGBAかどうか
- alphaの持ち方
- renderer / shaderの詳細
- SDKから各メイク特徴量を外部取得できるか
- FANCLの16タイプ推薦ルールの詳細な重みや判定式

### 論文

- 実メイクのalphaは真の物理計測ではなくpseudo label
- 2D keypoint / parsing依存のため、極端なposeやself-occlusionには限界がある
- natural / non-opaque makeupを前提とし、完全に不透明な特殊メイク等は対象外になりうる

---

## 参考

- FANCL パーソナルカラー×顔タイプ診断  
  https://www.fancl.co.jp/beauty/coloranalysis/index.html
- Perfect Corp. FANCL導入発表  
  https://www.perfectcorp.com/ja/business/news/Fancl-makeup
- Perfect Corp. ARバーチャルメイク  
  https://www.perfectcorp.com/ja/business/products/virtual-makeup
- Chau, Yu, Jiang, *Towards High-Fidelity, Identity-Preserving Real-Time Makeup Transfer: Decoupling Style Generation*  
  https://arxiv.org/html/2509.02445v2
