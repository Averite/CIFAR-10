# CIFAR-10分類プログラム README
CIFAR-10は初心者向けの定番プロジェクトですが、私は単なる模倣に留まらず、自分自身の考えや理解をもとにモデルを改良し、精度を向上させることができました。
このプロジェクトは、CIFAR-10画像を分類するためのディープラーニングパイプラインを示しています。独自の畳み込みニューラルネットワーク（CNN）モデルを使用し、データ準備、モデル学習、テスト、および個別の画像に対する推論を含みます。PyTorchとTensorBoardを利用して学習とログ記録を行います。

---

## 📄 **概要**

### **モデル**
`Tudui` はカスタムCNNモデルで、以下の特徴を持っています：
- **畳み込み層**: 特徴抽出のための複数の畳み込み層を搭載。
- **注意機構**: SEBlockを使用して重要な特徴マップを強調。
- **Dropout**: 学習時にニューロンをランダムに無効化して過学習を防止。
- **出力層**: 特徴表現をCIFAR-10の10クラスにマッピング。

---

### **学習**
- **データセット**: CIFAR-10データセット（PyTorchを通じて自動ダウンロード）。
- **損失関数**: 交差エントロピー損失（Cross-Entropy Loss） を使用して分類タスクを最適化。
- **最適化アルゴリズム**: 学習率0.01の確率的勾配降下法（SGD）。
- **バッチサイズ**: 訓練およびテスト時に64。
- **エポック数**: 訓練は10エポックを実行。

---

### **テスト**
- テストデータセット全体の損失と正解率を計算。
- TensorBoardを使用して学習過程を視覚化。

---

### **推論**
- 個別の画像（例: `dog.png`）に対する分類結果を確認可能。
- 学習済みモデル（例: `tudui_9.path`）をロードして推論を実行。

---

## 🚀 **使い方**

### 1. **必要なライブラリをインストール**
以下のコマンドで必要なパッケージをインストールします：
```bash
pip install torch torchvision tensorboard
```

### 2. **データセットの準備**
`./dataset` ディレクトリにCIFAR-10データセットをダウンロードして保存します。

### 3. **モデルの学習**
以下のコマンドを実行してモデルを学習します：
```bash
python train.py
```
- モデルの学習進捗は `./logs_train` にTensorBoardログとして保存されます。

### 4. **モデルのテスト**
テストデータセットに対してモデルを評価し、以下のようにテスト結果を表示します：
- 全体的な損失
- 正解率

### 5. **推論**
任意の画像（例: `dog.png`）に対して推論を実行します
結果として画像の分類結果（クラスIDおよび確率）が出力されます。

---

## 🔧 **プログラムの構造**
### **主要ファイル**
1. `train.py`:
   - モデルの学習およびテストを行うメインスクリプト。
   - 学習済みモデルを `tudui_{epoch}.pth` として保存。
2. `inference.py`:
   - 学習済みモデルをロードし、個別画像に対する推論を行います。
3. `model.py`:
   - `Tudui` モデルの定義ファイル。注意機構やDropoutを含む。

---

## 🖼️ **TensorBoardの使用**
学習ログを確認するには、以下のコマンドを実行します：
```bash
tensorboard --logdir=./logs_train
```
ブラウザで開き、学習損失や正解率のグラフを確認できます。

