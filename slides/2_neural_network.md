# 第?週<br>ニューラルネットワーク

機械学習帳 7章・8章

---

## 目次

1. ニューラルネットワークとは
1. 回帰、分類のニューラルネットワークによる表現
1. AND, OR, NANDの表現
1. 多層NNの表現力
1. NNの学習方法

Note:
全体15分。1〜3で3分、4で3.5分、5で7.5分、まとめ0.5分。

---

## 1. ニューラルネットワークとは

--

### 1.1 ニューラルネットワークとは

- 線形変換と活性化関数を組み合わせて、入力から出力を計算する関数
- 単層: 入力を線形変換し、活性化関数 $g$ を通す
- 多層: ある単層NNの出力を、別の単層NNの入力につなぐ

<img src="images/neural_network/networks.png" alt="単層と2層のニューラルネットワークの図" class="nn-figure">

--

### 1.2 活性化関数

| 名前 | 式 |
| --- | --- |
| シグモイド | $\sigma(a) = \frac{1}{1 + e^{-a}}$ |
| tanh | $\tanh a = \frac{e^a - e^{-a}}{e^a + e^{-a}} = 2\sigma(2a) - 1$ |
| ReLU | $\mathrm{ReLU}(a) = \max(0, a)$ |
| ステップ | $a > 0$ なら $1$、それ以外は $0$ |

- 非線形な活性化関数がなければ多層は単層で表現可能
- ステップ関数は微分が $0$ なので、学習では使いにくい

Note:
1と合わせて1.5分。

---

## 2. 回帰、分類のニューラルネットワークによる表現

--

### 2.1 これまでのモデルはすべて単層NN

$$
\boldsymbol{y} = g(\boldsymbol{x}\boldsymbol{W} + \boldsymbol{b})
$$

| モデル | 出力の次元 | 活性化関数 $g$ | 損失関数 |
| --- | --- | --- | --- |
| 単回帰 | $1$（入力も $1$） | 恒等写像 | 二乗誤差 |
| 重回帰 | $1$ | 恒等写像 | 二乗誤差 |
| 線形二値分類 | $1$ | シグモイド | 二値クロスエントロピー |
| 線形多クラス分類 | クラス数 | ソフトマックス | クロスエントロピー |

Note:
1分。前回までの担当者の回とのつながりを示す。

---

## 3. AND, OR, NANDの表現

--

### 3.1 論理ゲートは単層NNで表現できる

$\hat{y} = \mathrm{step}(w_1 x_1 + w_2 x_2 + b)$

<img src="images/neural_network/logic_gates.png" alt="AND、OR、NAND、XORの入力と分離直線" class="nn-figure">

- XORは直線1本では分けられない（線形分離不能）

Note:
30秒。重みと検算1例だけ。

---

## 4. 多層NNの表現力

--

### 4.1 2層にするとXORを表現できる

XOR $=$ OR $\land$ NAND

$$
\begin{aligned}
h_1 &= \mathrm{step}(x_1 + x_2 - 0.5) &&\text{(OR)} \\\\
h_2 &= \mathrm{step}(-x_1 - x_2 + 1.5) &&\text{(NAND)} \\\\
y &= \mathrm{step}(h_1 + h_2 - 1.5) &&\text{(AND)}
\end{aligned}
$$

<img src="images/neural_network/xor_hidden_space.png" alt="入力空間と中間層の空間でのXORの点の配置" class="nn-figure">

中間層が線形分離できる空間へ写している

--

### 4.2 任意のブール関数を表現できる

- AND, OR, NOTだけで、任意の真理値表を論理式にできる（積和標準形）
  1. 出力が $1$ の行ごとに、入力が $1$ の変数はそのまま、$0$ の変数は否定してANDでつなぐ
  1. それらをすべてORでつなぐ
- 例: XOR $= (\lnot x_1 \land x_2) \lor (x_1 \land \lnot x_2)$
- AND, OR, NOTは単層NNで表現できる
- よって、単層NNを多段にすれば任意のブール関数を表現できる

--

### 4.3 万能近似定理

隠れ層が1層のNNで、任意の連続関数を近似できる

$$
F(x) = \sum_{i=1}^{M} v_i \sigma(w_i x + b_i)
$$

<img src="images/neural_network/sigmoid_rectangle.png" alt="wを大きくしたシグモイド関数と、2つのシグモイド関数で作った矩形" class="nn-figure">

- $w$ 大で段差、位置は $x = -b/w$、高さは $v$。逆向きの段差を足すと矩形

--

### 4.3 矩形を並べて近似する

<img src="images/neural_network/universal_approximation.png" alt="矩形の数を増やして目標の関数を近似する様子" class="nn-figure">

- 矩形1つは隠れユニット2つで作れる
- 隠れ層内のユニットを増やすほど、細かく近似できる
- 定理が保証するのはパラメータの存在だけで、学習で見つかるとは限らない

Note:
4全体で3.5分。

---

## 5. NNの学習方法

--

### 5.1 勾配をどう求めるか

勾配法でパラメータを更新する

$$
\boldsymbol{w} \leftarrow \boldsymbol{w} - \eta \frac{\partial L}{\partial \boldsymbol{w}}
$$

- 単回帰なら勾配を手計算で導出できた
- 多層になると合成関数が深くなり、手で導出するのは大変
- 計算グラフを使えば、機械的な手順で求められる（自動微分）

--

### 5.2 計算グラフと前向き計算

$$
y=ax+b,\quad(a,b)=(-2,4),\quad x=3
$$

<img src="images/neural_network/graph_linear_forward.png" alt="y=ax+bの計算グラフと前向き計算の値" class="nn-figure">

- 葉（変数）から根（出力）へ、途中の値を保持しながら計算する

--

### 5.3 後ろ向き自動微分

<img src="images/neural_network/graph_linear_backward.png" alt="y=ax+bの計算グラフと各ノードでの偏微分の値" class="nn-figure">

- 根に $1$ を置き、根から葉へ向かって計算する
- ノードを通過するたびに「すでに求めた偏微分」と「そのノードの局所的な偏微分」を掛ける（連鎖律）

$$
\frac{\partial y}{\partial a} = \frac{\partial y}{\partial \alpha} \frac{\partial \alpha}{\partial a} = 1 \times x = 3
$$

--

### 5.4 単層NNの損失関数で考える

二値クロスエントロピー（正例 $y = 1$ とする）

$$
\hat{l} = -\log p, \quad p = \sigma(w_1 x_1 + w_2 x_2)
$$

$(x_1, x_2) = (1, -1)$、$(w_1, w_2) = (1, 0.5)$（本書と同じ例）

<img src="images/neural_network/graph_bce_forward.png" alt="二値クロスエントロピー損失の計算グラフと前向き計算の値" class="nn-figure">

--

### 5.5 各辺の局所的な偏微分

<img src="images/neural_network/graph_bce_local.png" alt="計算グラフの各辺に書いた局所的な偏微分" class="nn-figure">

- $-\log p$ の微分: $-\frac{1}{p}$
- シグモイドの微分: $\sigma'(\mu) = \sigma(\mu)(1 - \sigma(\mu)) = p(1 - p)$
- 掛け算は相手の値、足し算は $1$

各ノードは、自分の入出力だけ見れば微分できる

--

### 5.6 根から順に掛けていく

<img src="images/neural_network/graph_bce_backward.png" alt="二値クロスエントロピー損失の計算グラフと各ノードでの偏微分の値" class="nn-figure">

$$
\begin{aligned}
\frac{\partial \hat{l}}{\partial \mu} &= 1 \times \left(-\frac{1}{p}\right) \times p(1 - p) = p - 1 = -0.3775 \\\\
\frac{\partial \hat{l}}{\partial w_1} &= (p - 1)\, x_1 = -0.3775, \quad
\frac{\partial \hat{l}}{\partial w_2} = (p - 1)\, x_2 = 0.3775
\end{aligned}
$$

--

### 5.7 手で導出した勾配と一致する

負例 $y = 0$ も含めて計算すると

$$
\frac{\partial \hat{l}}{\partial \boldsymbol{w}} = (p - y)\, \boldsymbol{x}
$$

- 線形二値分類の勾配と同じ形になる
- 局所的な偏微分さえ分かれば、どんな形の関数でも同じ手順で勾配が求まる
- 本書では $\sigma$ をさらに細かいノード（$\exp$、$+1$、逆数）に分けているが、結果は同じ

--

### 5.8 多層NNでの誤差逆伝播法

2層NN: $\boldsymbol{h} = g(\boldsymbol{x}\boldsymbol{W}^{(1)} + \boldsymbol{b}^{(1)})$、$\hat{y} = \sigma(\boldsymbol{h}\boldsymbol{w}^{(2)} + b^{(2)})$

$$
\frac{\partial \hat{l}}{\partial \boldsymbol{W}^{(1)}} =
\underbrace{\frac{\partial \hat{l}}{\partial \boldsymbol{h}}}_{\text{2層目の計算で求め済み}}
\frac{\partial \boldsymbol{h}}{\partial \boldsymbol{W}^{(1)}}
$$

- 計算グラフが入力側に伸びるだけで、手順は同じ
- 出力側で求めた偏微分を、入力側のすべてのパラメータで使い回す
- 前向きに1回、後ろ向きに1回計算すれば、全パラメータの勾配が求まる
- これをNNに適用したものが誤差逆伝播法（backpropagation）

--

### 5.9 Pythonでライブラリを用いた実装

PyTorchでの学習ループの例

```python
for x, y in loader:
    y_pred = model(x)          # 前向き計算
    loss = loss_fn(y_pred, y)  # 損失を計算
    optimizer.zero_grad()      # 前回の勾配を消す
    loss.backward()            # 後ろ向き自動微分
    optimizer.step()           # 勾配法で更新
```

`loss.backward()` が、ここまでの計算グラフ上の手順を自動で行う

Note:
5全体で7.5分。時間が足りなければ5.9は飛ばす。

---

## まとめ

- 回帰・分類のモデルは、すべて単層NNとして書ける
- 単層NNは線形分離可能なものしか表現できないが、層を重ねると
  - XOR、任意のブール関数を表現できる
  - 任意の連続関数を近似できる（万能近似定理）
- 学習では、計算グラフ上の後ろ向き自動微分で勾配を求め、勾配法で更新する（誤差逆伝播法）
