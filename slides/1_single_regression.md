# 第1週 単回帰

---

## 0. 目次

1. 単回帰とは
2. パラメータの決め方
3. 平均二乗残差を最小化する

---

## 1. 単回帰とは

--

### 1.1 単回帰の定義

- 単一の説明変数を用いて目的変数を予測するモデル
- 説明変数: 予測に用いる変数
  - 例: 最高気温、勉強時間など
- 目的変数: 予測したい変数
  - 例: アイスクリームの売上、テストの点数など

--

説明変数$x$、目的変数 $y$ とすると次のように書ける
$$
y = f(x)
$$

$f$ を「モデル」と呼ぶ

--

### 1.2 単回帰の例

- 最高気温からアイスクリームの売上を予測する
- 勉強時間からテストの点数を予測する

--

## 1.3 モデルの例：一次関数モデル
目的変数 $y$ が、説明変数 $x$ を用いて次のように表せると仮定する
$$
y = ax + b
$$

$a$, $b$ はモデルを決定するパラメータである。
以降はこの一次関数モデルで議論を進める

---

## 2. パラメータの決め方

--

### 2.1 回帰の「よさ」を評価する

- パラメーターの取り方(直線の引き方)は無数にある
- どれが「よい」直線かを評価する基準は?
<img src="images/single_regression/candidate_lines.png" alt="4点のデータに対する複数の回帰直線" class="regression-figure">
--

### 2.2 よさの測り方と「残差」

点 $(x_i, y_i)$ とモデルの予測値 $\hat{y} = f(x_i)$ のずれ
$$
\epsilon_i = y_i - f(x_i)
$$

<img src="images/single_regression/residuals.png" alt="4点のデータと回帰直線、それぞれの残差" class="regression-figure">

--

### 2.2 残差のみでは不十分

- 残差のみでは$+$と$-$が打ち消し合ってしまう
- 以下の例では、どちらも残差の合計が $0$

<img src="images/single_regression/zero_sum_residuals.png" alt="残差の合計がともに0になる2本の直線" class="regression-figure">

--

### 2.3 残差二乗平均

そこで、残差の二乗の平均を最小化することを考える。

$$
\begin{aligned}
\hat{L} &= \frac{1}{n} \sum_{i=1}^{n} (y_i - f(x_i))^2 \\\\
&= \frac{1}{n} \sum_{i=1}^{n} \left(y_i - (ax_i + b)\right)^2
\end{aligned}
$$

この値が最小になるように、$a$ と $b$ を決定することにする。

--

### 2.4 残差二乗平均の計算例

先の例では、それぞれ $\hat{L}$ の値は以下の通り
<img src="images/single_regression/mse_comparison.png" alt="2本の直線の平均二乗残差の比較" class="regression-figure">

- 左: $\hat{L} = 0.80$
- 右: $\hat{L} = 5.31$

---

## 3. 平均二乗残差を最小化する

--

### 3.1 平均二乗残差の式から

平均二乗残差は次のように $a,b$ の関数である。
$$
\begin{aligned}
\hat{L} &= \frac{1}{n} \sum_{i=1}^{n} (y_i - f(x_i))^2 \\\\
&= \frac{1}{n} \sum_{i=1}^{n} \left(y_i - (ax_i + b)\right)^2 \\\\
&= \frac{1}{n} \sum_{i=1}^{n} \left(y_i^2 - 2y_i(ax_i + b) + (ax_i + b)^2\right) \\\\
&= \frac{1}{n} \sum_{i=1}^{n} \left(y_i^2 - 2y_i(ax_i + b) + (a^2x_i^2 + 2ax_ib + b^2)\right)
\end{aligned}
$$

--

### 3.1 平均二乗残差の式から

平均を表す記号として、$\overline{c}$ を用いることにする。

$$
\frac{1}{n} \sum_{i=1}^{n} x_i = \overline{x}, \quad
$$

$$
\frac{1}{n} \sum_{i=1}^{n} x_i^2 = \overline{x^2}, \quad
$$

--

### 3.1 平均二乗残差の式から

(続き)

$$
\begin{aligned}
\frac{1}{n} \sum_{i=1}^{n} \left(y_i^2 - 2y_i(ax_i + b) + (a^2x_i^2 + 2ax_ib + b^2)\right) \\\\
= \overline{y^2} - 2a\overline{xy} - 2b\overline{y} + a^2\overline{x^2} + 2ab\overline{x} + b^2 \\\\
= \overline{x^2}a^2 + 2\overline{x}ab + b^2 - 2\overline{xy}a - 2\overline{y}b + \overline{y^2}
\end{aligned}
$$

$\overline{x}$ や $\overline{y}$は定数なので、$a,b$の二次関数であることがわかる。

--

### 3.2 偏微分による極小値計算

$\hat{L}$ が $a,b$ に関する二次関数であることが分かったので、偏微分を用いて極小値を求めることができる。(凸性については後述)

今回の場合、ある点 $(a^{\ast}, b^{\ast})$ で最小値をとるための条件は、偏微分が $0$ になることである。

$$
\left. \frac{\partial \hat{L}}{\partial a} \right|_{(a^{\ast}, b^{\ast})} = 0
$$

$$
\left. \frac{\partial \hat{L}}{\partial b} \right|_{(a^{\ast}, b^{\ast})} = 0
$$

--

### 3.2 偏微分による極小値計算

$$
\begin{cases}
\frac{\partial \hat{L}}{\partial a} = 2\overline{x^2}a + 2\overline{x}b - 2\overline{xy} = 0\\\\
\frac{\partial \hat{L}}{\partial b} = 2\overline{x}a + 2b - 2\overline{y} = 0
\end{cases}
$$

単に連立方程式を解くことで、$a^{\ast}, b^{\ast}$ を求めることができる。

$$
a^{\ast} = \frac{\overline{xy} - \overline{x}\cdot\overline{y}}{\overline{x^2} - \overline{x}^2}
$$

$$
b^{\ast} = \overline{y} - a^{\ast}\overline{x}
$$

--

### 3.3 実例

例に示した4点から

| $i$  | $(x_i, y_i)$ |  $x$  | $x^2$ |  $y$  |  $xy$  |
| ---- | :----------: | ----- | ----- | ----- | ------ |
| 1    |   $(1, 2)$   | $1  $ | $1  $ | $2  $ | $2   $ |
| 2    |   $(2, 5)$   | $2  $ | $4  $ | $5  $ | $10  $ |
| 3    |   $(3, 4)$   | $3  $ | $9  $ | $4  $ | $12  $ |
| 4    |   $(4, 7)$   | $4  $ | $16 $ | $7  $ | $28  $ |
| 平均 | $(2.5, 4.5)$ | $2.5$ | $7.5$ | $4.5$ | $13.0$ |

--

### 3.3 実例

傾き
$$
a^{\ast} = \frac{\overline{xy} - \overline{x}\cdot\overline{y}}{\overline{x^2} - \overline{x}^2}
= \frac{13.0 - 2.5\times4.5}{7.5 - 2.5^2} = 1.4
$$
切片
$$
b^{\ast} = \overline{y} - a^{\ast}\overline{x}
= 4.5 - 1.4\times2.5 = 1.0
$$

--

### 3.3 実例

平均二乗残差を最小化する回帰直線は次のとおり
$$
\hat{y} = 1.4x + 1.0
$$

<img src="images/single_regression/optimal_line.png" alt="平均二乗残差を最小化する回帰直線" class="regression-figure">

---

## 4. プログラムで計算してみる

--

## 4.1 Pythonでの計算例

Pythonの`scikit-learn`ライブラリを用いる。
```python
import numpy as np
from sklearn.linear_model import LinearRegression

# データの準備
X = np.array([[1], [2], [3], [4]])  # 説明変数
y = np.array([2, 5, 4, 7])          # 目的変数

# モデルの作成
model = LinearRegression()

# モデルの学習
model.fit(X, y)
```

<!--
ここまでで、一通りツールとして単回帰が使えるようになるはず。
以降では、統計的な話(決定係数)や、数理的な話(平均と推定値平均の一致、目的変数と残差が直交、など)に触れる。
-->

---

## 5. 回帰直線の性質


--

### 5.1 回帰直線の傾き

回帰直線の傾きは、次のように表せる

$$
a = \frac{\overline{xy} - \overline{x}\cdot\overline{y}}{\overline{x^2} - \overline{x}^2}
$$

---

## 6. まとめ

- 単回帰は、単一の説明変数を用いて目的変数を予測するモデルである
- 残差に基づく評価指標を用いてパラメータを決定する
- プログラミングではライブラリが用意されている場合が多い
---

## 参考文献
岡崎直観『機械学習帳』
1\. 単回帰
https://chokkan.github.io/mlnote/regression/01.html
