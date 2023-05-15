{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.model_selection import train_test_split\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "dataset = pd.read_csv(\"~/train.csv\")\n",
    "df = pd.DataFrame(dataset)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(df[\"LotArea\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(df.loc[[1]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_binary = df[[\"YearBuilt\", \"SalePrice\"]]\n",
    "df_binary"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_binary.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(df_binary.loc[df_binary[\"YearBuilt\"] > 2000])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "sorted_by_price = df_binary.sort_values(\"SalePrice\", ascending=True)\n",
    "sorted_by_price"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_average = df[[\"YearBuilt\", \"LotArea\", \"BsmtFinSF1\", \"SalePrice\"]]\n",
    "df_average.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_average[\"Average\"] = df_average[[\"LotArea\", \"BsmtFinSF1\"]].mean(axis=1)\n",
    "df_average"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_average.to_pickle(path=\"./df_average.pkl\",compression=None)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "read_pkl = pd.read_pickle(\"./df_average.pkl\")\n",
    "read_pkl"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = np.array(df_binary[\"YearBuilt\"]).reshape(-1, 1) \n",
    "y = np.array(df_binary[\"SalePrice\"]).reshape(-1, 1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_bar = np.transpose(X, axes=None)\n",
    "w = np.zeros([len(X),len(X)]) \n",
    "b = np.zeros([len(X),])\n",
    "A = X*X_bar"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def compute_error(b, m, inputs):\n",
    "    totalError = 0\n",
    "    for i in range (0, len(inputs)):\n",
    "        x = inputs[i, 0]\n",
    "        y = inputs[i, 1]\n",
    "        hypothesis = (m * x) + b\n",
    "        totalError += (hypothesis-y)**2      \n",
    "    return totalError/ (2 * float(len(inputs)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "compute_error(b, w, A)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.scatter(X_train, y_train)\n",
    "Y_pred = lr.predict(X_train)\n",
    "plt.plot(X_train, Y_pred, color='red')\n",
    "plt.title(\"Year of Built vs Sale Price\")\n",
    "plt.xlabel(\"Year of Built\")\n",
    "plt.ylabel(\"Sale Price\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "lr = LinearRegression() \n",
    "lr.fit(X_train, y_train)\n",
    "lr.score(X_test, y_test)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
