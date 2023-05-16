{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 143,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.linear_model import SGDClassifier\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.metrics import accuracy_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 144,
   "metadata": {},
   "outputs": [],
   "source": [
    "dataset = pd.read_csv(\"~/datatraining.txt\")\n",
    "df = pd.DataFrame(dataset)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 145,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>date</th>\n",
       "      <th>Temperature</th>\n",
       "      <th>Humidity</th>\n",
       "      <th>Light</th>\n",
       "      <th>CO2</th>\n",
       "      <th>HumidityRatio</th>\n",
       "      <th>Occupancy</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2015-02-04 17:51:00</td>\n",
       "      <td>23.18</td>\n",
       "      <td>27.2720</td>\n",
       "      <td>426.0</td>\n",
       "      <td>721.250000</td>\n",
       "      <td>0.004793</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2015-02-04 17:51:59</td>\n",
       "      <td>23.15</td>\n",
       "      <td>27.2675</td>\n",
       "      <td>429.5</td>\n",
       "      <td>714.000000</td>\n",
       "      <td>0.004783</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2015-02-04 17:53:00</td>\n",
       "      <td>23.15</td>\n",
       "      <td>27.2450</td>\n",
       "      <td>426.0</td>\n",
       "      <td>713.500000</td>\n",
       "      <td>0.004779</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2015-02-04 17:54:00</td>\n",
       "      <td>23.15</td>\n",
       "      <td>27.2000</td>\n",
       "      <td>426.0</td>\n",
       "      <td>708.250000</td>\n",
       "      <td>0.004772</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>2015-02-04 17:55:00</td>\n",
       "      <td>23.10</td>\n",
       "      <td>27.2000</td>\n",
       "      <td>426.0</td>\n",
       "      <td>704.500000</td>\n",
       "      <td>0.004757</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8139</th>\n",
       "      <td>2015-02-10 09:29:00</td>\n",
       "      <td>21.05</td>\n",
       "      <td>36.0975</td>\n",
       "      <td>433.0</td>\n",
       "      <td>787.250000</td>\n",
       "      <td>0.005579</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8140</th>\n",
       "      <td>2015-02-10 09:29:59</td>\n",
       "      <td>21.05</td>\n",
       "      <td>35.9950</td>\n",
       "      <td>433.0</td>\n",
       "      <td>789.500000</td>\n",
       "      <td>0.005563</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8141</th>\n",
       "      <td>2015-02-10 09:30:59</td>\n",
       "      <td>21.10</td>\n",
       "      <td>36.0950</td>\n",
       "      <td>433.0</td>\n",
       "      <td>798.500000</td>\n",
       "      <td>0.005596</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8142</th>\n",
       "      <td>2015-02-10 09:32:00</td>\n",
       "      <td>21.10</td>\n",
       "      <td>36.2600</td>\n",
       "      <td>433.0</td>\n",
       "      <td>820.333333</td>\n",
       "      <td>0.005621</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8143</th>\n",
       "      <td>2015-02-10 09:33:00</td>\n",
       "      <td>21.10</td>\n",
       "      <td>36.2000</td>\n",
       "      <td>447.0</td>\n",
       "      <td>821.000000</td>\n",
       "      <td>0.005612</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>8143 rows × 7 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                     date  Temperature  Humidity  Light         CO2  \\\n",
       "1     2015-02-04 17:51:00        23.18   27.2720  426.0  721.250000   \n",
       "2     2015-02-04 17:51:59        23.15   27.2675  429.5  714.000000   \n",
       "3     2015-02-04 17:53:00        23.15   27.2450  426.0  713.500000   \n",
       "4     2015-02-04 17:54:00        23.15   27.2000  426.0  708.250000   \n",
       "5     2015-02-04 17:55:00        23.10   27.2000  426.0  704.500000   \n",
       "...                   ...          ...       ...    ...         ...   \n",
       "8139  2015-02-10 09:29:00        21.05   36.0975  433.0  787.250000   \n",
       "8140  2015-02-10 09:29:59        21.05   35.9950  433.0  789.500000   \n",
       "8141  2015-02-10 09:30:59        21.10   36.0950  433.0  798.500000   \n",
       "8142  2015-02-10 09:32:00        21.10   36.2600  433.0  820.333333   \n",
       "8143  2015-02-10 09:33:00        21.10   36.2000  447.0  821.000000   \n",
       "\n",
       "      HumidityRatio  Occupancy  \n",
       "1          0.004793          1  \n",
       "2          0.004783          1  \n",
       "3          0.004779          1  \n",
       "4          0.004772          1  \n",
       "5          0.004757          1  \n",
       "...             ...        ...  \n",
       "8139       0.005579          1  \n",
       "8140       0.005563          1  \n",
       "8141       0.005596          1  \n",
       "8142       0.005621          1  \n",
       "8143       0.005612          1  \n",
       "\n",
       "[8143 rows x 7 columns]"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 146,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df.iloc[:,1:-1].values\n",
    "y = df.iloc[:,-1].values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 114,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(8143, 5)"
      ]
     },
     "execution_count": 114,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 115,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(8143,)"
      ]
     },
     "execution_count": 115,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Splitting the data in 70% training, 30% testing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 147,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3) "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Halfspace classifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 148,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Perceptron(eta0=1, max_iter=70)"
      ]
     },
     "execution_count": 148,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.linear_model import Perceptron\n",
    "perceptron_halfspace = Perceptron(tol=1e-3, max_iter=70, eta0=1, random_state=0)\n",
    "perceptron_halfspace.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 149,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9805263157894737"
      ]
     },
     "execution_count": 149,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "perceptron_halfspace.score(X_train, y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Logistic Regression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 150,
   "metadata": {},
   "outputs": [],
   "source": [
    "lr = LogisticRegression()\n",
    "logReg = lr.fit(X_train, y_train)\n",
    "predictions = logReg.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 151,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9847368421052631"
      ]
     },
     "execution_count": 151,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "logReg.score(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 152,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9889480147359804"
      ]
     },
     "execution_count": 152,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "accuracy_score(y_test, predictions)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### SVM classifier using a linear kernel"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 153,
   "metadata": {},
   "outputs": [],
   "source": [
    "svm_linear = SVC(kernel =\"linear\").fit(X_train, y_train)\n",
    "predictions1 = svm_linear.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 154,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9850877192982456"
      ]
     },
     "execution_count": 154,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "svm_linear.score(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 155,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9889480147359804"
      ]
     },
     "execution_count": 155,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "accuracy_score(y_test, predictions1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[  36  222  292  344  353  357  365  367  715  782  825  845  943  959\n",
      "  961  994 1074 1128 1151 1169 1190 1275 1303 1359 1507 1545 1595 1702\n",
      " 1708 1744 1850 1851 1871 1881 1926 1971 1976 2030 2041 2150 2196 2212\n",
      " 2265 2410 2421 2544 2619 2777 2816 2854 2962 3014 3039 3118 3145 3212\n",
      " 3217 3242 3266 3277 3311 3421 3495 3509 3545 3581 3630 3798 3881 4003\n",
      " 4039 4045 4126 4219 4236 4286 4359 4370 4387 4395 4429 4468 4496 4502\n",
      " 4549 4580 4582 4585 4593 4612 4736 4774 4790 4809 4870 4943 4990 5017\n",
      " 5020 5023 5033 5111 5163 5170 5217 5231 5254 5277 5341 5454 5488 5569\n",
      " 5617 5652  241  257  263  293  302  343  462  515  686  726  749  766\n",
      "  767  826  836  939  991  998 1017 1058 1059 1105 1119 1123 1213 1284\n",
      " 1300 1304 1335 1354 1400 1451 1454 1456 1461 1497 1565 1721 1784 1808\n",
      " 1817 1876 1909 1978 1990 2080 2128 2135 2138 2264 2305 2309 2345 2356\n",
      " 2360 2381 2403 2443 2469 2491 2498 2526 2595 2737 2784 2861 2879 2925\n",
      " 3111 3159 3222 3420 3468 3493 3535 3602 3611 3625 3649 3663 3728 3755\n",
      " 3860 3867 3894 3974 4058 4128 4129 4148 4160 4200 4246 4283 4430 4525\n",
      " 4543 4591 4804 4860 5121 5130 5141 5158 5197 5296 5359 5391 5423 5450\n",
      " 5497 5513 5642 5682 5696]\n"
     ]
    }
   ],
   "source": [
    "support_vector_indices_linear1 = svm_linear.support_\n",
    "print(support_vector_indices_linear1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 156,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[116 116]\n"
     ]
    }
   ],
   "source": [
    "# number of support vectors per class\n",
    "support_vectors_per_class = svm_linear.n_support_\n",
    "print(support_vectors_per_class)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### SVM classifier using a Polynomial kernel"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 157,
   "metadata": {},
   "outputs": [],
   "source": [
    "svm_poly = SVC(kernel =\"poly\", degree = 3).fit(X_train, y_train)\n",
    "predictions2 = svm_poly.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 158,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9791228070175438"
      ]
     },
     "execution_count": 158,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "svm_poly.score(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 159,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9856733524355301"
      ]
     },
     "execution_count": 159,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "accuracy_score(y_test, predictions2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 160,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[  41   64  133  163  186  188  203  249  346  430  535  549  608  653\n",
      "  677  704  727  738  749  757  766  827  852  875  945  964  973 1004\n",
      " 1037 1046 1115 1152 1172 1196 1212 1250 1258 1291 1302 1303 1331 1344\n",
      " 1386 1392 1401 1437 1495 1506 1511 1553 1554 1569 1579 1584 1603 1633\n",
      " 1638 1647 1670 1707 1710 1780 1791 1842 1845 1879 1926 1932 1996 2006\n",
      " 2009 2029 2044 2071 2138 2174 2176 2185 2202 2233 2285 2308 2331 2351\n",
      " 2361 2381 2383 2450 2462 2463 2465 2481 2490 2505 2664 2667 2689 2707\n",
      " 2747 2790 2832 2837 2851 2885 2906 2915 2952 3026 3036 3039 3053 3068\n",
      " 3086 3101 3127 3200 3206 3221 3350 3368 3394 3401 3424 3462 3525 3532\n",
      " 3539 3549 3555 3557 3591 3607 3621 3655 3702 3704 3748 3767 3795 3810\n",
      " 3851 3867 3881 3884 3934 3937 3950 4085 4125 4166 4207 4239 4247 4264\n",
      " 4318 4379 4388 4414 4445 4450 4459 4492 4498 4565 4576 4776 4809 4847\n",
      " 4906 4909 4926 4957 4961 4964 4985 4987 4997 5008 5051 5085 5089 5102\n",
      " 5152 5153 5156 5157 5168 5197 5264 5351 5352 5385 5401 5408 5434 5439\n",
      " 5482 5500 5521 5540 5543 5612 5646 5653 5676   13   28   34   63   72\n",
      "   84   89   92   97  104  162  172  193  294  302  315  401  417  440\n",
      "  474  485  491  508  543  548  555  563  569  582  593  596  622  678\n",
      "  699  781  826  846  877  885  910  953 1044 1075 1091 1141 1174 1251\n",
      " 1317 1324 1325 1333 1337 1399 1415 1476 1485 1515 1536 1556 1567 1574\n",
      " 1585 1591 1601 1611 1656 1671 1672 1674 1752 1769 1835 1890 1928 1931\n",
      " 1954 1980 2014 2034 2079 2109 2132 2136 2146 2208 2250 2296 2302 2323\n",
      " 2365 2416 2430 2469 2484 2526 2545 2546 2614 2649 2670 2677 2746 2816\n",
      " 2847 2861 2878 2886 2918 2944 2953 2972 3045 3046 3074 3105 3122 3129\n",
      " 3163 3174 3179 3196 3228 3241 3301 3330 3331 3367 3380 3392 3403 3434\n",
      " 3453 3563 3606 3687 3743 3830 3944 3988 4066 4073 4078 4223 4237 4259\n",
      " 4281 4317 4340 4343 4344 4350 4373 4392 4399 4401 4426 4439 4442 4447\n",
      " 4533 4605 4622 4624 4671 4688 4717 4734 4737 4743 4771 4840 4853 4861\n",
      " 4903 4929 5006 5064 5140 5150 5159 5180 5181 5259 5262 5328 5338 5339\n",
      " 5348 5354 5364 5398 5420 5505 5519 5529 5530 5554 5601 5603 5614 5666\n",
      " 5680 5691 5694]\n"
     ]
    }
   ],
   "source": [
    "support_vector_indices_poly1 = svm_poly.support_\n",
    "print(support_vector_indices_poly1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 161,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[205 204]\n"
     ]
    }
   ],
   "source": [
    "# number of support vectors per class\n",
    "support_vectors_per_class_poly = svm_poly.n_support_\n",
    "print(support_vectors_per_class_poly)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### SVM classifier using a Gaussian kernel"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 162,
   "metadata": {},
   "outputs": [],
   "source": [
    "svm_gaus = SVC(kernel = \"rbf\", C=100).fit(X_train, y_train) \n",
    "#C as a part of regularisation for better performance\n",
    "predictions3 = svm_gaus.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 163,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9873684210526316"
      ]
     },
     "execution_count": 163,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "svm_gaus.score(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 164,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9914040114613181"
      ]
     },
     "execution_count": 164,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "accuracy_score(y_test, predictions3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 165,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[  64  133  186  249  276  346  430  677  727  738  827  964 1037 1046\n",
      " 1115 1152 1172 1212 1250 1303 1344 1386 1392 1495 1584 1638 1647 1842\n",
      " 1845 1879 1996 2009 2029 2071 2138 2154 2174 2331 2361 2383 2463 2465\n",
      " 2481 2490 2573 2664 2832 2885 2906 2915 2952 3039 3053 3101 3221 3394\n",
      " 3555 3621 3655 3795 3851 3867 3934 4085 4166 4239 4318 4445 4459 4492\n",
      " 4498 4565 4847 4906 4985 5089 5152 5153 5157 5168 5197 5264 5408 5439\n",
      " 5482 5540 5646 5653   63   89  162  172  258  265  306  315  332  339\n",
      "  464  500  555  564  596  678  699  715  793  846  953  966  991 1119\n",
      " 1199 1293 1408 1455 1476 1536 1591 1835 1890 1931 2014 2034 2079 2117\n",
      " 2132 2292 2306 2310 2365 2484 2507 2546 2593 2766 2780 2847 2861 3060\n",
      " 3122 3163 3174 3241 3286 3322 3331 3413 3415 3496 3504 3521 3617 3985\n",
      " 4046 4073 4140 4237 4280 4344 4373 4419 4421 4542 4598 4650 4748 4805\n",
      " 4838 4840 4903 4911 5159 5256 5307 5338 5376 5519 5526 5607]\n"
     ]
    }
   ],
   "source": [
    "support_vector_indices_gaussian1 = svm_gaus.support_\n",
    "print(support_vector_indices_gaussian1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Logistic Regression using the SGD"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 166,
   "metadata": {},
   "outputs": [],
   "source": [
    "lr_sgd = SGDClassifier(loss = \"log\").fit(X_train, y_train)\n",
    "predictions4 = lr_sgd.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 167,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9893573475235367"
      ]
     },
     "execution_count": 167,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "accuracy_score(y_test, predictions4)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Splitting the data in 80% training, 20% testing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 168,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size = 0.2) "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Halfspace classifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 169,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Perceptron(eta0=1, max_iter=70)"
      ]
     },
     "execution_count": 169,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.linear_model import Perceptron\n",
    "perceptron_halfspace = Perceptron(tol=1e-3, max_iter=70, eta0=1, random_state=0)\n",
    "perceptron_halfspace.fit(X_train1, y_train1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 171,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9706785385323917"
      ]
     },
     "execution_count": 171,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "perceptron_halfspace.score(X_train1, y_train1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Logistic Regression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 172,
   "metadata": {},
   "outputs": [],
   "source": [
    "lr = LogisticRegression()\n",
    "logReg = lr.fit(X_train1, y_train1)\n",
    "predictions = logReg.predict(X_test1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 173,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9864947820748926"
      ]
     },
     "execution_count": 173,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "accuracy_score(y_test1, predictions)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### SVM classifier using a linear kernel"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 174,
   "metadata": {},
   "outputs": [],
   "source": [
    "svm_linear = SVC(kernel =\"linear\").fit(X_train1, y_train1)\n",
    "predictions1 = svm_linear.predict(X_test1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 175,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9840392879066913"
      ]
     },
     "execution_count": 175,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "accuracy_score(y_test1, predictions1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 176,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[   2    7   32   51   75  103  153  252  289  361  391  402  519  523\n",
      "  629  657  699  704  803  807  932  938 1041 1084 1093 1180 1198 1261\n",
      " 1413 1562 1581 1731 1758 1759 1855 1860 1871 1911 2035 2077 2091 2226\n",
      " 2282 2295 2411 2428 2451 2590 2651 2710 2731 2930 2965 3014 3061 3155\n",
      " 3306 3543 3557 3643 3647 3690 3746 3770 3789 3824 3991 4026 4044 4103\n",
      " 4161 4249 4283 4474 4476 4520 4523 4556 4663 4674 4780 4789 4900 4924\n",
      " 4994 5006 5039 5128 5171 5327 5345 5349 5394 5400 5592 5665 5688 5727\n",
      " 5801 5855 5876 5919 5967 6010 6016 6072 6074 6116 6145 6169 6256 6364\n",
      " 6369 6480   98  184  204  263  281  323  333  335  362  372  395  506\n",
      "  554  594  661  684  731  740  753  754  772  791  820  913  985  988\n",
      " 1010 1042 1231 1253 1278 1386 1389 1397 1602 1631 1654 1725 1753 1830\n",
      " 1945 1947 1962 2004 2200 2220 2248 2305 2376 2703 2729 2978 3006 3036\n",
      " 3063 3167 3199 3203 3217 3252 3254 3281 3336 3361 3400 3407 3591 3607\n",
      " 3654 3745 3761 3840 3968 4019 4127 4215 4233 4245 4467 4551 4585 4592\n",
      " 4779 4798 4920 4940 4988 5137 5241 5251 5269 5300 5301 5350 5356 5414\n",
      " 5543 5552 5573 5653 5662 5697 5721 5766 5854 5882 6018 6121 6220 6272\n",
      " 6282 6298 6403 6429]\n"
     ]
    }
   ],
   "source": [
    "support_vector_indices_linear2 = svm_linear.support_\n",
    "print(support_vector_indices_linear2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### SVM classifier using a Polynomial kernel"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 177,
   "metadata": {},
   "outputs": [],
   "source": [
    "svm_poly = SVC(kernel =\"poly\", degree = 3).fit(X_train1, y_train1)\n",
    "predictions2 = svm_poly.predict(X_test1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 178,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9815837937384899"
      ]
     },
     "execution_count": 178,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "accuracy_score(y_test1, predictions2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 179,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[   2    7   32   36   51   69   75  153  229  232  272  289  336  361\n",
      "  369  376  402  408  433  447  519  524  629  632  657  691  699  704\n",
      "  707  724  757  803  807  847  852  885  888  932  938 1084 1093 1097\n",
      " 1101 1129 1142 1156 1180 1198 1223 1260 1261 1413 1431 1448 1490 1528\n",
      " 1562 1581 1615 1658 1723 1731 1758 1759 1818 1834 1836 1844 1855 1860\n",
      " 1871 1883 1911 1991 2013 2035 2077 2092 2232 2282 2295 2411 2427 2428\n",
      " 2451 2461 2465 2525 2590 2637 2651 2688 2710 2731 2736 2743 2748 2812\n",
      " 2821 2930 2936 2939 2944 2950 2953 3014 3061 3155 3288 3306 3348 3398\n",
      " 3532 3543 3557 3561 3643 3690 3736 3746 3753 3760 3765 3770 3789 3800\n",
      " 3852 3932 3934 3956 3991 4024 4026 4044 4079 4094 4103 4161 4249 4259\n",
      " 4283 4304 4347 4382 4447 4474 4476 4519 4520 4546 4556 4608 4663 4665\n",
      " 4780 4789 4796 4818 4900 4924 4957 4976 4983 4994 5006 5039 5050 5051\n",
      " 5133 5150 5171 5316 5318 5327 5334 5345 5349 5394 5400 5496 5592 5665\n",
      " 5688 5727 5747 5801 5841 5855 5876 5919 5929 5950 5956 5967 6016 6037\n",
      " 6074 6116 6145 6166 6169 6237 6256 6260 6364 6377 6382 6396 6479 6480\n",
      " 6489    8   28  105  116  144  184  204  234  237  267  281  323  333\n",
      "  364  372  395  399  452  485  506  533  562  577  658  684  698  731\n",
      "  753  772  791  836  871  913  940  970  985  988 1010 1013 1050 1105\n",
      " 1134 1150 1253 1278 1340 1389 1482 1494 1571 1602 1605 1695 1699 1753\n",
      " 1830 1945 1947 1962 1967 1985 1995 2004 2042 2086 2148 2153 2196 2200\n",
      " 2220 2305 2306 2311 2358 2360 2371 2373 2376 2399 2405 2423 2514 2551\n",
      " 2572 2597 2603 2632 2643 2653 2803 2874 3006 3036 3063 3065 3124 3157\n",
      " 3167 3203 3209 3252 3254 3255 3256 3260 3330 3361 3400 3407 3415 3517\n",
      " 3546 3548 3566 3592 3607 3654 3742 3745 3761 3778 3806 3817 3840 3896\n",
      " 3930 3968 4003 4019 4061 4062 4117 4152 4201 4233 4295 4316 4351 4367\n",
      " 4381 4481 4511 4551 4585 4591 4592 4630 4643 4684 4755 4779 4798 4874\n",
      " 4885 4890 4940 4953 4964 5038 5101 5182 5208 5213 5235 5241 5300 5342\n",
      " 5350 5352 5375 5376 5387 5410 5498 5543 5573 5624 5653 5662 5672 5684\n",
      " 5697 5705 5720 5764 5766 5780 5825 5854 5882 5894 5895 5934 6000 6018\n",
      " 6059 6097 6121 6176 6220 6272 6275 6282 6294 6298 6299 6395 6429 6435\n",
      " 6451 6460]\n"
     ]
    }
   ],
   "source": [
    "support_vector_indices_poly2 = svm_poly.support_\n",
    "print(support_vector_indices_poly2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### SVM classifier using a Gaussian kernel"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 180,
   "metadata": {},
   "outputs": [],
   "source": [
    "svm_gaus = SVC(kernel = \"rbf\", C=100).fit(X_train1, y_train1)\n",
    "predictions3 = svm_gaus.predict(X_test1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 181,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9871086556169429"
      ]
     },
     "execution_count": 181,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "accuracy_score(y_test1, predictions3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 182,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[   2    7   32   51   75  153  289  433  519  629  657  699  704  803\n",
      "  807  932  938 1084 1093 1180 1198 1223 1261 1448 1490 1581 1758 1855\n",
      " 2035 2194 2282 2295 2411 2428 2465 2590 2651 2731 2736 2930 2939 3014\n",
      " 3061 3155 3306 3557 3690 3746 3789 3956 3991 4026 4044 4103 4161 4239\n",
      " 4283 4474 4476 4520 4663 4780 4789 4900 4994 5006 5039 5171 5327 5345\n",
      " 5349 5394 5592 5614 5688 5727 5876 5919 5929 5967 6016 6074 6145 6169\n",
      " 6256 6382 6480   62  105  139  204  234  304  323  333  363  372  452\n",
      "  576  677  727  731  772  893  901  913 1060 1150 1253 1278 1315 1506\n",
      " 1532 1602 1613 1753 1898 1972 2004 2086 2161 2196 2306 2329 2371 2551\n",
      " 2603 2741 2790 2850 2856 2867 3002 3006 3094 3111 3217 3303 3474 3546\n",
      " 3592 3607 3654 3745 3817 3887 3896 3968 4350 4407 4477 4650 4656 4684\n",
      " 4902 4964 4986 4990 4999 5021 5300 5301 5350 5357 5375 5517 5653 5662\n",
      " 5697 5814 5989 6005 6018 6041 6052 6103 6119 6121 6127 6429]\n"
     ]
    }
   ],
   "source": [
    "support_vector_indices_gaussian2 = svm_gaus.support_\n",
    "print(support_vector_indices_gaussian2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Logistic Regression using the SGD"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 184,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9821976672805403"
      ]
     },
     "execution_count": 184,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "lr_sgd = SGDClassifier(loss = \"log\").fit(X_train1, y_train1)\n",
    "predictions4 = lr_sgd.predict(X_test1)\n",
    "accuracy_score(y_test1, predictions4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
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
