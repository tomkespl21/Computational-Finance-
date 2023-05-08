# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 21:34:32 2023

@author: Tomke
"""

# C-Exercise 06
# a)
# Use the one formula defined at the top to carry out all operations

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def CRR_AmEuPut(S_0, r, sigma, T, M, K, EU):
    delta_t = (T/M)
    beta = (1/2) * (np.exp(-r*delta_t) + np.exp((r+sigma**2)*delta_t))
    u = beta + np.sqrt(beta**2-1)
    d = (1/u)
    q = (np.exp(r*delta_t) - d) / (u-d)

    # print(f"u: {u}")
    # print(f"d: {d}")
    # print(f"q: {q}")

    stock = np.empty((M+1, M+1))
    payoff = np.empty((M+1, M+1))
    value = np.empty((M+1, M+1))

    for i in range(0, M+1):
        for j in range(0, i+1):
            # print(f'S_{j}_{i} = u^ {j} * d^{i-j}')
            stock[j, i] = S_0 * u**j * d**(i-j)
            payoff[j,i] = np.maximum(K-stock[j,i],0)
            
    # determine the payoffs at maturity
    value[:, M] = payoff[:, M]

    # Calculate reversed the value
    for i in range(M-1, -1, -1):
        for j in range(0, i+1):
            if EU == 1:
              value[j, i] = np.exp(-r*delta_t) * (q * value[j+1, i+1] + (1-q) * value[j, i+1])
            else:
                value[j,i] = np.maximum(payoff[j,i],np.exp(-r*delta_t) * (q * value[j+1, i+1] + (1-q) * value[j, i+1]))  
    # return stock, payoff, value
    return value[0, 0]
          
CRR_AmEuPut(S_0=100.00, r=0.05, sigma=0.3, T=1, M=500, K=120.00, EU=1)
# 19.989996980016112

    



# b) BS-Formula for European put options
def BlackScholes_EuPut(t: int, S_t: float, r: float, sigma: float, T: float, K: float) -> float:
    """Computes the option price in the BSM model for a european put option

    Args:
        t (int): time step
        S_t (float): current stock price
        r (float): interest rate
        sigma (float): volatility, sigma>0
        T (float): max time horizon
        K (float): strike price of the option

    Returns:
        float: Fair option price of BSM model
    """
    # Compute static values first
    # Compute d1 and d2
    d1 = (np.log(S_t/K) + (((sigma**2)/2)+r) * (T-t)) / sigma * (np.sqrt(T-t))
    d2 = d1 - sigma * (np.sqrt(T-t))

    EuPut = K * np.exp(-r * (T-t)) * norm.cdf(-d2, loc=0, scale=1) - S_t * norm.cdf(-d1, loc=0, scale=1)
    return EuPut

BlackScholes_EuPut(t=0, S_t=1.0, r=0.05, sigma=0.3, T=3, K=1.2)
#0.20963750433549244
# c) Test the function
S_0 = 100.00
t = 0
r = 0.05
sigma = 0.3
T = 1
K = 120.00


m_variants = np.arange(10, 501, 10)
eu_put_crr = model_error = np.empty(len(m_variants), dtype=float)

for i in range(0,len(m_variants),1):
    eu_put_crr[i] = CRR_AmEuPut(S_0=S_0, r=r, sigma=sigma, T=T, M=int(m_variants[i]), K=K,EU=1)

    


#for ind, m_variant in zip(range(0, len(m_variants)), m_variants):
#    eu_put_crr[ind] = CRR_AmEU(S_0=S_0, r=r, sigma=sigma, T=T, M=m_variant, K=K, EU=EU, type_=type_)

# Looks not very convincing here 
plt.plot(m_variants, eu_put_crr, color="#603086")
plt.ylabel("EU Put CRR Model Price")
plt.xlabel("M variants", fontsize=6)
plt.title("EU Put CRR Model Price vs. Number of M steps", fontsize=14)
plt.xticks(rotation=45, fontsize=6)
plt.show()


# Plot also the fair BSM model price with same parameters in same plot
plt.plot(m_variants, eu_put_crr, color="#603086")
plt.ylabel("EU Put CRR Model Price")
plt.xlabel("M variants", fontsize=6)
plt.axhline(y=BlackScholes_EuPut(t=t, S_t=S_0, r=r, sigma=sigma, T=T, K=K), xmin=0, xmax=max(m_variants), color="black")
plt.text(x=310.00, y=BlackScholes_EuPut(t=t, S_t=S_0, r=r, sigma=sigma, T=T, K=K)+0.5, s=f"BSM EU Put Price: {round(BlackScholes_EuPut(t=t, S_t=S_0, r=r, sigma=sigma, T=T, K=K), 4)}")
plt.title("EU Put CRR Model Price vs. Number of M steps", fontsize=14)
plt.xticks(rotation=45, fontsize=6)
plt.show()

# Price of Am Put in CRR model
print("Price of an American Put Option in the CRR Model", CRR_AmEuPut(S_0=S_0, r=r, sigma=sigma, T=T, M=500, K=K, EU=0))
