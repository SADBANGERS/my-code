import sys; sys.path.insert(0, "../../")
from dualbounds.generic import DualBounds

import numpy as np
import matplotlib.pyplot as plt
from utilities import generate_data, treatment, cot_estimator
from truevalue import true_vip1dim, true_vc1dim, true_vu1dim, true_vcmdim
import statistics


seed = 123
np.random.seed(seed)

b0 = np.array([[-0.6, 0.3, 0.1, 0.1, -0.3]])
b1 = np.array([[1.6, -0.8, -0.1, -0.1, 0.3]])
dy = 1
dz = 20


def main_cp(N, tvc, SIMSIZE=120, gap=True):
    ridge_res = 0
    knn_res = 0
    cot_res = []
    

    for k in range(SIMSIZE):
        # generate raw data 
        raw_data = generate_data(N, b0, b1)
        
        # generate post-treatment data
        A0, A1, data = treatment(raw_data, dy, dz)
        
        # Direct Vc estimate
        cot = cot_estimator(A0, A1)
        if gap:
            cot_res.append(abs(cot - tvc))
        else:
            cot_res.append(cot)
        
        # LL's estimate
        dbnd = DualBounds(
            f=lambda y0, y1, x: (y0 - y1) ** 2,
            covariates=data['X'],
            treatment=data['W'],
            outcome=data['y'],
            propensities=data['pis'],
            outcome_model='ridge',
        )
        
        results = dbnd.fit().results()
        LL_brd = results.at['Estimate', 'Lower']
        if gap:
            ridge_res += abs(LL_brd - tvc)
        else:
            ridge_res += LL_brd
        
        
        # LL's estimate3
        dbnd3 = DualBounds(
            f=lambda y0, y1, x: (y0 - y1) ** 2,
            covariates=data['X'],
            treatment=data['W'],
            outcome=data['y'],
            propensities=data['pis'],
            outcome_model='knn',
        )
        
        results3 = dbnd3.fit().results()
        LL_knn = results3.at['Estimate', 'Lower']
        if gap:
            knn_res += abs(LL_knn - tvc)
        else:
            knn_res += LL_knn
        
        
        print(f'{k+1}/{SIMSIZE} complete.')
    
    ridge_res /= SIMSIZE
    knn_res /= SIMSIZE
    
    
    return ridge_res, knn_res, cot_res

# compute the true cot value
tvc = true_vcmdim(b0, b1)

# main run
gap=True #record estimation error (True) or estimation value (False)

sizelist = [200, 600, 800]
ridge_reslist = []
knn_reslist = []
cot_reslist_mean = []
cot_reslist_std = []

for sz in sizelist:
    ridge_res, knn_res, cot_res = main_cp(sz, tvc, SIMSIZE=100, gap=gap)
    ridge_reslist.append(ridge_res)
    knn_reslist.append(knn_res)
    cot_reslist_mean.append(statistics.mean(cot_res))
    cot_reslist_std.append(statistics.stdev(cot_res))


plt.figure()
plt.plot(sizelist, np.array(ridge_reslist)/tvc, label='ridge', marker='o')
plt.plot(sizelist, np.array(knn_reslist)/tvc, label='knn', marker='o')
plt.plot(sizelist, np.array(cot_reslist_mean)/tvc, label='adapt', marker='o', color='purple')
# plt.errorbar(sizelist, np.array(cot_reslist_mean)/tvc, yerr=np.array(cot_reslist_std)/tvc * (500)**(-0.5), fmt='-o', capsize=5, color='purple')

if not gap: 
    plt.axhline(y=tvc, linestyle='--', label='true')

plt.xlabel('Sample size')
plt.ylabel('Average relative error')
# plt.ylim(-0.01, 0.17)
plt.legend()
# plt.savefig("new500error_well_error.pdf")
plt.show()



    

 