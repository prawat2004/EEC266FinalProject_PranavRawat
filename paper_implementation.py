#Pranav Rawat
#EEC 266 Final Project
#Due: 12/7/2025

#Implementing D_m recursive formula + confirming Normalized size bound

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
#I needed to use this because I am using Ubuntu WSL
matplotlib.use('Agg')

#bit reversals required in order to properly map the Decoding Order for Z
def bitReversal(N):
    #the amount of bits I have
    numBits = int(np.log2(N))
    #inputs from 0 - N-1
    index = np.arange(N, dtype=int)
    
    result = np.zeros(N, dtype=int)
    
    for stuff in range(numBits):
        #bit shift left, add last bit of index to result, and move on
        result = result << 1
        result = result | (index & 1)
        index = index >> 1
        
    return result

def getHammingWeights(N):
    #This just literally counts how many 1 bits there are for each binary number
    return np.array([bin(num).count('1') for num in range(N)])

#dynamic Reed Muller indices by bit reversal after sorting by hamming weights
def dRMIndices(N, K):
    hamming = getHammingWeights(N) 
    #sort the info by hamming weights
    original_info = np.argsort(hamming, kind='stable')[N-K:]
    
    inv = bitReversal(N)
    decoding_info = set(inv[original_info])
    return decoding_info

#This is a recursive calculation of the Bhattacharyya parameters
def ZValues(N, epsilon):
    if N == 1:
        return np.array([epsilon])
    
    z_previous = ZValues(N // 2, epsilon)
    z_negative = 2*z_previous-z_previous*z_previous
    z_plus = z_previous*z_previous
    return np.concatenate([z_negative, z_plus]) 


def dMRecursiveFormula(N, info_set, epsilon):

    zVals = ZValues(N, epsilon)
    inv = bitReversal(N)
    #for decoding
    z_vals = zVals[inv]
    D = np.zeros(N + 1)
    
    for m in range(N):
        d_prev = D[m]
        eps_m = z_vals[m]
        if m in info_set:
            #update if its an information bit
            D[m+1] = d_prev + eps_m
        else:
            #if its a frozen bit, update by how much it reduces the solution space
            prob = 1.0 - (2.0**(-d_prev)) if d_prev > 1e-9 else 0.0
            reduction = (1.0 - eps_m) * prob
            D[m+1] = max(0.0, d_prev - reduction)
            
    return D[1:] # Return values for stages 1 to N

#Markov Chain Approximation for figure 9, calculates the full PMF of D_m
def MarkovChainApproximation(N, K, info_set, epsilon):

    Zvals = ZValues(N, epsilon)
    inv = bitReversal(N)
    z_vals = Zvals[inv]
    
    P = np.zeros(N + 1)
    P[0] = 1.0 
    
    D_mean = np.zeros(N)
    for m in range(N):
        eps_m = z_vals[m]
        P_next = np.zeros(N+1)
        
        #only run on probabilities greater than 1*10^-15
        active_indices = np.where(P > 1e-15)[0]
        
        #information bit
        if m in info_set:

            if len(active_indices) > 0:
                # i -> i+1
                valid_next = active_indices + 1
                mask = valid_next < N+1
                P_next[valid_next[mask]] += P[active_indices[mask]] * eps_m
                
                # i -> i
                P_next[active_indices] += P[active_indices] * (1.0 - eps_m)
        #frozen bit
        else:
            prob = 1.0 - (2.0**(-active_indices))
            prob[active_indices == 0] = 0.0 
            
            reduced = (1.0 - eps_m) * prob   
            P_next[active_indices] += P[active_indices] * (1-reduced)
            actualReduced = active_indices > 0
            toReduce = active_indices[actualReduced]
            P_next[toReduce - 1] += P[toReduce] * reduced[actualReduced]
            
        P = P_next
        P = P/np.sum(P)
        
        # Calculate Expected Value
        D_mean[m] = np.sum(np.arange(N+1) * P)
            
    return D_mean
#plotting all the figures
def generateFigures():
    
    N_start = 8192
    K_start = 8192 // 2
    erasure = 0.48
    
    print(f"Figure 9 Data Start")
    info_set_9 = dRMIndices(N_start, K_start)
    d_rec_9 = dMRecursiveFormula(N_start, info_set_9, erasure)
    d_markov_9 = MarkovChainApproximation(N_start, K_start, info_set_9, erasure)
    
    #figure 11 goes up to 2^23, this is just getting all of the values in an array
    Ns_FIG11 = [2**i for i in range(9, 24, 2)] 
    print("figure 11 Data Start")
    fig11_data = []
    for N_val in Ns_FIG11:
        print(f"Processing N={N_val}...")
        K_val = N_val // 2
        info_set = dRMIndices(N_val, K_val)
        d_vals = dMRecursiveFormula(N_val, info_set, erasure)
        fig11_data.append((N_val, d_vals))

    plt.style.use('default') 
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    #plotting figure 9
    ax9 = axes[0]
    x_axis_9 = np.arange(1, N_start + 1)
    #plotting, I made markov chain dashed red like the paper, and the mean formula bluish
    ax9.plot(x_axis_9, d_markov_9, color="#FF0000", linestyle='--', linewidth=3, label='Markov Chain Approximation') 
    ax9.plot(x_axis_9, d_rec_9, color="#0D00FF", linestyle='-', linewidth=2, alpha=0.9, label='Recursive Mean Formula') 
    #basic title, axis, and legend stuff
    ax9.set_title(f'Figure 9: dRM code N={N_start}, epsilon={erasure})')
    ax9.set_xlabel('W', fontsize=14)
    ax9.set_ylabel('Dm', fontsize=14)
    ax9.legend(loc='upper right')

    
    #figure 11 plotting
    ax11 = axes[1]
    #just an array of colors to mimic the paper
    colors = plt.cm.plasma(np.linspace(0.3, 0.8, len(Ns_FIG11)))
    #iterating through every N value and plotting it
    for index, (N_plot, d) in enumerate(fig11_data):
        w = np.linspace(0, 1, N_plot) 
        d_norm = d / N_plot      
        label_text = f'N={N_plot}'
        ax11.plot(w, d_norm, color=colors[index], linewidth=2.5, label=label_text)
    #title stuff for figure 11
    ax11.set_title(f'Figure 11:Normalized Dm vs Normalized Decoding Stage (W)')
    ax11.set_xlabel('W = m/N (normalized decoding)')
    ax11.set_ylabel('Dm` = Dm/N')
    ax11.legend(title='Block Length', title_fontsize=12, fontsize=11, loc='upper right')
    plt.savefig('Figure9&Figure11.png', dpi=300)
    print("Finished! Image should be saved in same directory as Figure9&11.png")


generateFigures()