# -*- coding: utf-8 -*-
"""
State variables:

Substrates:

Sd       :   Sd conc. -- organic matter Substrate, dissolved 
O2       :   O2 conc.
NO3      :   NO3 conc.
NO2      :   NO2 conc.
NH4      :   NH4 conc.
N2       :   N-N2 conc. -- gaseous N2, but total N atoms are tracked
N2O      :   N-N2O conc. -- gaseous N2O, but total N atoms are tracked

Functional types:

#Aerobic heterotrophs:
baer     :   Obliate Aerobic Heterotroph, conc. of biomass
bFac     :   Facultatively Aerobic Het, conc. of biomass (not necessary for steady state analysis)

#Speciation of denitrification steps: (all obligate anaerobic heterotrophs):
b1Den    :   1Den, conc. of biomass (NO3-->NO2) 
b2Den    :   2Den conc. of biomass (NO2-->N2)
b3Den    :   3Den conc. of biomass (NO3-->N2)
b4Den    :   4Den conc. of biomass (NO2-->N2O)
b5Den    :   5Den conc. of biomass (N2O-->N2)
b6Den    :   6Den conc. of biomass (NO3-->N2O)

#Speciation of DNRA steps: (all obligate anaerobic heterotrophs):
bDNRA1    :  DNRA conc. of biomass (NO3-->NH4)
bDNRA2    :  DNRA conc. of biomass (NO2-->NH4)

#Chemoautotrophs:
bAOO     :   AOO conc.of biomass
bNOO     :   NOO conc.of biomass
bAOX     :   AOX conc. of biomass


-------

Purpose
-------
    Original purpose: A 0D chemostat model of redox reactions occuring in anoxic water columns
    
    Edits: use existing model (Zakem et al. 2020) to assess competition between denitrification and DNRA (both heterotrophic)
    Modular denitrification included, yields of denitrifiers depend on Gibbs free energy. 

@authors: Jemma Fadum, Xin Sun, Emily Zakem, Pearse Buchanan

"""

#####################################################################
#%% imports
import sys
import os
import numpy as np
import xarray as xr
import pandas as pd

# plotting packages
import seaborn as sb
sb.set(style='ticks')
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.gridspec import GridSpec


#####################################################################
#%% Set variable initial conditions and incoming concentrations to chemostat experiment

### Organic matter (Sd)
#Sdin_vary = np.array([10])  
#analysis range:
Sdin_vary = np.arange(0, 10.001, 0.01)  

### Nitrite supply
NO2in_vary = np.array([25])  #For Fig 4 
#NO2in_vary = np.arange(0, 30.001, 0.1) #For Fig 3

#here, list which two are varying for initilization of arrays below
var1 = Sdin_vary
var2 = NO2in_vary

#####################################################################
### model parameters for running experiments

dil = 0.04  # dilution rate (1/day)
days = 2000 #1e2  # number of days to run chemostat
dt = 0.001  # timesteps per day (days)
timesteps = days/dt     # number of timesteps
out_at_day = 10.0       # output results this often (days)
nn_output = days/out_at_day     # number of entries for output
avg_ntimesteps = 1 #200 #number of timesteps over which to average final results (for pulse). if no pulse, avg_ntimesteps = 1 for last time point only

if dil == 0: #setting dilution rate to zero will let us assure that N (mass) is conserved.
    print("dilution = 0, check N balance")

#####################################################################
#Initialize Arrays

#for variable/diagnostic X, each empty (nan) array is constructed as 
#fin_X = np.ones((len(var1), len(var2))) * np.nan

exec(open('initialize_arrays_JF.py').read())

#####################################################################
#%% set traits of the different biomasses
#from traits_Xin_GibbsEnergyAvgConc_StepPenalty_NOign import * #NOign means ignore NO reductase in long step penalty
exec(open('traits_DNRADenitAna_JF.py').read())

#####################################################################
#%% calculate R*-stars for all microbes (not needed for model results, only for plots)
exec(open('Rstars_calc_JF.py').read())

#####################################################################
#%% begin k and m loops (for varying supply rates)

from model_DNRADenitAna_JF import OMZredox

for k in np.arange(len(var1)):
    for m in np.arange(len(var2)):

        print("k = ",k,", m = ",m)
        print("Org matter = ",Sdin_vary[k],", NO2 = ",NO2in_vary[m])
        
        # 1) Set supply rates: chemostat inputs (µM-N or µM O2)
        in_Sd =  Sdin_vary[k]
        in_O2 = 0.0
        in_NO3 = 0
        in_NO2 = NO2in_vary[m]
        in_NH4 = Sdin_vary[k]*0.6
        #in_NH4 = 0.0
        in_N2 = 0.0
        in_N2O = 0.0
        
        # 2) Set initial conditions 
        #Initial substrate concentrations: (for now, these are the same as the supply rates) 
        IC_Sd = 10 
        IC_O2 = 0.0 
        IC_NO3 = 0 
        IC_NO2 = 10
        IC_NH4 = 10  
        IC_N2 = 0.0
        IC_N2O = 0.0
        
        #Initial biomasses (set to 0.0 to exclude, set > 0 to turn on)
        IC_baer = 0 #obligate aerobic het
        IC_bFac = 0 # turn off facultative
        IC_b1Den = 0.0 # NO3-->NO2  
        IC_b2Den = 0.0 # NO2-->N2
        IC_b3Den = 0.0 # NO3-->N2  
        IC_b4Den = 0.1 # NO2-->N2O
        IC_b5Den = 0.1 # N2O-->N2
        IC_b6Den = 0.0 # NO3-->N2O  
        IC_bAOO = 0
        IC_bNOO = 0
        IC_bAOX = 0.1
        IC_bDNRA1 = 0.0 # NO3-->NH4 
        IC_bDNRA2 = 0.1 # NO2-->NH4

        
        # Call and run main model function (function defined in model_DNRADenitAna_JF.py)
        results = OMZredox(timesteps, nn_output, dt, dil, out_at_day, \
                     pcoef_O2_aer, pcoef_O2_AOO, pcoef_O2_NOO, \
                     K_O2_aer, K_O2_AOO, K_O2_NOO, \
                     pcoef_N2O_Den, K_N2O_Den, \
                     VmaxS, K_S, \
                     VmaxDIN_Den, K_DIN_Den, \
                     VmaxDIN_DNRA, K_DIN_DNRA, \
                     VmaxNH4_AOO, K_NH4_AOO, VmaxNO2_NOO, K_NO2_NOO, \
                     VmaxNH4_AOX, K_NH4_AOX, VmaxNO2_AOX, K_NO2_AOX, \
                     y_OM_aer, y_O2_aer, y_OM_aerFac, y_O2_aerFac, y_OM_1DenFac, y_NO3_1DenFac, \
                     y_OM_1Den, y_NO3_1Den, y_OM_2Den, y_NO2_2Den, y_OM_3Den, y_NO3_3Den, \
                     y_OM_4Den, y_NO2_4Den, y_OM_5Den, y_N2O_5Den, y_OM_6Den, y_NO3_6Den, y_OM_DNRA1, y_OM_DNRA2, y_NO3_DNRA, y_NO2_DNRA, \
                     y_NH4_AOO, y_O2_AOO, y_NO2_NOO, y_O2_NOO, y_NH4_AOX, y_NO2_AOX, \
                     e_NO2_1Den, e_N2_2Den, e_N2_3Den, e_N2O_4Den, e_N2_5Den, e_N2O_6Den, \
                     e_NO3_AOX, e_N2_AOX, e_NH4_DNRA1, e_NH4_DNRA2, \
                     in_Sd, in_O2, in_NO3, in_NO2, in_NH4, in_N2, in_N2O, \
                     IC_Sd, IC_O2, IC_NO3, IC_NO2, IC_NH4, IC_N2, IC_N2O, \
                     IC_baer, IC_bFac, IC_b1Den, IC_b2Den, IC_b3Den, IC_bAOO, IC_bNOO, IC_bAOX, IC_b4Den, IC_b5Den, IC_b6Den, IC_bDNRA1, IC_bDNRA2)
     
            
           
        out_Sd = results[0]
        out_O2 = results[1]
        out_NO3 = results[2]
        out_NO2 = results[3]
        out_NH4 = results[4]
        out_N2O = results[5] 
        out_N2 = results[6]
        out_baer = results[7]
        out_bFac = results[8]
        out_b1Den = results[9]
        out_b2Den = results[10]
        out_b3Den = results[11]
        out_b4Den = results[12] 
        out_b5Den = results[13] 
        out_b6Den = results[14] 
        out_bAOO = results[15]
        out_bNOO = results[16]
        out_bAOX = results[17]
        out_bDNRA1 = results[18]  
        out_bDNRA2 = results[19] 
        out_facaer = results[20]
        out_time = results[21]
    
        
###########################################################################
    
        # 5) Record solutions in initialised arrays (here, as averages, for pulse experiments, but all should be exactly the same
        fin_O2[k,m] = np.nanmean(out_O2[-avg_ntimesteps::])
        fin_Sd[k,m] = np.nanmean(out_Sd[-avg_ntimesteps::])
        fin_NO3[k,m] = np.nanmean(out_NO3[-avg_ntimesteps::])
        fin_NO2[k,m] = np.nanmean(out_NO2[-avg_ntimesteps::])
        fin_NH4[k,m] = np.nanmean(out_NH4[-avg_ntimesteps::])
        fin_N2[k,m] = np.nanmean(out_N2[-avg_ntimesteps::])
        fin_N2O[k,m] = np.nanmean(out_N2O[-avg_ntimesteps::]) # add N2O
        fin_baer[k,m] = np.nanmean(out_baer[-avg_ntimesteps::])
        fin_bFac[k,m] = np.nanmean(out_bFac[-avg_ntimesteps::])
        fin_b1Den[k,m] = np.nanmean(out_b1Den[-avg_ntimesteps::])
        fin_b2Den[k,m] = np.nanmean(out_b2Den[-avg_ntimesteps::])
        fin_b3Den[k,m] = np.nanmean(out_b3Den[-avg_ntimesteps::])
        fin_b4Den[k,m] = np.nanmean(out_b4Den[-avg_ntimesteps::]) # add Den4
        fin_b5Den[k,m] = np.nanmean(out_b5Den[-avg_ntimesteps::]) # add Den5
        fin_b6Den[k,m] = np.nanmean(out_b6Den[-avg_ntimesteps::]) # add Den6
        fin_bAOO[k,m] = np.nanmean(out_bAOO[-avg_ntimesteps::])
        fin_bNOO[k,m] = np.nanmean(out_bNOO[-avg_ntimesteps::])
        fin_bAOX[k,m] = np.nanmean(out_bAOX[-avg_ntimesteps::])
        fin_bDNRA1[k,m] = np.nanmean(out_bDNRA1[-avg_ntimesteps::]) 
        fin_bDNRA2[k,m] = np.nanmean(out_bDNRA2[-avg_ntimesteps::]) 
        
#Here ends the k, m loop


###########################################################################
 #Check conservation of mass if dilution rate is set to zero
if dil == 0.0:
    end_N = fin_Sd + fin_NO3 + fin_NO2 + fin_NH4 + fin_N2 + fin_N2O + \
            fin_baer + fin_bFac + fin_b1Den + fin_b2Den + fin_b3Den + fin_b4Den \
            + fin_b5Den + fin_b6Den + fin_bAOO + fin_bNOO + fin_bAOX + fin_bDNRA1 \
            + fin_bDNRA2 
    initial_N = IC_Sd + IC_NO3 + IC_NO2 + IC_NH4 + IC_N2 + IC_N2O  \
            + IC_baer + IC_bFac + IC_b1Den + IC_b2Den + IC_b3Den + IC_b4Den \
            + IC_b5Den + IC_b6Den + IC_bAOO + IC_bNOO + IC_bAOX +IC_bDNRA1  \
            + IC_bDNRA2 
    for k in np.arange(len(Sdin_vary)):
        for m in np.arange(len(O2in_vary)):
           print(" Checking conservation of N mass ")
           print(" Initial Nitrogen =", initial_N)
           print(" Final Nitrogen =", end_N[k,m])
 

              
 
###########################################################################

#EXPORT

#Biomass
np.savetxt("DNRA_biomass_Fig4.csv", fin_bDNRA2, delimiter=",")
np.savetxt("IncompDenit_biomass_Fig4.csv", fin_b4Den, delimiter=",")
np.savetxt("CompDenit_biomass_Fig4.csv", fin_b5Den, delimiter=",")
np.savetxt("Anammox_biomass_Fig4.csv", fin_bAOX, delimiter=",")

np.savetxt("fin_OM_Fig4.csv", fin_Sd, delimiter=",")
np.savetxt("fin_NH4_Fig4.csv", fin_NH4, delimiter=",")
np.savetxt("fin_NO2_Fig4.csv", fin_NO2, delimiter=",")


