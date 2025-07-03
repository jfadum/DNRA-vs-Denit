# -*- coding: utf-8 -*-
"""

Purpose
-------
    DNRA vs. Denitrification vs Anammox project 
    Modified traits based on Zakem et al. 2019 ISME
    
NOTE: Oxygen uptake code retained for option of turning on aerobes but not used in this analysis

@authors: Jemma Fadum, Xin Sun, Pearse Buchanan and Emily Zakem
"""

import numpy as np
from diffusive_gas_coefficient import po_coef
from diffusive_gas_coefficient import pn2o_coef  


############################################################
# diffusive oxygen and n2o requirements based on cell diameters and carbon contents

# cell volumes (um^3)
vol_aer = 0.05  # based on SAR11 (Giovannoni 2017 Ann. Rev. Marine Science)
vol_den = vol_aer
vol_aoo = np.pi * (0.2*0.5)**2 * 0.8    # [rods] N. maritimus SCM1 in Table S1 from Hatzenpichler 2012 App. Envir. Microbiology
vol_noo = np.pi * (0.3*0.5)**2 * 3      # [rods] based on average cell diameter and length of Nitrospina gracilis (Spieck et al. 2014 Sys. Appl. Microbiology)
vol_aox = vol_aer 

# cell sizes (um), where vol = 4/3 * pi * r^3
diam_aer = ( 3 * vol_aer / (4 * np.pi) )**(1./3)*2
diam_den = ( 3 * vol_den / (4 * np.pi) )**(1./3)*2
diam_aoo = ( 3 * vol_aoo / (4 * np.pi) )**(1./3)*2
diam_noo = ( 3 * vol_noo / (4 * np.pi) )**(1./3)*2
diam_aox = ( 3 * vol_aox / (4 * np.pi) )**(1./3)*2

# cellular C:N of microbes
CN_aer = 5.0   # Zimmerman et al. 2014
CN_den = 5.0
CN_aoo = 5.0    
CN_noo = 5.0    
CN_aox = 5.0    

# g carbon per cell (assumes 0.1 g DW/WW for all microbial types as per communication with Marc Strous)
Ccell_aer = 0.1 * (12*CN_aer / (12*CN_aer + 7 + 16*2 + 14)) / (1e12 / vol_aer)
Ccell_den = 0.1 * (12*CN_den / (12*CN_den + 7 + 16*2 + 14)) / (1e12 / vol_den)
Ccell_aoo = 0.1 * (12*CN_aoo / (12*CN_aoo + 7 + 16*2 + 14)) / (1e12 / vol_aoo)
Ccell_noo = 0.1 * (12*CN_noo / (12*CN_noo + 7 + 16*2 + 14)) / (1e12 / vol_noo)
Ccell_aox =  0.1 * (12*CN_aox / (12*CN_aox + 7 + 16*2 + 14)) / (1e12 / vol_aox) 

# cell quotas (mol C / um^3)
Qc_aer = Ccell_aer / vol_aer / 12.0 * (6.5e-15 / Ccell_aer) # normalise to 6.5 fg C measured by White et al 2019
Qc_den = Ccell_den / vol_den / 12.0 * (6.5e-15 / Ccell_aer) # normalise to 6.5 fg C measured by White et al 2019
Qc_aoo = Ccell_aoo / vol_aoo / 12.0 * (6.5e-15 / Ccell_aer) # normalise to 6.5 fg C measured by White et al 2019
Qc_noo = Ccell_noo / vol_noo / 12.0 * (6.5e-15 / Ccell_aer) # normalise to 6.5 fg C measured by White et al 2019
Qc_aox = Ccell_aox / vol_aox / 12.0 * (6.5e-15 / Ccell_aer) # normalise to 6.5 fg C measured by White et al 2019

# diffusive oxygen and n2o coefficient
dc = 1.5776 * 1e-5      # cm^2/s for 12C, 35psu, 50bar, Unisense Seawater and Gases table (.pdf)
dc = dc * 1e-4 * 86400  # cm^2/s --> m^2/day
dc_n2o = dc * 1.0049    # m^2/day (this is necessary to calculate N2O star in call_moel_XinanoxicODZ’s ‘#%% calculate R*-stars for all microbes’)

#coefficients for diffusive uptake of O2 and N2O
pcoef_O2_aer = po_coef(diam_aer, Qc_aer, CN_aer)
pcoef_O2_AOO = po_coef(diam_aoo, Qc_aoo, CN_aoo)
pcoef_O2_NOO = po_coef(diam_noo, Qc_noo, CN_noo)
pcoef_N2O_Den = pn2o_coef(diam_den, Qc_den, CN_den)

#############################################################

### Yields (y) and excretion products (e) 

### Delta G depended organic matter yield for heterotrophs (aerobes and denitrifiers)
##### THERMODYNAMIC DEFINITIONS
##### authors: Emily Zakem and Xin Sun, DNRA additions by Jemma Fadum 

## define parameters
T = 25 + 273.15 # temperature (degree C)
pH = 7.5 
R = 8.3145 #J/(mol*K) Ideal gas constant
H = 10**(-pH) #Converting pH to concentration in units of mol/L

## Free energy of formation for relevant compounds and phases all values
## from (Amend & Shock 2001) at 25 degrees C
## EJZ: at pH = 0, not 7 (diff than R+McC chart)

DGf_NO3 = -110.91 * 1e3 #J/mol (aq) [NO3-]
DGf_NO2 = -32.22  * 1e3 #J/mol (aq) [NO2-]
DGf_H2O = -237.18 * 1e3 #J/mol (l)
DGf_N2O =  113.38 * 1e3 #J/mol (aq)
DGf_H   =    0    * 1e3 #J/mol (aq) [H+]
DGf_NO  =  102.06 * 1e3 #J/mol (aq)
DGf_N2  =   18.18 * 1e3 #J/mol (aq)
DGf_O2  =   16.54 * 1e3 #J/mol (aq) 
DGf_NH4 = -79.45  * 1e3 #J/mol          

## Calculating the standard free energy for the half reactions
# unit of DGo below is J/mol

DGoo = 0.5*DGf_H2O - 1/4*DGf_O2 - DGf_H #water-oxygen: 1/4O2 + H+ + e- --> 1/2H2O
DG1o = 0.5*DGf_NO2 + 0.5*DGf_H2O - 0.5*DGf_NO3 - DGf_H 
DG4o = 0.5*DGf_N2 + 0.5*DGf_H2O - 0.5*DGf_N2O - DGf_H 
DG123o = 1/8*DGf_N2O + 5/8*DGf_H2O - 1/4*DGf_NO3 - 5/4*DGf_H   
DG1234o = 1/10*DGf_N2 + 3/5*DGf_H2O - 1/5*DGf_NO3 - 6/5*DGf_H 
DG23o = 1/4*DGf_N2O + 3/4*DGf_H2O - 1/2*DGf_NO2 - 3/2*DGf_H 
DG234o = 1/6*DGf_N2 + 2/3*DGf_H2O - 1/3*DGf_NO2 - 4/3*DGf_H 
DGdnrao1 = 1/8*DGf_NH4 + 3/8*DGf_H2O - 1/8*DGf_NO3 - 5/4*DGf_H   #pg 135 in EnvBioTech
DGdnrao2 = 1/6*DGf_NH4 + 1/3*DGf_H2O - 1/6*DGf_NO2 - 4/3*DGf_H   

#avg concentrations (just to initiate -- not to determine solution):
NO3avg = 30 * 1e-6 #mol/L 
NO2avg = 30 * 1e-6 #mol/L 
N2Oavg = 1e-8 #mol/L 
N2avg = 1e-4 #mol/L 
O2avg = 1e-6 # Xin: add this for aerobic heterotrophs
NH4avg = 1e-6 #mol/L  


#per e- (In Julia, default based of log is e)
DGo = DGoo + R*T*np.log(1/O2avg**(1/4)/H) # Xin: add a set for aerobic heterotrophs
DG1 = DG1o + R*T*np.log(NO2avg**0.5/NO3avg**0.5/H) 
DG4 = DG4o + R*T*np.log(N2avg**0.5/N2Oavg**0.5/H)
DG123  = DG123o  + R*T*np.log(N2Oavg**(1/8)/NO3avg**(1/4)/H**(5/4)) 
DG1234 = DG1234o + R*T*np.log(N2avg**(1/10)/NO3avg**(1/5)/H**(6/5))  
DG23  = DG23o  + R*T*np.log(N2Oavg**(1/4)/NO2avg**(1/2)/H**(3/2)) 
DG234 = DG234o + R*T*np.log(N2avg**(1/6)/NO2avg**(1/3)/H**(4/3)) 
DGdnra1 = DGdnrao1 + R*T*np.log(NH4avg**(1/8)/NO3avg**(1/8)/H**(5/4))  
DGdnra2 = DGdnrao2 + R*T*np.log(NH4avg**(1/6)/NO2avg**(1/6)/H**(4/3))  

## Gibbs rxn energy for oxidation of organic matter
## (here, estimated for avg marine OM):
    
om_energy = 3.33*1e3 #J/g cells from R&McC (estimate)
Cd=6.6; Hd=10.9; Od=2.6; Nd=1 # Based on (Anderson 1995)
## denominator for decomposition to nh4:
dD=4*Cd+Hd-2*Od-3*Nd;
DGom = om_energy*(Cd*12+Hd+Od*16+Nd*14)/dD

# ## energy to form biomass b from pyruvate -- R&McC:
# b_energy = 3.33*1e3 #J/g cells from R&McC (estimate)
Cb=5; Hb=7; Ob=2; Nb=1 # Based on (Zimmerman et al. 2014)
dB=4*Cb+Hb-2*Ob-3*Nb
# DGpb = b_energy*(Cb*12+Hb+Ob*16+Nb*14)/dB #energy to synthesize cells from pyruvate
# 
# ## efficiency of e transfer:
ep = 0.6

############################################
#Estimate of cost of cell synthesis by backtracking from observed average organic matter yield for aerobic hets:
DGs_with_ep = 300*1e3 # reset DGs_with_ep to get the yield of aerobic hetero close to obs (0.2~0.3)

#get empirical f from empirical yom assuming aerobic heterotrophy:
y_emp = 0.25 #ranges from 0.2 to 0.3, use 0.25 for analysis 
f_emp = y_emp*dB/dD #f = y*dB/dD
#now backtrack what DGs_with_ep should be from f equation:
    
## f = 1/(1-DGs_with_ep/ep/DGr) where DGr = energy of energy-generating rxn, = DGe - DGom

## Energy balance: f*DGs_with_ep = -(1-f)*DGr*ep

#Aerobic heterotrophy: DGe = DGo, DGom = DGom
DGr = DGo - DGom
DGs_with_ep = DGr*ep*(1. - 1./f_emp) #energy balance!

############################################
#Now calculate Yields from Gibbs free energies for heterotrophs:

def calc_f(DGe, DGom, ep, DGs_with_ep):
    
    #Free energies of reactions:
    # DGr = DGe - DGom #catabolic rxn: energy released from redox rxn
    # DGs_with_ep #Energy required for cell synthesis (including the extra energy required due to inefficiency ep)
    #Energy balance:
    # f * DGs_with_ep = (1 - f) * DGr * ep #energy towards synthesis equals energy generated from catabolic rxn
    # f = 1/(1-DGs_with_ep/ep/DGr)
    f = 1/(1-DGs_with_ep/ep/(DGe - DGom))

    return f

###################################################
# aerobic heterotrophy ("aer") 
f = calc_f(DGo, DGom, ep, DGs_with_ep) 
y_OM_aer =  f * dD/dB 
y_O2_aer = (f/dB) / ((1.0-f)/4.0) 


# nitrate reduction to nitrite (NO3 --> NO2)
f = calc_f(DG1, DGom, ep, DGs_with_ep)       # fraction of electrons used for biomass synthesis
y_OM_1Den = f * dD/dB
y_NO3_1Den = (f/dB) / ((1.0-f)/2.0) # yield of biomass per unit nitrate consumed (full equation for NO3->NO2 functional type)
e_NO2_1Den = 1.0 / y_NO3_1Den      # mols NO2 produced per mol biomass synthesised

# nitrite reduction to N2 (NO2 --> N2)
f = calc_f(DG234, DGom, ep, DGs_with_ep)     # fraction of electrons used for biomass synthesis
y_OM_2Den = f * dD/dB
#y_OM_2Den = 0.2900
y_NO2_2Den = (f/dB) / ((1.0-f)/3.0)       # yield of biomass per unit nitrite consumed
#y_NO2_2Den= 0.02490
e_N2_2Den = 1.0 / y_NO2_2Den         # moles N-N2 produced per mole biomass synthesised

# Full denitrification (NO3 --> N2)
f = calc_f(DG1234, DGom, ep, DGs_with_ep)      # fraction of electrons used for biomass synthesis
y_OM_3Den = f * dD/dB
y_NO3_3Den = (f/dB) / ((1.0-f)/5.0) # yield of biomass per unit nitrate consumed (Zakem et al., 2019 A14)
e_N2_3Den = 1.0 / y_NO3_3Den         # moles N-N2 produced per mole biomass synthesised

# nitrite reduction to N2O (NO2 --> N2O)
f = calc_f(DG23, DGom, ep, DGs_with_ep)   # fraction of electrons used for biomass synthesis
y_OM_4Den = f * dD/dB
y_NO2_4Den = (f/dB) / ((1.0-f)/2.0) # yield of biomass per unit nitrite consumed
e_N2O_4Den = 1.0 / y_NO2_4Den         # moles N-N2O produced per mole biomass synthesised

# N2O reduction to N2 (N2O --> N2)
f = calc_f(DG4, DGom, ep, DGs_with_ep)    # fraction of electrons used for biomass synthesis
y_OM_5Den = f * dD/dB
y_N2O_5Den = (f/dB) / ((1.0-f)/1.0) # yield of biomass per unit N-N2O consumed
e_N2_5Den = 1.0 / y_N2O_5Den         # moles N-N2 produced per mole biomass synthesised

# nitrate reduction to N2O (NO3 --> N2O)
f = calc_f(DG123, DGom, ep, DGs_with_ep)   # fraction of electrons used for biomass synthesis
y_OM_6Den = f * dD/dB
y_NO3_6Den = (f/dB) / ((1.0-f)/4.0) # yield of biomass per unit nitrate consumed
e_N2O_6Den = 1.0 / y_NO3_6Den         # moles N-N2O produced per mole biomass synthesised

# # Facultative heterotrophy (oxygen and nitrate reduction to nitrite)
fac_penalty = 1# no penalty unless we are interested in facultative-obligate competition. 0.8
y_OM_aerFac = y_OM_aer * fac_penalty
f = y_OM_aerFac * dB/dD         # The fraction of electrons used for biomass synthesis (Eq A9 in Zakem et al. 2019 ISME)
y_O2_aerFac = (f/dB) / ((1.0-f)/4.0)  # yield of biomass per unit oxygen reduced

y_OM_1DenFac = y_OM_1Den * fac_penalty #1Den = NO3 to NO2
f = y_OM_1DenFac * dB/dD
y_NO3_1DenFac = (f/dB) / ((1.0-f)/2.0)

# Chemoautotrophic ammonia oxidation (NH3 --> NO2)
y_NH4_AOO = 0.0245         # mol N biomass per mol NH4 (Bayer et al. 2022; Zakem et al. 2022)
f_AOO = y_NH4_AOO / (6*(1/dB - y_NH4_AOO/dB))         # fraction of electrons 
y_O2_AOO = f_AOO/dB / ((1-f_AOO)/4.0)      # mol N biomass per mol O2 !!not O-O2 

# Chemoautotrophic nitrite oxidation (NO2 --> NO3)
y_NO2_NOO = 0.0126         # mol N biomass per mol NO2 (Bayer et al. 2022)
f_NOO = (y_NO2_NOO * dB) /2          # fraction of electrons #for future meeting
y_O2_NOO = 4*f_NOO*(1-f_NOO)/dB

# Chemoautotrophic anammox (NH4 + NO2 --> NO3 + N2)
y_NH4_AOX = 1./75                  # mol N biomass per mol NH4 (Lotti et al. 2014 Water Research) ***Rounded to nearest whole number
y_NO2_AOX = 1./89                  # mol N biomass per mol NO2 (Lotti et al. 2014 Water Research) ***Rounded to nearest whole number 
e_N2_AOX = 150                     # mol N-N2 formed per mol biomass N synthesised ***Rounded to nearest whole number
e_NO3_AOX = 13                     # mol NO3 formed per mol biomass N synthesised ***Rounded to nearest whole number

# nitrate reduction to ammonium (DNRA, NO3 --> NH4)
f = calc_f(DGdnra1, DGom, ep, DGs_with_ep)      # fraction of electrons used for biomass synthesis
y_OM_DNRA1 = f * dD/dB
y_NO3_DNRA = (f/dB) / ((1.0-f)/8.0) # yield of biomass per unit nitrate consumed
e_NH4_DNRA1 = (dB - dD*f) / (dD*f) + dB*(1-f) /(8*f)      # mols NH4 produced per mol biomass synthesised



# nitrite reduction to ammonium (DNRA, NO2 --> NH4)

f = calc_f(DGdnra2, DGom, ep, DGs_with_ep)      # fraction of electrons used for biomass synthesis
y_OM_DNRA2 = f * dD/dB
y_NO2_DNRA = (f/dB) / ((1.0-f)/6.0) # yield of biomass per unit nitrate consumed
ytest = (8*f)/ (dB*(1-f))    
e_NH4_DNRA2 = (dB - dD*f) / (dD*f) + dB*(1-f) /(6*f)  # mols NH4 produced per mol biomass synthesised



############################################################
#4. Kinetic parameters (other than diffusion-limited uptake parameters calculated above):

#Estimate overall Vmax of OM from an estimate of bulk maximum growth rate and the organic matter yield
mumax_aer = 1 #1/day 
VmaxS = 3 # max mol Org N consum. rate per (mol BioN) per day
VmaxDIN_Den = 30
VmaxNH4_AOO = 50.8 
VmaxNO2_NOO = 50.8 
VmaxNH4_AOX = 30 #set to match denit
VmaxNO2_AOX = 30 #set to match denit
VmaxDIN_DNRA = 30 #set to match denit

#Half-saturation constants (since the input conc will be in µM = mmol/m3, k will also be in µM)
K_S = 1        # organic nitrogen (uncertain) uM N from jia et al 2020
K_DIN_Den = .1     # 4 – 25 µM NO2 for denitrifiers (Almeida et al. 1995) #used in jia et al. 2020
K_NH4_AOO = 0.1       # Martens-Habbena et al. 2009 Nature
K_NO2_NOO = 0.1       # Reported from OMZ (Sun et al. 2017) and oligotrophic conditions (Zhang et al. 2020)
K_NH4_AOX = 0.1    # Awata et al. 2013 for Scalindua
K_NO2_AOX = 0.1    # Awata et al. 2013 for Scalindua actually finds a K_no2 of 3.0 uM, but this excludes anammox completely in our experiments
K_DIN_DNRA = .1   # Set equal to toher functional metabolism groups 

# set K = 0 for N2O or O2 if want to model their uptake rate as a linear function (only constrained by diffusion)
K_N2O_Den = 0.3*2 #1e-4 # 0.3*2   # *2 convert µM N2O into µM N-N2O based on (Sun et al., 2021) in ODZ k = 0.3 µM N2O, in oxic layer = 1.4~2.8 µM

K_O2_aer = 0.2   # (µM-O2) 10 to 200 nM at extremely low oxygen concentrations (Tiano et al., 2014); 200 nM (low affinity terminal oxidases) (Morris et al., 2013)
K_O2_AOO = 0.333  # (µM-O2) 333 nM at oxic-anoxic interface (Bristow et al., 2016)
K_O2_NOO = 0.8   # (µM-O2) 778 nM at oxic-anoxic interface or 0.5 nM and 1750 nM breaking into two M-M curves (Bristow et al., 2016)


############################################################
#PRINT
print("Calculated yields:")  
print("....")
print("y_OM_Den NO2 -> N2O = ", y_OM_4Den)
print("y_OM_DNRA NO2 -> NH4 =", y_OM_DNRA2)
print("....")
print("y_NO2_Den NO2 -> N2O = ", y_NO2_4Den)
print("y_NO2_DNRA NO2 -> NH4 =" , y_NO2_DNRA) 

