# -*- coding: utf-8 -*-

"""
Original: Revised from model_Xin_GibbsEnergyAvgConc.py in April 2023 by EJZ


Purpose
-------
See accomanying R markdown for full project description

@author: Jemma, Xin Sun, Pearse Buchanan, and Emily Zakem

"""

import numpy as np

from numba import jit

#jit makes things run faster

@jit(nopython=True)
def OMZredox(timesteps, nn_output, dt, dil, out_at_day, \
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
             IC_baer, IC_bFac, IC_b1Den, IC_b2Den, IC_b3Den, IC_bAOO, IC_bNOO, IC_bAOX, IC_b4Den, IC_b5Den, IC_b6Den, IC_bDNRA1, IC_bDNRA2):

    '''
    ______________________________________________________
    0D Chemostat model for aerobic and anaerobic processes
    ------------------------------------------------------
    
    INPUTS
    ------
    
        timesteps   :   number of timesteps in run
        nn_output   :   number of times we save output
        dt          :   number of timesteps per day
        dil         :   dilution rate of chemostat
        out_at_day  :   day interval to record output
  
        pulse_Sd    :   instant injection of OM into the model (mmol)
        pulse_baer  :   instant injection of baer into the model (mmol)
        pulse_bFac  :   instant injection of bFac into the model (mmol)
        pulse_O2    :   instant injection of O2 into the model (mmol)
              
        pcoef_O2_aer      :   diffusion-limited O2 uptake by aerobic heterotrophs (m3 / mmol N-biomass / day)
        pcoef_O2_AOO      :   diffusion-limited O2 uptake by aerobic ammonia oxidisers (m3 / mmol N-biomass / day)
        pcoef_O2_NOO      :   diffusion-limited O2 uptake by aerobic nitrite oxidisers (m3 / mmol N-biomass / day)
        pcoef_N2O_Den     :   diffusion-limited N2O uptake by Den5 (N2O-->N2) (m3 / mmol N-biomass / day)
       
        K_O2_aer
        K_O2_AOO
        K_O2_NOO  

        K_N2O_Den   :   half-saturation constant for N2O uptake
        mumax_Het, fac_penalty, Den_penalty, mumax_AOO, mumax_NOO, mumax_AOX, GrowthRateMM: these were used for double Michaelis-Menten kinetics
        
        VmaxS       :   maximum uptake rate of organic matter by all heterotrophs (mol orgN / mol N-cell / day )
        K_S         :   half saturation coefficient for organic matter uptake
        
        VmaxN_Den   :   maximum uptake rate of NO3 or NO2 by denitrifying heterotrophs 
        K_N_Den     :   half saturation coefficient for DIN uptake by heterotrophic bacteria
        
        VmaxN_AOO   :   maximum uptake rate of NH4 by ammonia oxidisers (mol NH4 / mol N-cell / day )
        K_N_AOO     :   half saturation coefficient for NH4 uptake by ammonia oxidisers
        
        VmaxN_NOO   :   maximum uptake rate of NO2 by nitrite oxidisers (mol NO2 / mol N-cell / day )
        K_N_NOO     :   half saturation coefficient for NO2 uptake by nitrite oxidisers
        
        VmaxNH4_AOX :   maximum uptake rate of NH4 by anammox bacteria (mol NH4 / mol N-cell / day )
        VmaxNO2_AOX :   maximum uptake rate of NO2 by anammox bacteria (mol NO2 / mol N-cell / day )
        K_NH4_AOX   :   half saturation coefficient for NH4 uptake by anaerobic ammonium oxidisers
        K_NO2_AOX   :   half saturation coefficient for NO2 uptake by anaerobic ammonium oxidisers
        
        VmaxNO3_DNRA:   maximum uptake rate of NO3 by DNRA bacteria (mol NO3 / mol N-cell / day)
        K_DIN_DNRA  :   half saturation coefficient for NO3 uptake by DNRA bacteria/ nitrate reducers ##JF added
        
        y_OM_aer    :   mol N-biomass / mol N-organics consumed for aerobic heterotrophy
        y_O2_aer    :   mol N-biomass / mol O2 reduced for aerobic heterotrophy
        y_OM_aerFac :   mol N-biomass / mol N-organics consumed for facultative heterotrophy
        y_O2_aerFac :   mol N-biomass / mol O2 reduced for facultative heterotrophy
        
        y_OM_1DenFac  :   mol N-biomass / mol N-organics consumed
        y_NO3_1DenFac :   mol N-biomass / mol NO3 reduced
        
        y_OM_1Den     :   mol N-biomass / mol N-organics consumed
        y_NO3_1Den    :   mol N-biomass / mol NO3 reduced (NO3 --> NO2)
        
        y_OM_2Den     :   mol N-biomass / mol N-organics consumed
        y_NO2_2Den    :   mol N-biomass / mol NO2 reduced (NO2 --> N2)
        
        y_OM_3Den     :   mol N-biomass / mol N-organics consumed
        y_NO3_3Den    :   mol N-biomass / mol NO3 reduced (NO3 --> N2)
        
        y_OM_4Den     :   mol N-biomass / mol N-organics consumed
        y_NO2_4Den    :   mol N-biomass / mol NO2 reduced (NO2 --> N2O)
        
        y_OM_5Den     :   mol N-biomass / mol N-organics consumed
        y_N2O_5Den    :   mol N-biomass / mol N-N2O reduced (N2O --> N2)
        
        y_OM_6Den     :   mol N-biomass / mol N-organics consumed
        y_NO3_6Den    :   mol N-biomass / mol NO3 reduced (NO3 --> N2O)
        
        y_NH4_AOO      :   mol N-biomass / mol NH4 oxidised
        y_O2_AOO       :   mol N-biomass / mol O2 reduced
        
        y_NO2_NOO      :   mol N-biomass / mol NO2 oxidised
        y_O2_NOO       :   mol N-biomass / mol O2 reduced
        
        y_NH4_AOX      :   mol N-biomass / mol NH4 oxidised
        y_NO2_AOX      :   mol N-biomass / mol NO2 reduced
        
        y_NO3_DNRA     :   mol N-biomass / mol NO3 reduced   
        y_OM_DNRA1     :   mol N-biomass / mol N organics consumed   
        
        y_NO2_DNRA     :   mol N-biomass / mol NO2 reduced   
        y_OM_DNRA2     :   mol N-biomass / mol N organics consumed   
        
        e_N2_2Den   :   production of N-N2 by nitrite reducing denitrifiers (mol N-N2 / mol N-biomass)
        e_N2_3Den   :   production of N-N2 by nitrate reducing denitrifiers (mol N-N2 / mol N-biomass)
        e_NO3_AOX   :   production of NO3 by anammox bacteria (mol NO3 / mol N-biomass)
        e_N2_AOX    :   production of N-N2 by anammox bacteria (mol N-N2 / mol N-biomass)
        e_NO2_1Den  :   mols NO2 produced per mol biomass synthesised
        e_N2O_n4Den :   moles N-N2O produced per mole biomass synthesised
        e_N2_5Den   :   moles N-N2 produced per mole biomass synthesised
        e_N2O_6Den  :   moles N-N2O produced per mole biomass synthesised
        e_NH4_DNRA1 :   moles of N-NH4 produced per mole of nitrate reducing biomass synthesised ##JF added
        e_NH4_DNRA2 :   moles of N-NH4 produced per mole of nitrite reducing biomass synthesised ##JF added
        
        in_Sd       :   incoming Sd concentration
        in_O2       :   incoming O2 concentration
        in_NO3      :   incoming NO3 concentration
        in_NO2      :   incoming NO2 concentration
        in_NH4      :   incoming NH4 concentration
        in_N2       :   incoming N2 concentration 
        in_N2O      :   incoming N2O concentration
        
        IC_Sd       :   initial Sd concentration
        IC_O2       :   initial O2 concentration
        IC_NO3      :   initial NO3 concentration
        IC_NO2      :   initial NO2 concentration
        IC_NH4      :   initial NH4 concentration
        IC_N2       :   initial N2 concentration 
        IC_N2O      :   initial N2O concentration
        
        IC_baer     :   initial Het concentration of biomass
        IC_bFac     :   initial Fac concentration of biomass
        IC_b1Den    :   initial 1Den concentration of biomass
        IC_b2Den    :   initial 2Den concentration of biomass
        IC_b3Den    :   initial 3Den concentration of biomass
        IC_b4Den    :   initial 4Den concentration of biomass
        IC_b5Den    :   initial 5Den concentration of biomass
        IC_b6Den    :   initial 6Den concentration of biomass
        IC_bAOO     :   initial AOO concentration of biomass
        IC_bNOO     :   initial NOO concentration of biomass
        IC_bAOX     :   initial AOX concentration of biomass
        IC_bDNRA1   :   initial DNRA1 concentration of biomass ## JF added 
        IC_bDNRA2   :   initial DNRA2 concentration of biomass ## JF added 

        
    OUTPUTS
    -------
    
        out_time     :   of days that have passed
        out_Sd       :   Sd conc.
        out_O2       :   O2 conc.
        out_NO3      :   NO3 conc.
        out_NO2      :   NO2 conc.
        out_NH4      :   NH4 conc.
        out_N2       :   N-N2 conc.
        out_N2O      :   N-N2O conc.
        
        
        out_baer     :   Het conc. of biomass
        out_bFac     :   Fac conc. of biomass
        out_b1Den    :   1Den conc. of biomass (NO3-->NO2)
        out_b2Den    :   2Den conc. of biomass (NO2-->N2)
        out_b3Den    :   3Den conc. of biomass (NO3-->N2)
        out_b4Den    :   4Den conc. of biomass (NO2-->N2O)
        out_b5Den    :   5Den conc. of biomass (N2O-->N2)
        out_b6Den    :   6Den conc. of biomass (NO3-->N2O)
        out_bAOO     :   AOO concentration of biomass
        out_bNOO     :   NOO concentration of biomass
        out_bAOX     :   AOX concentration of biomass
        out_bDNRA1   :   DNRA1 concentration of biomass   
        out_bDNRA2   :   DNRA2 concentration of biomass   
    '''
    
    # transfer initial conditions to model variables
    m_Sd = IC_Sd
    m_O2 = IC_O2 # make sure the unit is O2 for m_O2
    m_NO3 = IC_NO3
    m_NO2 = IC_NO2
    m_NH4 = IC_NH4
    m_N2 = IC_N2  # make sure the unit is N-N2 for m_N2
    m_N2O = IC_N2O  # make sure the unit is N-N2O (mmol-N/m3 = µM) for m_N2O 
    m_baer = IC_baer
    m_bFac = IC_bFac
    m_b1Den = IC_b1Den
    m_b2Den = IC_b2Den
    m_b3Den = IC_b3Den
    m_b4Den = IC_b4Den 
    m_b5Den = IC_b5Den 
    m_b6Den = IC_b6Den 
    m_bDNRA1 = IC_bDNRA1 
    m_bDNRA2 = IC_bDNRA2 
    m_bAOO = IC_bAOO
    m_bNOO = IC_bNOO
    m_bAOX = IC_bAOX
    
    # set the output arrays 
    out_Sd = np.ones((int(nn_output)+1)) * np.nan
    out_O2 = np.ones((int(nn_output)+1)) * np.nan
    out_NO3 = np.ones((int(nn_output)+1)) * np.nan
    out_NO2 = np.ones((int(nn_output)+1)) * np.nan
    out_NH4 = np.ones((int(nn_output)+1)) * np.nan
    out_N2 = np.ones((int(nn_output)+1)) * np.nan
    out_N2O = np.ones((int(nn_output)+1)) * np.nan
    out_baer = np.ones((int(nn_output)+1)) * np.nan
    out_bFac = np.ones((int(nn_output)+1)) * np.nan
    out_b1Den = np.ones((int(nn_output)+1)) * np.nan
    out_b2Den = np.ones((int(nn_output)+1)) * np.nan
    out_b3Den = np.ones((int(nn_output)+1)) * np.nan
    out_b4Den = np.ones((int(nn_output)+1)) * np.nan
    out_b5Den = np.ones((int(nn_output)+1)) * np.nan
    out_b6Den = np.ones((int(nn_output)+1)) * np.nan
    out_bAOO = np.ones((int(nn_output)+1)) * np.nan
    out_bNOO = np.ones((int(nn_output)+1)) * np.nan
    out_bAOX = np.ones((int(nn_output)+1)) * np.nan
    out_bDNRA1 = np.ones((int(nn_output)+1)) * np.nan
    out_bDNRA2 = np.ones((int(nn_output)+1)) * np.nan
    out_facaer = np.ones((int(nn_output)+1)) * np.nan
    out_time = np.ones((int(nn_output)+1)) * np.nan
    
    # set the array for recording average activity of facultative anaerobes
    interval = int((1/dt * out_at_day))
    facaer = np.ones((interval)) * np.nan
    
    # record the initial conditions
    i = 0
    out_time[i] = 0. 
    out_Sd[i] = m_Sd 
    out_O2[i] = m_O2
    out_NO3[i] = m_NO3
    out_NO2[i] = m_NO2 
    out_NH4[i] = m_NH4 
    out_N2[i] = m_N2
    out_N2O[i] = m_N2O # newly added
    out_baer[i] = m_baer
    out_bFac[i] = m_bFac
    out_b1Den[i] = m_b1Den
    out_b2Den[i] = m_b2Den
    out_b3Den[i] = m_b3Den 
    out_b4Den[i] = m_b4Den # newly added
    out_b5Den[i] = m_b5Den # newly added
    out_b6Den[i] = m_b6Den # newly added
    out_bDNRA1[i] = m_bDNRA1
    out_bDNRA2[i] = m_bDNRA2
    out_bAOO[i] = m_bAOO
    out_bNOO[i] = m_bNOO
    out_bAOX[i] = m_bAOX
    
    # begin the loop
    for t in np.arange(1,timesteps+1,1):

        #Uptake rates of gases (using diffusive limit, a linear function of gas concentration):
        p_O2_aer = (pcoef_O2_aer * m_O2)      # O2 uptake rate for aerobic heterotrophs. mol O2/day/mol Biomass N 
        p_O2_AOO = (pcoef_O2_AOO * m_O2)      # O2 uptake rate for AOA
        p_O2_NOO = (pcoef_O2_NOO * m_O2)      # O2 uptake rate for NOB
        p_N2O_Den = (pcoef_N2O_Den * m_N2O)       # N2O uptake rate. mol N-N2O / day / mol-BiomassN 
     
        #Uptake rates of other substrates, using Michaelis-Menten kinetics. Units: mol substrate/ day / mol Biomass N
        p_Sd = VmaxS * m_Sd / (K_S + m_Sd)   
        p_DenNO3 = VmaxDIN_Den * m_NO3 / (K_DIN_Den + m_NO3) 
        p_DNRANO3 = VmaxDIN_DNRA * m_NO3 / (K_DIN_DNRA + m_NO3) 
        p_DNRANO2 = VmaxDIN_DNRA * m_NO2 / (K_DIN_DNRA + m_NO2)           
        p_DenNO2 = VmaxDIN_Den * m_NO2 / (K_DIN_Den + m_NO2)   
        p_NH4_AOO = VmaxNH4_AOO * m_NH4 / (K_NH4_AOO + m_NH4)  
        p_NO2_NOO = VmaxNO2_NOO * m_NO2 / (K_NO2_NOO + m_NO2)  
        p_NH4_AOX = VmaxNH4_AOX * m_NH4 / (K_NH4_AOX + m_NH4)  
        p_NO2_AOX = VmaxNO2_AOX * m_NO2 / (K_NO2_AOX + m_NO2)  



        
        #Growth rates (units: day^(-1)) determined by the min rate of the substrate (e.g., OM, DIN, O2) uptake
        u_aer = np.fmax(0.0, np.fmin(p_Sd * y_OM_aer, p_O2_aer * y_O2_aer))          
        u_FacO2 = np.fmax(0.0, np.fmin(p_Sd * y_OM_aerFac, p_O2_aer * y_O2_aerFac))  
        u_FacNO3 = np.fmax(0.0, np.fmin(p_Sd * y_OM_1DenFac, p_DenNO3 * y_NO3_1DenFac)) 
        u_1Den = np.fmax(0.0, np.fmin(p_Sd * y_OM_1Den, p_DenNO3 * y_NO3_1Den))         
        u_2Den = np.fmax(0.0, np.fmin(p_Sd * y_OM_2Den, p_DenNO2 * y_NO2_2Den))        
        u_3Den = np.fmax(0.0, np.fmin(p_Sd * y_OM_3Den, p_DenNO3 * y_NO3_3Den))        
        u_4Den = np.fmax(0.0, np.fmin(p_Sd * y_OM_4Den, p_DenNO2 * y_NO2_4Den))         
        u_5Den = np.fmax(0.0, np.fmin(p_Sd * y_OM_5Den, p_N2O_Den * y_N2O_5Den))       
        u_6Den = np.fmax(0.0, np.fmin(p_Sd * y_OM_6Den, p_DenNO3 * y_NO3_6Den))       
        u_DNRA1 = np.fmax(0.0, np.fmin(p_Sd * y_OM_DNRA1, p_DNRANO3 * y_NO3_DNRA))   
        u_DNRA2 = np.fmax(0.0, np.fmin(p_Sd * y_OM_DNRA2, p_DNRANO2 * y_NO2_DNRA))   
        u_AOO = np.fmax(0.0, np.fmin(p_NH4_AOO * y_NH4_AOO, p_O2_AOO * y_O2_AOO))   
        u_NOO = np.fmax(0.0, np.fmin(p_NO2_NOO * y_NO2_NOO, p_O2_NOO * y_O2_NOO))   
        u_AOX = np.fmax(0.0, np.fmin(p_NO2_AOX * y_NO2_AOX, p_NH4_AOX * y_NH4_AOX)) 
        
        # facultative bacteria growth rates and excretions
        if u_FacO2 >= u_FacNO3:
            u_Fac = u_FacO2
            uptake_Sd_Fac = u_Fac * m_bFac / y_OM_aerFac
            uptake_O2_Fac = u_Fac * m_bFac / y_O2_aerFac
            uptake_NO3_Fac = 0.0
            prod_NO2_Fac = 0.0
            prod_NH4_Fac = u_Fac * m_bFac * (1./y_OM_aerFac - 1)
            facaer[int(t % interval)] = 1.0 #track whether fac pop is operating aerobically or not
        else:
            u_Fac = u_FacNO3
            uptake_Sd_Fac = u_Fac * m_bFac / y_OM_1DenFac # OM uptake rate (µM-N) of facultative nitrate reducer
            uptake_O2_Fac = 0.0
            uptake_NO3_Fac = u_Fac * m_bFac / y_NO3_1DenFac
            prod_NO2_Fac = uptake_NO3_Fac 
            prod_NH4_Fac = u_Fac * m_bFac * (1./y_OM_1DenFac - 1)
            facaer[int(t % interval)] = 0.0 #track whether fac pop is operating aerobically or not
        
        #track a few things here for clarity:

        #organic matter uptake by all heterotrophs
        uptake_Sd = u_aer * m_baer / y_OM_aer \
                  + uptake_Sd_Fac \
                  + u_1Den * m_b1Den / y_OM_1Den \
                  + u_2Den * m_b2Den / y_OM_2Den \
                  + u_3Den * m_b3Den / y_OM_3Den \
                  + u_4Den * m_b4Den / y_OM_4Den \
                  + u_5Den * m_b5Den / y_OM_5Den \
                  + u_6Den * m_b6Den / y_OM_6Den \
                  + u_DNRA1 * m_bDNRA1 / y_OM_DNRA1 \
                  + u_DNRA2 * m_bDNRA2 / y_OM_DNRA2
                  
        
        #Oxygen uptake by all aerobes
        uptake_O2 = u_aer * m_baer / y_O2_aer \
                  + uptake_O2_Fac \
                  + u_AOO * m_bAOO / y_O2_AOO \
                  + u_NOO * m_bNOO / y_O2_NOO

        #NH4 uptake
        uptake_NH4 = u_AOO * m_bAOO / y_NH4_AOO \
                   + u_AOX * m_bAOX / y_NH4_AOX \
                   + u_NOO * m_bNOO  #biomass synthesis for NOO (tiny)
        #NO3 uptake
        uptake_NO3 = u_1Den * m_b1Den / y_NO3_1Den \
                  + uptake_NO3_Fac \
                  + u_3Den * m_b3Den / y_NO3_3Den \
                  + u_6Den * m_b6Den / y_NO3_6Den  \
                  + u_DNRA1 * m_bDNRA1 / y_NO3_DNRA 

        #NO2 uptake
        uptake_NO2 = u_2Den * m_b2Den / y_NO2_2Den \
                   + u_4Den * m_b4Den / y_NO2_4Den \
                   + u_NOO * m_bNOO / y_NO2_NOO \
                   + u_AOX * m_bAOX / y_NO2_AOX \
                   + u_DNRA2 * m_bDNRA2 / y_NO2_DNRA  

        #N2O uptake
        uptake_N2O = u_5Den * m_b5Den / y_N2O_5Den

        #NH4 production
        prod_NH4 = u_aer * m_baer * (1./y_OM_aer - 1.) \
                 + prod_NH4_Fac \
                 + u_1Den * m_b1Den * (1./y_OM_1Den - 1.) \
                 + u_2Den * m_b2Den * (1./y_OM_2Den - 1.) \
                 + u_3Den * m_b3Den * (1./y_OM_3Den - 1.) \
                 + u_4Den * m_b4Den * (1./y_OM_4Den - 1.) \
                 + u_5Den * m_b5Den * (1./y_OM_5Den - 1.) \
                 + u_6Den * m_b6Den * (1./y_OM_6Den - 1.) \
                 + u_DNRA1 * m_bDNRA1 * e_NH4_DNRA1 \
                 + u_DNRA2 * m_bDNRA2 * e_NH4_DNRA2 
                   #combined terms of biogen and resp  
##Change then change dilution rate to zero 
        #NO3 production
        prod_NO3 = u_AOX * m_bAOX * e_NO3_AOX \
                 + u_NOO * m_bNOO / y_NO2_NOO

        #NO2 production
        prod_NO2 = u_1Den * m_b1Den * e_NO2_1Den \
                 + prod_NO2_Fac \
                 + u_AOO * m_bAOO * (1./y_NH4_AOO - 1.)

        #N2O production (tracking moles of N, not N2O)
        prod_N2O = u_4Den * m_b4Den * e_N2O_4Den \
                 + u_6Den * m_b6Den * e_N2O_6Den

        #N2 production (tracking moles of N, not N2)
        prod_N2 = u_2Den * m_b2Den * e_N2_2Den \
                + u_3Den * m_b3Den * e_N2_3Den \
                + u_5Den * m_b5Den * e_N2_5Den \
                + u_AOX * m_bAOX * e_N2_AOX 

        ##################################################################
        ### Differential equations (rates of change of state variables)

        #Substrates

        ddt_Sd = dil * (in_Sd - m_Sd) \
                 - uptake_Sd
        
        ddt_O2 = dil * (in_O2 - m_O2) \
                 - uptake_O2
        
        ddt_NH4 = dil * (in_NH4 - m_NH4) \
                  - uptake_NH4 \
                  + prod_NH4
                 
        ddt_NO3 = dil * (in_NO3 - m_NO3) \
                  - uptake_NO3 \
                  + prod_NO3
        
        ddt_NO2 = dil * (in_NO2 - m_NO2) \
                  - uptake_NO2 \
                  + prod_NO2
        
        ddt_N2O = dil * (in_N2O-m_N2O) \
                  - uptake_N2O \
                  + prod_N2O
        
        ddt_N2 = dil * (in_N2-m_N2) \
                 + prod_N2
        
    
        #Biomasses

        ddt_baer = dil * (-m_baer)              \
                   + u_aer * m_baer 
        ddt_bFac = dil * (-m_bFac)              \
                   + u_Fac * m_bFac        
        ddt_b1Den = dil * (-m_b1Den)            \
                   + u_1Den * m_b1Den     
        ddt_b2Den = dil * (-m_b2Den)            \
                   + u_2Den * m_b2Den       
        ddt_b3Den = dil * (-m_b3Den)            \
                   + u_3Den * m_b3Den 
        ddt_b4Den = dil * (-m_b4Den)            \
                   + u_4Den * m_b4Den 
        ddt_b5Den = dil * (-m_b5Den)            \
                   + u_5Den * m_b5Den          
        ddt_b6Den = dil * (-m_b6Den)            \
                   + u_6Den * m_b6Den 
        ddt_bDNRA1 = dil * (-m_bDNRA1)            \
                   + u_DNRA1 * m_bDNRA1 
        ddt_bDNRA2 = dil * (-m_bDNRA2)            \
                   + u_DNRA2 * m_bDNRA2 
        ddt_bAOO = dil * (-m_bAOO)              \
                   + u_AOO * m_bAOO 
        ddt_bNOO = dil * (-m_bNOO)              \
                   + u_NOO * m_bNOO
        ddt_bAOX = dil * (-m_bAOX)              \
                   + u_AOX * m_bAOX 
        
        #integrate (simple scheme here) 
        m_Sd = m_Sd + ddt_Sd * dt
        m_O2 = m_O2 + ddt_O2 * dt
        m_NO3 = m_NO3 + ddt_NO3 * dt
        m_NO2 = m_NO2 + ddt_NO2 * dt
        m_NH4 = m_NH4 + ddt_NH4 * dt
        m_N2 = m_N2 + ddt_N2 * dt
        m_N2O = m_N2O + ddt_N2O * dt 
        m_baer = m_baer + ddt_baer * dt
        m_bFac = m_bFac + ddt_bFac * dt
        m_b1Den = m_b1Den + ddt_b1Den * dt
        m_b2Den = m_b2Den + ddt_b2Den * dt
        m_b3Den = m_b3Den + ddt_b3Den * dt
        m_b4Den = m_b4Den + ddt_b4Den * dt 
        m_b5Den = m_b5Den + ddt_b5Den * dt 
        m_b6Den = m_b6Den + ddt_b6Den * dt 
        m_bDNRA1 = m_bDNRA1 + ddt_bDNRA1 * dt
        m_bDNRA2 = m_bDNRA2 + ddt_bDNRA2 * dt
        m_bAOO = m_bAOO + ddt_bAOO * dt
        m_bNOO = m_bNOO + ddt_bNOO * dt
        m_bAOX = m_bAOX + ddt_bAOX * dt
        
        
        ### Record output at regular interval set above
        if t % interval == 0:
            #print(t*dt)
            i += 1
            #print("Recording output at day",i*out_at_day)
            out_Sd[i] = m_Sd 
            out_O2[i] = m_O2
            out_NO3[i] = m_NO3
            out_NO2[i] = m_NO2 
            out_NH4[i] = m_NH4 
            out_N2O[i] = m_N2O 
            out_N2[i] = m_N2
            out_baer[i] = m_baer
            out_bFac[i] = m_bFac
            out_b1Den[i] = m_b1Den
            out_b2Den[i] = m_b2Den
            out_b3Den[i] = m_b3Den
            out_b4Den[i] = m_b4Den 
            out_b5Den[i] = m_b5Den 
            out_b6Den[i] = m_b6Den
            out_bAOO[i] = m_bAOO
            out_bNOO[i] = m_bNOO
            out_bAOX[i] = m_bAOX
            out_bDNRA1[i] = m_bDNRA1
            out_bDNRA2[i] = m_bDNRA2
            out_facaer[i] = np.nanmean(facaer) #EJZ: should take mean of steady state only -- change later?
            out_time[i] = t*dt #days 
            
    return [out_Sd, out_O2, out_NO3, out_NO2, out_NH4, out_N2O, out_N2, \
            out_baer, out_bFac, out_b1Den, out_b2Den, out_b3Den, out_b4Den, out_b5Den, out_b6Den, out_bAOO, out_bNOO, out_bAOX, \
            out_bDNRA1, out_bDNRA2, out_facaer, out_time]
