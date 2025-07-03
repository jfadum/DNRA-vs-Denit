
#initialize arrays

# Nutrients
fin_O2 = np.ones((len(var1), len(var2))) * np.nan
fin_Sd = np.ones((len(var1), len(var2))) * np.nan
fin_NO3 = np.ones((len(var1), len(var2))) * np.nan
fin_NO2 = np.ones((len(var1), len(var2))) * np.nan
fin_NH4 = np.ones((len(var1), len(var2))) * np.nan
fin_N2 = np.ones((len(var1), len(var2))) * np.nan
fin_N2O = np.ones((len(var1), len(var2))) * np.nan # add N2O
fin_N2O = np.ones((len(var1), len(var2))) * np.nan # add N2O

# Biomasses
fin_baer = np.ones((len(var1), len(var2))) * np.nan
fin_bFac = np.ones((len(var1), len(var2))) * np.nan
fin_b1Den = np.ones((len(var1), len(var2))) * np.nan
fin_b2Den = np.ones((len(var1), len(var2))) * np.nan
fin_b3Den = np.ones((len(var1), len(var2))) * np.nan
fin_b4Den = np.ones((len(var1), len(var2))) * np.nan # add biomass for Den4
fin_b5Den = np.ones((len(var1), len(var2))) * np.nan # add biomass for Den5
fin_b6Den = np.ones((len(var1), len(var2))) * np.nan # add biomass for Den6 
fin_bAOO = np.ones((len(var1), len(var2))) * np.nan
fin_bNOO = np.ones((len(var1), len(var2))) * np.nan
fin_bAOX = np.ones((len(var1), len(var2))) * np.nan
fin_bDNRA1 = np.ones((len(var1), len(var2))) * np.nan
fin_bDNRA2 = np.ones((len(var1), len(var2))) * np.nan

# track facultative average respiration
fin_facaer = np.ones((len(var1), len(var2))) * np.nan

