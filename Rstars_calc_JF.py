

#####################################################################
#%% calculate R*-stars for all microbes (not needed for model results, only for plots)
from diffusive_gas_coefficient import po_coef
from diffusive_gas_coefficient import pn2o_coef # use N2O diffusion to define the max N2O uptake rate of a cell (den5 N2O-->N2)
from O2_star_Xin import O2_star
from N2O_star_Xin import N2O_star # add new equation to cal N2O star based on diffusion
from R_star_Xin import R_star

#Gasuptake == 2:
# O2 (nM-O2) 
O2_star_aer = O2_star(dil, Qc_aer, diam_aer, dc, y_O2_aer*CN_aer, K_O2_aer)[1]
O2_star_fac = O2_star(dil, Qc_aer, diam_aer, dc, y_O2_aerFac*CN_aer, K_O2_aer)[1]
O2_star_aoo = O2_star(dil, Qc_aoo, diam_aoo, dc, y_O2_AOO*CN_aoo, K_O2_AOO)[1]
O2_star_noo = O2_star(dil, Qc_noo, diam_noo, dc, y_O2_NOO*CN_noo, K_O2_NOO)[1]
# N2O (nM-N)
N2O_star_den5 = N2O_star(dil, Qc_den, diam_den, dc_n2o, y_N2O_5Den*CN_den, K_N2O_Den)[1]
# OM
OM_star_aer = R_star(dil, K_S, VmaxS, y_OM_aer)
#OM_star_fac = R_star(dil, K_S, VmaxS, y_OM_aerFac)
OM_star_den1 = R_star(dil, K_S, VmaxS, y_OM_1Den)
OM_star_den2 = R_star(dil, K_S, VmaxS, y_OM_2Den)
OM_star_den3 = R_star(dil, K_S, VmaxS, y_OM_3Den)
OM_star_den4 = R_star(dil, K_S, VmaxS, y_OM_4Den)
OM_star_den5 = R_star(dil, K_S, VmaxS, y_OM_5Den)
OM_star_den6 = R_star(dil, K_S, VmaxS, y_OM_6Den)
# Ammonia
Amm_star_aoo = R_star(dil, K_NH4_AOO, VmaxNH4_AOO, y_NH4_AOO)
Amm_star_aox = R_star(dil, K_NH4_AOX, VmaxNH4_AOX, y_NH4_AOX)
# Nitrite
nitrite_star_den2 = R_star(dil, K_DIN_Den, VmaxDIN_Den, y_NO2_2Den)
nitrite_star_den4 = R_star(dil, K_DIN_Den, VmaxDIN_Den, y_NO2_4Den)
nitrite_star_noo = R_star(dil, K_NO2_NOO, VmaxNO2_NOO, y_NO2_NOO)
nitrite_star_aox = R_star(dil, K_NO2_AOX, VmaxNO2_AOX, y_NO2_AOX)
nitrite_star_dnra = R_star(dil, K_DIN_DNRA, VmaxDIN_DNRA, y_NO2_DNRA)
# Nitrate
nitrate_star_fac = R_star(dil, K_DIN_Den, VmaxDIN_Den, y_NO3_1DenFac)
nitrate_star_den1 = R_star(dil, K_DIN_Den, VmaxDIN_Den, y_NO3_1Den)
nitrate_star_den3 = R_star(dil, K_DIN_Den, VmaxDIN_Den, y_NO3_3Den)
nitrate_star_den6 = R_star(dil, K_DIN_Den, VmaxDIN_Den, y_NO3_6Den)
nitrate_star_dnra = R_star(dil, K_DIN_DNRA, VmaxDIN_DNRA, y_NO3_DNRA)



print("Rstars calculated")
