#Sept 26 2024: plotting script draft by EJZ for Jemma

#Modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr #can use this to format colorbar tick labels
import pandas as pd

#Axes:

Sin = np.arange(0, 10.001, 0.01) #Sdin_vary, length 101
Nin = np.arange(0, 30.001, 0.1) #NO2in_vary, length 301
dil = 0.04 #per day
Ssupp = Sin#*dil*5 #uM org C/day
Nsupp = Nin#*dil #uM NO2/day

#Import model data:

#fdnra = pd.read_csv("fin_OM_Fig3.csv")
#fdnra = pd.read_csv("DNRA_biomass.csv") #original 
fdnra = pd.read_csv("DNRA_biomass_Fig3_ModAMX.csv") #original 
dnra = fdnra[fdnra.columns[1:]].values #bc first column is the index for Nin. dnra is 101 x 301

#fden = pd.read_csv("fin_NH4_Fig3.csv")
#fden = pd.read_csv("TotalDenit.csv") #original
fden = pd.read_csv("CombinedDenit_biomass_Fig3_ModAMX.csv") 
den = fden[fden.columns[1:]].values

#fanx = pd.read_csv("fin_NO2_Fig3.csv")
#fanx = pd.read_csv("Anammox_biomass.csv") #original
fanx = pd.read_csv("Anammox_biomass_Fig3_ModAMX.csv")
anx = fanx[fden.columns[1:]].values

#fanx = pd.read_csv("Balance.csv") #BALANCE
#anx = fanx[fden.columns[1:]].values

#fanx = pd.read_csv("ratio_check.csv") #ratio check
#anx = fanx[fden.columns[1:]].values


#Yields
#OM
ys_den = 0.286
ys_dnra = 0.154

#NO2
#yn_den = 0.024 #step 1
yn_den = 0.036 #full denit
yn_dnra= 0.035

#kinetic params
vmaxn = 30
kn = 0.1
vmaxs = 3
ks = 1

#R* values:
Ss_den = ks/(ys_den*vmaxs/dil - 1.) 
Ss_dnra = ks/(ys_dnra*vmaxs/dil - 1.) 
Ns_den = kn/(yn_den*vmaxn/dil - 1.)
Ns_dnra = kn/(yn_dnra*vmaxn/dil - 1.)

#Consumption vectors, all starting at point (x, y) = (Ns_den, Ss_dnra)
#Denit consumption
Sv_den = Ss_dnra + yn_den/ys_den*(Nsupp - Ns_den) 
Sv_dnra = Ss_dnra + yn_dnra/ys_dnra*(Nsupp - Ns_den) 
bal = ((yn_den/ys_den)+(yn_dnra/ys_dnra))/2 
Sv_balance = Ss_dnra + bal*(Nsupp - Ns_den) 


#Plot:

fig = plt.figure(figsize = (12,4))

ax1 = plt.subplot(132) 
plt.pcolor(Nsupp, Ssupp, den, rasterized=True, cmap='viridis')
plt.ylim([0, np.max(Ssupp)])
plt.title("B. Denitrification Biomass", loc = "left")
#plt.title("E. NH4 Accumulation", loc = "left")
plt.colorbar(label = "$\mu$M N")
plt.ylabel("Org C supply ($\mu$M C d$^{-1}$)")

#Plot ZNGIs:
#plt.plot(np.array([Ns_den, Ns_den]),np.array([0, np.max(Ssupp)]), color = "white", linestyle = "dashed")
#plt.plot(np.array([Ns_dnra, Ns_dnra]),np.array([0, np.max(Ssupp)]), color = "white", linestyle = "-.")
#plt.plot(np.array([0, np.max(Nsupp)]),np.array([Ss_den, Ss_den]),color = "white", linestyle = "dashed")
#plt.plot(np.array([0, np.max(Nsupp)]),np.array([Ss_dnra, Ss_dnra]),color = "white", linestyle = "-.")
#Plot consumption vectors:
plt.plot(Nsupp, Sv_den, 'w--')
plt.plot(Nsupp, Sv_dnra, 'w-.')
#plt.plot(Nsupp, Sv_balance, 'w--')

ax2 = plt.subplot(131)
plt.pcolor(Nsupp, Ssupp, dnra, rasterized=True, cmap='viridis')
#plt.clim([0,1])
plt.ylim([0, np.max(Ssupp)])
plt.title("A. DNRA Biomass", loc = "left")
#plt.title("D. OM Accumulation", loc = "left")
plt.colorbar(label = "$\mu$M N")
plt.ylabel("Org C supply ($\mu$M C d$^{-1}$)")
#Plot consumption vectors:
plt.plot(Nsupp, Sv_den, 'w--')
plt.plot(Nsupp, Sv_dnra, 'w-.')
#plt.plot(Nsupp, Sv_balance, 'w--')

ax3 = plt.subplot(133)
plt.pcolor(Nsupp, Ssupp, anx, rasterized=True, cmap='viridis')
plt.ylim([0, np.max(Ssupp)])
plt.title("C. Anammox Biomass", loc = "left")
#plt.title("F. NO$_2^-$ Accumulation", loc = "left")
plt.colorbar(label = "$\mu$M N ")
#Plot consumption vectors:
plt.plot(Nsupp, Sv_den, 'w--')
plt.plot(Nsupp, Sv_dnra, 'w-.')
#plt.plot(Nsupp, Sv_balance, 'w--')



#Set axis labels:
ax1.axes.xaxis.set_ticklabels([])
ax2.axes.xaxis.set_ticklabels([])
plt.ylabel("Org C supply ($\mu$M C d$^{-1}$)")

ax3 = plt.xlabel("NO$_2$ supply ($\mu$M N )")


plt.tight_layout()
plt.savefig("FigS2A-C_raw.png", dpi = 600)
plt.show()



