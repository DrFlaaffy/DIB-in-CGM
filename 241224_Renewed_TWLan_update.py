#!/usr/bin/env python
# coding: utf-8

# In[76]:


from scipy.optimize import curve_fit
import numpy as np
import astropy.io.fits as pf
import matplotlib.pyplot as plt
import time
import matplotlib
from scipy.optimize import curve_fit
from matplotlib import rc
#Fonts for plots
rc = {"font.family" : "serif", 
      "mathtext.fontset" : "stix",'ytick.labelsize':15,'xtick.labelsize':15}
plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
#------------------------------
#Data and presets
#------------------------------
#Data
data = pf.open('250101_bootstrapped_spectra_TWLan_mod_500.fits')
bt_spectra = data[0].data
#bt_spectra_4180 = data[1].data
#bt_spectra_4200 = data[2].data
bt_spectra_4220 = data[1].data
has_data = pf.open('241125_new_NMF_method_catalog_v3_mask_radius_35_no_synthetic_signals.fits')[3].data
has_data = np.where(has_data == 1)[0]
s_orig = pf.open('241125_new_NMF_method_catalog_v3_mask_radius_35_no_synthetic_signals.fits')[1].data[has_data]
#s_4180 = pf.open('241126_new_NMF_method_catalog_v3_mask_radius_35_synthetic_signals_4180.fits')[1].data[has_data]
#s_4200 = pf.open('241125_new_NMF_method_catalog_v2_mask_radius_35_synthetic_signals_4200.fits')[1].data[has_data]
s_4220 = pf.open('241126_new_NMF_method_catalog_v3_mask_radius_35_synthetic_signals_4220.fits')[1].data[has_data]
MA = pf.open('DR16_absorber.fits')
catalog = MA[1].data
A_redshift = np.array(catalog['Z_ABS'])[has_data]
W_0 = np.array(catalog['REST_EW_MGII_2796'])[has_data]
#Presets
bins = [0.4, 0.6628908034679974, 1.098560543306118, 1.8205642030260805, 3.0170881682725827, np.amax(W_0)]
log_rest_frame = np.arange(np.log10(2700),np.log10(5000),1e-4)
x = 10**log_rest_frame

#------------------------------
#Functions
#------------------------------
#Composite spectra maker
def composite_bins(spectra):
    '''
    Input:
        spectra: catalog of spectra
        *Predefine bins (boundaries of REW) and W_0 (REW of MgII)*
    Output:
        composite_spectra: array of composite spectra as a function of REW
    '''
    composite_spectra = np.zeros((len(bins)-1,len(spectra[1])))
    for i in range(len(bins)-1):
        spec = np.where((W_0 > bins[i]) & (W_0 < bins[i+1]))
        spectrum = []
        for i_spec in range(len(spectra[1])):
            spectrum.append(np.nanmedian(spectra[spec,i_spec]))
        composite_spectra[i] = spectrum
    return composite_spectra
#Spectral regions
def region(fitting_size,mu):
    model_region = np.where((x > mu-fitting_size) & (x < mu+fitting_size))[0]
    return model_region
#Mock spectra function
def model_gaussian(x,A,mu):
    return 1 - A * np.exp(- (x - mu)**2 / 2 / 6.9**2)
#Fitting code
bound = [[-1],[1]]
def DIB_fitting(spectra,model,region): #2D spectra array
    params = []
    amp = []
    for i in range(len(spectra)):
        popt, pcov = curve_fit(model,x[region],spectra[i][region],p0=(0.001),method='trf',bounds=bound, maxfev=10000)
        params.append(popt[0] * 6.9 * np.sqrt(2 * np.pi))
        amp.append(popt[0])
        #print(i)
    REW_err = np.std(params)
    Amp = np.median(amp)
    return(params,REW_err,amp)
#Measurement error
def composite_error(spectra):
    '''
    Input should be bootstrapped spectra
    '''
    spectrum = []
    for i_spec in range(2677):
        spectrum.append(np.nanstd(spectra[:,i_spec]))
    return np.array(spectrum)
#E(B-V)
def E_BV(Rest_EW,z): 
    return 0.017*(Rest_EW**1.6)*(1+z)**(-0.01)/1.55/(1+z)**1.2
#REW of DIB4430
def EWDIB(EBV):
    return 1.22*EBV**0.89

def EW_DIB(EBV):
    return 1.18*EBV**0.87,1.26*EBV**0.91
#Mock spectra fitting trend
def trend(x,A):
    return A*x**0.89
#Median of bins
def median(value):
    '''
    Returns an list with len(list) = len(bins) - 1
    '''
    median = []
    for i in range(len(bins)-1):
        spec = np.where((W_0 > bins[i]) & (W_0 < bins[i+1]))
        median.append(np.median(value[spec]))
    return median
#Bootstrap error
def bootstrap_error(number,bt_spectra,model,region):
    std = np.zeros((len(bins)-1,number))
    for i_bin in range(len(bins)-1):
        for i_spec in range(number): 
            o = DIB_fitting(bt_spectra[:,i_spec],model,region)[0]
            std[i_bin,i_spec] = np.array(o[i_bin])
    err = []
    for i in range(5):
        err.append(np.std(std[i]))
    return np.array(err), std
#Plotting and convenience code
def synthetic_4220(x,A):
    return 1. - A * 1.0* np.exp(- (x - 4220.0)**2 / 2. / 6.9**2)
def synthetic_4200(x,A):
    return 1. - A * 1.0* np.exp(- (x - 4200.0)**2 / 2. / 6.9**2)
def synthetic_4180(x,A):
    return 1. - A * 1.0* np.exp(- (x - 4180.0)**2 / 2. / 6.9**2)
def synthetic_gaussian(x,A):
    return 1. - A * 1.0* np.exp(- (x - 4429.0)**2 / 2. / 6.9**2)


# In[3]:


def DIB_fitting_v2(spectra,model,region): #2D spectra array
    params = []
    amp = []
    for i in range(len(spectra)):
        popt, pcov = curve_fit(model,x[region],spectra[i][region],p0=(0.0))
        params.append(popt[0] * 6.9 * np.sqrt(2 * np.pi))
    
    return params


# In[4]:


fitting_size = 70
std = np.zeros((5,500))
for i_bin in range(5):
    for i_spec in range(500): 
        o = DIB_fitting_v2(bt_spectra_4220[:,i_spec],synthetic_4220,region(fitting_size,mu=4220))
        std[i_bin,i_spec] = np.array(o[i_bin])


# In[5]:


#making composite spectra
c_orig = composite_bins(s_orig)
#c_4180 = composite_bins(s_4180)
#c_4200 = composite_bins(s_4200)
c_4220 = composite_bins(s_4220)


# In[6]:


orig_spectra = DIB_fitting(c_orig,synthetic_gaussian,region(fitting_size,mu=4429.9))
#synth_spectra_4180 = DIB_fitting(c_4180,synthetic_4180,region(fitting_size,mu=4180))
#synth_spectra_4200 = DIB_fitting(c_4200,synthetic_4200,region(fitting_size,mu=4200))
synth_spectra_4220 = DIB_fitting(c_4220,synthetic_4220,region(fitting_size,mu=4220))


# In[7]:


o_err,matrix = bootstrap_error(500,bt_spectra,synthetic_gaussian,region(fitting_size,mu=4429.9))
#err_4180 = bootstrap_error(500,bt_spectra_4180,synthetic_4180,region(fitting_size,mu=4180))
#err_4200 = bootstrap_error(500,bt_spectra_4200,synthetic_4200,region(fitting_size,mu=4200))
err_4220,erro_max = bootstrap_error(500,bt_spectra_4220,synthetic_4220,region(fitting_size,mu=4220))


# In[8]:


err_4220


# In[9]:


composite_err = np.zeros((5,2677))
for i in range(len(bins)-1):
    composite_err[i] = composite_error(bt_spectra[i])
    
#composite_err_4180 = np.zeros((5,2677))
#for i in range(len(bins)-1):
#    composite_err_4180[i] = composite_error(bt_spectra_4180[i])
#composite_err_4200 = np.zeros((5,2677))
#for i in range(len(bins)-1):
#    composite_err_4200[i] = composite_error(bt_spectra_4200[i])
composite_err_4220 = np.zeros((5,2677))
for i in range(len(bins)-1):
    composite_err_4220[i] = composite_error(bt_spectra_4220[i])


# In[10]:


m_w0 = np.array(median(W_0))
m_rs = np.array(median(A_redshift))
x_dust = E_BV(m_w0,m_rs)
print(x_dust)


# In[11]:


popt_o, pcov_o = curve_fit(trend,x_dust,orig_spectra[0],sigma = o_err, absolute_sigma='True')
perr_o = np.sqrt(np.diag(pcov_o))
print(popt_o[0],'+-',perr_o[0],'*E_{B-V}^0.89')

#popt_1, pcov_1 = curve_fit(trend,x_dust,synth_spectra_4180[0], sigma = err_4180)
#perr_1 = np.sqrt(np.diag(pcov_1))
#print(popt_1[0],'+-',perr_1[0],'*E_{B-V}^0.89')

#popt_2, pcov_2 = curve_fit(trend,x_dust,synth_spectra_4200[0], sigma = err_4200)
#perr_2 = np.sqrt(np.diag(pcov_2))
#print(popt_2[0],'+-',perr_2[0],'*E_{B-V}^0.89')

popt_3, pcov_3 = curve_fit(trend,x_dust,synth_spectra_4220[0], sigma = err_4220, absolute_sigma='True')
perr_3 = np.sqrt(np.diag(pcov_3))
print(popt_3[0],'+-',perr_3[0],'*E_{B-V}^0.89')


# In[77]:


EBV = np.linspace(0,E_BV(6,np.median(A_redshift)),100)
expected = EW_DIB(EBV)


# In[79]:


fig = plt.figure(figsize=(10,7.5))
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()
ax1.set_xscale('log')
ax2.set_xscale('log')

# rc('text', usetex=True)

ax1.plot(EBV,trend(EBV,*popt_o),lw=2,ls=':',color='tomato', zorder=15)
# ax1.plot(EBV,trend(EBV,*popt_1),lw=1,ls='dashed',color='darkorange', zorder=15)
# ax1.plot(EBV,trend(EBV,*popt_2),lw=1,ls='dashed',color='darkgreen', zorder=15)
ax1.plot(EBV,trend(EBV,*popt_3),lw=2,ls=':',color='royalblue', zorder=10)
ax1.fill_between(EBV,expected[0],expected[1],color='lightgray',zorder=9)
ax1.errorbar(x_dust,orig_spectra[0],yerr=o_err,fmt = '.',mfc='red',ecolor='black',marker = '.',mec='black',mew=2,lw=2,markersize=25,capsize=5,label='Measured DIB Absorption Strength in CGM',zorder=15)
ax1.errorbar(x_dust,synth_spectra_4220[0],yerr=err_4220,fmt = '.',mfc='cornflowerblue',ecolor='black',marker = '.',mec='black',mew=2,lw=2,markersize=25,capsize=5,label='Mock Absorption at 4220 $\AA$',zorder=15)
ax1.errorbar([0.03,0.04,0.05],np.array([68,94,115])*0.001,yerr=[0.005,0.011,0.014],fmt = '.', marker='*',uplims=[0.005,0.011,0.014],mfc='darkorchid',ecolor='black',mec='black',mew=2,lw=2,markersize=23,capsize=5,zorder=14,label='3$\sigma$ Upper Limits in DLAs')
ax1.errorbar([0.0274,0.0478],np.array([39.54,100.015])*0.001,yerr=[0.006,0.011],fmt = '.', marker='s',mfc='green',ecolor='black',mec='black',mew=2,lw=2,markersize=15,capsize=5,zorder=13,label='DIB Detections in Milky Way QSO Sightlines')
# ax1.errorbar(x_dust,test,yerr=test_err,fmt = '.',color='black',ecolor='black',marker = '.',markersize=15,capsize=7,label='Average Mock DIB Absorption',zorder=15)
ax1.set_xscale('log')
ax1.set_xticks([0.00125,0.0025,0.005,0.01,0.02,0.04],['0.00125','0.0025','0.005','0.01','0.02','0.04'],fontsize=15)
ax1.set_yticks([-0.05,-0.025,0.00,0.025,0.05,0.075,0.100,0.125],['-0.05','','0.00','','0.05','','0.10',''],fontsize=15)

ax1.set_xlim(E_BV(0.4,np.median(A_redshift)),E_BV(4,np.median(A_redshift)))
ax1.set_ylim(-0.05,0.14)
ax1.legend(loc='upper left',frameon = False,fontsize = 16,labelcolor = ['red','royalblue','darkorchid','darkgreen'],markerscale=0.7)

ax2.set_xscale('log')
ax2.set_xlim(0.3,4.0)
ax2.set_xticks([0.5,1.0,2.0,3.0,4.0],[0.5,1.0,2.0,3.0,4.0],fontsize=16)

ax2.set_xlabel('$W_0^{\lambda2796}\ [\AA]$',fontsize=16)
ax1.set_xlabel('$E_{(B-V)}\ [mag]$',fontsize=16)
ax1.set_ylabel('$W_0^{\lambda4430}\ [\AA]$',fontsize=16)
plt.savefig('250528_DIB_vs_Mock_signals.pdf',dpi=300)


# In[13]:


fig = plt.figure(figsize=(8,6))
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()
ax1.set_xscale('log')
ax2.set_xscale('log')

ax1.plot(EBV,trend(EBV,*popt_o),lw=1,ls='dashed',color='red', zorder=15)
# ax1.plot(EBV,trend(EBV,*popt_1),lw=1,ls='dashed',color='darkorange', zorder=15)
# ax1.plot(EBV,trend(EBV,*popt_2),lw=1,ls='dashed',color='darkgreen', zorder=15)
ax1.plot(EBV,trend(EBV,*popt_3),lw=1,ls='dashed',color='blue', zorder=15)
ax1.plot(EBV,expected,color='black',label='Expected Absorption from MW Trend')
ax1.errorbar(x_dust,orig_spectra[0],yerr=o_err,fmt = '.',color='red',ecolor='red',marker = '.',markersize=15,capsize=7,label='Measured DIB Absorption Strength in CGM',zorder=15)
# ax1.errorbar(x_dust,synth_spectra_4180[0],yerr=err_4180,fmt = '.',color='darkgreen',ecolor='darkgreen',marker = '.',markersize=15,capsize=7,alpha=0.5,label='Mock Absorption at 4180 $\AA$',zorder=15)
# ax1.errorbar(x_dust,synth_spectra_4200[0],yerr=err_4200,fmt = '.',color='darkorange',ecolor='darkorange',marker = '.',markersize=15,capsize=7,alpha=0.5,label='Mock Absorption at 4200 $\AA$',zorder=15)
ax1.errorbar(x_dust,synth_spectra_4220[0],yerr=err_4220,fmt = '.',color='blue',ecolor='blue',marker = '.',markersize=15,capsize=7,alpha=0.5,label='Mock Absorption at 4220 $\AA$',zorder=15)
ax1.errorbar([0.03,0.04,0.05],np.array([68,94,115])*0.001,yerr=[0.005,0.011,0.014],fmt = '.', marker='*', markersize=15,uplims=[0.005,0.011,0.014],color='darkorchid',ecolor='darkorchid',label='3$\sigma$ Upper Limits in DLAs')
ax1.errorbar([0.0274,0.0478],np.array([39.54,100.015])*0.001,yerr=[0.006,0.011],fmt = '.', marker='s', markersize=10,capsize=7,color='darkgreen',ecolor='darkgreen',label='DIB Detections in Milky Way QSO Sightlines')
# ax1.errorbar(x_dust,test,yerr=test_err,fmt = '.',color='black',ecolor='black',marker = '.',markersize=15,capsize=7,label='Average Mock DIB Absorption',zorder=15)
ax1.set_xscale('log')
ax1.set_xticks([0.00125,0.0025,0.005,0.01,0.02,0.04],['0.00125','0.0025','0.005','0.01','0.02','0.04'],fontsize=15)
ax1.set_yticks([-0.05,-0.025,0.00,0.025,0.05,0.075,0.100,0.125],['-0.05','','0.00','','0.05','','0.10',''],fontsize=15)

ax1.set_xlim(E_BV(0.4,np.median(A_redshift)),E_BV(4,np.median(A_redshift)))
ax1.set_ylim(-0.05,0.14)
ax1.legend(loc='upper left',frameon = False,fontsize = 14,labelcolor = ['black','red','blue','darkorchid','darkgreen'])

ax2.set_xscale('log')
ax2.set_xlim(0.3,4.0)
ax2.set_xticks([0.5,1.0,2.0,3.0,4.0],[0.5,1.0,2.0,3.0,4.0],fontsize=15)

ax2.set_xlabel('$W_0^{\lambda2796}\ [\AA]$',fontsize=15)
ax1.set_xlabel('$E_{(B-V)}\ [mag]$',fontsize=15)
ax1.set_ylabel('$W_0^{\lambda4430}\ [\AA]$',fontsize=15)
plt.savefig('250225_DIB_vs_Mock_signals_3.png',dpi=300)


# In[82]:


fig,axs = plt.subplots(len(bins)-1,2,sharex='col',sharey='row',figsize=(7.5,10),gridspec_kw={'width_ratios': [1, 1]})
plt.subplots_adjust(wspace=0,hspace=0,left=0.5,bottom=0.1)
#Setting x limits
axs[0,0].set_xlim(4380,4480)
axs[0,1].set_xlim(4170,4270)
# axs[0,2].set_xlim(4150,4250)
# axs[0,3].set_xlim(4170,4270)
axs[4,0].set_xticks([4390,4410,4430,4450,4470])
# axs[4,1].set_xticks([4140,4160,4180,4200,4220])
# axs[4,2].set_xticks([4160,4180,4200,4220,4240])
axs[4,1].set_xticks([4180,4200,4220,4240,4260])
#Setting up y limits and other shennenigans
for i in range(len(bins)-1):
    axs[i,0].set_ylim(0.99,1.005)
    axs[i,1].set_ylim(0.99,1.005)
    axs[i,0].set_yticks([0.995,1,1.005],[0.995,1,1.005],fontsize=15)
    axs[i,0].tick_params(direction='in',left=True,right=False,labelleft=True,labelright=False)
    axs[i,1].tick_params(direction='in',left=True,right=False,labelleft=False,labelright=False)
    if i<4:
        axs[i,0].text(4390,0.992,'$%0.2f<W_{0}^{\lambda2796}<%0.2f \ \AA$' % (bins[i],bins[i+1]),fontsize=12)
    else:
        axs[i,0].text(4390,0.985,'$W_{0}^{\lambda2796}>%0.2f \ \AA$' % (bins[i]),fontsize=12)
        
#     axs[i,2].tick_params(direction='in',left=True,right=False,labelleft=False,labelright=False)
#     axs[i,3].tick_params(direction='in',left=True,right=False,labelleft=False,labelright=False)
#Data
    
for i in range(len(bins)-1):
    axs[i,0].fill_between(x,c_orig[i]+composite_err[i],c_orig[i]-composite_err[i],color='lightsteelblue')
    axs[i,0].plot(x,c_orig[i],color='blue')
    axs[i,0].plot(x,synthetic_gaussian(x,orig_spectra[2][i]),color='red')
    
#     axs[i,1].fill_between(x,c_4180[i]+composite_err_4180[i],c_4180[i]-composite_err_4180[i],color='lightsteelblue')
#     axs[i,1].plot(x,c_4180[i],color='blue')
#     axs[i,1].plot(x,synthetic_4180(x,synth_spectra_4180[2][i]),color='red')
    
#     axs[i,2].fill_between(x,c_4200[i]+composite_err_4200[i],c_4200[i]-composite_err_4200[i],color='lightsteelblue')
#     axs[i,2].plot(x,c_4200[i],color='blue')
#     axs[i,2].plot(x,synthetic_4200(x,synth_spectra_4200[2][i]),color='red')
    
    axs[i,1].fill_between(x,c_4220[i]+composite_err_4220[i],c_4220[i]-composite_err_4220[i],color='lightsteelblue')
    axs[i,1].plot(x,c_4220[i],color='blue')
    axs[i,1].plot(x,synthetic_4220(x,synth_spectra_4220[2][i]),color='red')
#Corrections
axs[4,0].set_ylim(0.98,1.015)
axs[4,1].set_yticks([0.98,1.0],[0.98,1.0],fontsize=15)
#Labels
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel("Wavelength [$\AA$]",fontsize=15)
plt.text(-0.13,0.4, 'Normalized Flux', rotation=90, fontsize=15)
fig.tight_layout()
fig.subplots_adjust(top=0.93)
#fig.suptitle('Comparison of Observed and Mock Signal',fontsize=16)
axs[0,0].set_title('Observed DIB$\lambda$4430 Signal',size=15)
axs[0,1].set_title('Mock Signal at $\lambda=4220 \ \AA$',size=15)
# plt.savefig('Plot_1.png',dpi=300)
plt.savefig('Composite_spectra.pdf')


# In[ ]:





# In[ ]:





# In[15]:


pf.info('MA_catalog.fits')


# In[16]:


indv = pf.open('MA_catalog.fits')[0].data
indv


# In[18]:


data = pf.open('250213_new_NMF_method_catalog_v1_extending_to_6100.fits')
# synth_data = pf.open('241126_new_NMF_method_catalog_v3_mask_radius_35_synthetic_signals_4180.fits')
MA = pf.open('DR16_absorber.fits')
has_data = data[3].data
has_data = np.where((has_data == 1)&(np.array(catalog['Z_ABS'])>0.4)&(np.array(catalog['Z_ABS'])<0.54))[0]
spectra = data[1].data[has_data]
# s_spectra = synth_data[1].data[has_data]
catalog = MA[1].data

Q_redshift = np.array(catalog['Z_QSO'])[has_data]
A_redshift = np.array(catalog['Z_ABS'])[has_data]
W_0 = np.array(catalog['REST_EW_MGII_2796'])[has_data]


# In[27]:


bins = [0,7]
test = composite_bins(spectra)


# In[49]:


log_rest_frame = np.arange(np.log10(2700),np.log10(6100),1e-4)
x = 10**log_rest_frame
mgii =  np.where((x > 2794) & (x < 2805))[0]
plt.plot(x,test[0])
# plt.xlim(5800,6000)
plt.xlim(3900,4000)
plt.ylim(0.95,1.01)


# In[39]:


np.where(W_0 > 4)


# In[87]:


fig,axs = plt.subplots(2,1,sharex='col',sharey='row',figsize=(16,9))
plt.subplots_adjust(wspace=0,hspace=0,left=0,bottom=0)
for i in range(2):
    axs[i].set_ylim(0.4,1.6)
    axs[i].set_xlim(2700,5000)
    axs[i].set_yticks([0.5,1,1.5],[0.5,1,1.5],fontsize=24)
    axs[i].set_xticks([3000,3500,4000,4500,5000],[3000,3500,4000,4500,5000],fontsize=24)
axs[0].plot(x,spectra[347],color = 'blue',lw=0.5)
axs[1].plot(x,spectra[347],color = 'lightsteelblue',alpha=0.3)
axs[1].plot(x,test[0],color='blue',lw=2)
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel("Wavelength [$\AA$]",fontsize=24)
plt.text(-0.08,0.35, 'Normalized Flux', rotation=90, fontsize=24)
plt.savefig('250515_demonstration_1.png',dpi=300,bbox_inches = 'tight')


# In[89]:


fig,axs = plt.subplots(2,1,sharex='col',sharey='row',figsize=(8,9))
plt.subplots_adjust(wspace=0,hspace=0,left=0,bottom=0)
for i in range(2):
    axs[i].set_ylim(0.99,1.01)
    axs[i].set_xlim(3820,3880)
    axs[i].set_yticks([0.995,1,1.005],[0.995,1,1.005],fontsize=24)
#     axs[i].set_xticks([3000,3500,4000,4500,5000],[3000,3500,4000,4500,5000],fontsize=24)
axs[0].plot(x,spectra[347],color = 'red',lw=2)
# axs[1].plot(x,spectra[347],color = 'lightsteelblue',alpha=0.3)
axs[1].plot(x,test[0],color='red',lw=2)
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
# plt.xlabel("Wavelength [$\AA$]",fontsize=24)
# plt.text(-0.08,0.35, 'Normalized Flux', rotation=90, fontsize=24)
plt.savefig('250515_demonstration.png',dpi=300,bbox_inches = 'tight')

