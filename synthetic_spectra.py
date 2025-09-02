#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy.optimize import curve_fit
import numpy as np
import astropy.io.fits as pf
import data_info as data_info
import matplotlib.pyplot as plt
import os
import scipy.signal as ss
from astropy.wcs import WCS
from astropy.visualization import ZScaleInterval


# In[2]:


def smooth_spectrum(spectrum,window):
    test_array = np.zeros((1,len(spectrum)))
    test_array[0]=spectrum
    median_smooth = ss.medfilt(test_array,[1,window])
    return median_smooth[0]

def tell_me_which_basis(redshift):
    if redshift<0.6:
        index=0 
    elif (redshift<1.0) & (redshift>=0.6):
        index=1
    elif (redshift<2.5) & (redshift>=1.0):
        index=2
    elif (redshift>=2.5) & (redshift<4.7):
        index=3

    return index 


# In[3]:


def load_quasar_NMF_basis():
    filename = ['high_resolutionQSO_NMF_basis_z000_100_norm4150.fits','high_resolutionQSO_NMF_basis_z040_179_norm3020.fits','high_resolutionQSO_NMF_basis_z080_280_norm2150.fits','high_resolutionQSO_NMF_basis_z200_479_norm1420.fits']
    dir = './'
    dir_structure ={}
    for index,i_filename in enumerate(filename):
        data = pf.open(dir+i_filename)
        #log_wave = np.log10(data[1].data['WAVE'][0])
        #eigenvector = data[1].data['EIGEN_VECTORS'][0]
        log_wave = np.log10(data[1].data)
        eigenvector = data[0].data

        dir_structure[index]={'log_wave':log_wave,'eigenvectors':eigenvector}

    return dir_structure

log_rest_frame = np.arange(np.log10(2700),np.log10(5000),1e-4)
NMF_basis = load_quasar_NMF_basis()

def NMF_normalization_v2(log_obs_wave,spectrum,inv,redshift,NMF_basis):
	'''
	Key function 
	input: 
	log_obs_wave - observed wavelength in log10 scale
	spectrum - observed spectrum in observed wavelength
	inv - inverse variance of the spectrum
	Redshift - the QSO redshift
	NMF_basis - from function load_quasar_NMF_basis
	
	Output:
	log_obs_wave,spectrum_back,inv_back,spectrum_back_median,inv_back_median,model_back, chi2_red
	log_obs_wave  - original observed wavelength in log10 scale
	spectrum_back - QSO spectrum / the best NMF model
	inv_back - the corresponding inverse-variance
	spectrum_back_median - QSO spectrum / the best NMF model / a median filter smoothing
	inv_back_median - the corresponding inverse-variance of spectrum_back_median
	model_back - the best fit model
	
	'''
	basis_id = tell_me_which_basis(redshift)
	log_base_wave = NMF_basis[basis_id]['log_wave']
	
	wavelength_coverage = np.where(inv!=-999)
	used_log_obs_wave = log_obs_wave[wavelength_coverage[0]]
	log_qso_restframe = used_log_obs_wave-np.log10(1.+redshift)
	spectrum_restframe = data_info.spline_respec(log_qso_restframe,spectrum[wavelength_coverage[0]],log_base_wave)
	inv_restframe = data_info.spline_respec(log_qso_restframe,inv[wavelength_coverage[0]],log_base_wave)
	
	mask_region = np.where((log_base_wave>max(log_qso_restframe)) | (log_base_wave<min(log_qso_restframe)))
	spectrum_restframe[mask_region[0]]=0
	inv_restframe[mask_region[0]]=0
	

	X = np.zeros((1,len(log_base_wave)))
	V = np.zeros((1,len(log_base_wave)))
	M = np.ones((1,len(log_base_wave)),dtype=bool)
	
	X[0,:]=spectrum_restframe
	V[0,:]=inv_restframe
	M[0,mask_region[0]]=False


	H = NMF_basis[basis_id]['eigenvectors']
	g = nmf.NMF(X, V=V, M=M, H=H.T,n_components=12)
	chi2_red, time_used = g.SolveNMF(W_only=True)
	
	best_fit_model = np.dot(g.W[0], g.H)
	SEDs_normalized = np.array(spectrum_restframe)*1.0/best_fit_model
	inv_normalized = np.array(inv_restframe)*1.0*(best_fit_model**2)

	SEDs_subtracted = np.array(spectrum_restframe)*1.0-best_fit_model
	inv_subtracted = np.array(inv_restframe)*1.0


	SEDs_normalized_median = np.array(SEDs_normalized)*1.0/smooth_spectrum(SEDs_normalized,71)
	inv_normalized_median = np.array(inv_normalized)*1.0*(smooth_spectrum(SEDs_normalized,71)**2)

	SEDs_subtracted_median = np.array(SEDs_subtracted)-smooth_spectrum( np.array(SEDs_subtracted),71)
	inv_subtracted_median = np.array(inv_restframe)*1.0
	
	#search = np.where(inv==999999)
	SEDs_normalized_median[mask_region[0]]=0
	inv_normalized_median[mask_region[0]]=0
	SEDs_normalized_median[np.isnan(SEDs_normalized_median)]=0
	SEDs_normalized_median[np.isinf(SEDs_normalized_median)]=0

	inv_normalized_median[np.isnan(SEDs_normalized_median)]=0
	inv_normalized_median[np.isinf(SEDs_normalized_median)]=0

	SEDs_subtracted_median[mask_region[0]]=-999
	SEDs_subtracted_median[np.isnan(SEDs_subtracted_median)]=-999
	SEDs_subtracted_median[np.isinf(SEDs_subtracted_median)]=-999
	inv_subtracted_median[mask_region[0]]=0
	inv_subtracted_median[np.isnan(SEDs_subtracted_median)]=0
	inv_subtracted_median[np.isinf(SEDs_subtracted_median)]=0

	model_back = data_info.spline_respec(log_base_wave+np.log10(1.+redshift),best_fit_model,log_obs_wave)
	spectrum_back = data_info.spline_respec(log_base_wave+np.log10(1.+redshift),SEDs_normalized,log_obs_wave)
	inv_back = data_info.spline_respec(log_base_wave+np.log10(1.+redshift),inv_normalized,log_obs_wave)

	spectrum_back_median = data_info.spline_respec(log_base_wave+np.log10(1.+redshift),SEDs_normalized_median,log_obs_wave)
	inv_back_median = data_info.spline_respec(log_base_wave+np.log10(1.+redshift),inv_normalized_median,log_obs_wave)


	spectrum_back_substract = data_info.spline_respec(log_base_wave+np.log10(1.+redshift),SEDs_subtracted,log_obs_wave)
	inv_back_substract = data_info.spline_respec(log_base_wave+np.log10(1.+redshift),inv_subtracted,log_obs_wave)


	spectrum_back_substract_median = data_info.spline_respec(log_base_wave+np.log10(1.+redshift),SEDs_subtracted_median,log_obs_wave)
	inv_back_substract_median = data_info.spline_respec(log_base_wave+np.log10(1.+redshift),inv_subtracted_median,log_obs_wave)

	wavelength_coverage_back = np.where(inv==999999)
	spectrum_back[wavelength_coverage_back[0]]=0
	inv_back[wavelength_coverage_back[0]]=0

	spectrum_back_median[wavelength_coverage_back[0]]=0
	inv_back_median[wavelength_coverage_back[0]]=0

	spectrum_back_substract_median[wavelength_coverage_back[0]]=-999
	inv_back_substract_median[wavelength_coverage_back[0]]=0
	
	spectrum_back_substract[wavelength_coverage_back[0]]=-999
	inv_back_substract[wavelength_coverage_back[0]]=0

	return log_obs_wave,spectrum_back,inv_back,spectrum_back_median,inv_back_median	,model_back, chi2_red

def shift_to_galaxy_rest_frame(log_obs_wave,spectrum,inv,redshift,log_rest_frame):
    wavelength_coverage = np.where(inv!=0)
    used_log_obs_wave = log_obs_wave[wavelength_coverage[0]]
    log_gal_wave = used_log_obs_wave-np.log10(1.+redshift)
    
    spectrum_restframe = data_info.spline_respec(log_gal_wave,spectrum[wavelength_coverage[0]],log_rest_frame)
    inv_restframe = data_info.spline_respec(log_gal_wave,inv[wavelength_coverage[0]],log_rest_frame)
    
    mask_region = np.where((log_rest_frame>max(log_gal_wave)) | (log_rest_frame<min(log_gal_wave)))
    spectrum_restframe[mask_region[0]]=0
    inv_restframe[mask_region[0]]=0 
    
    return log_rest_frame,spectrum_restframe,inv_restframe


# In[ ]:


MA = pf.open("DR16_absorber.fits")
catalog = MA[1].data
mjd = catalog['MJD']
plate = catalog['PLATE']
fiber = catalog['FIBER_ID']
QSO_redshift = catalog['Z_QSO']
ABS_redshift = catalog['Z_ABS']
W_0 = catalog['REST_EW_MGII_2796']

def to_4s(s):
    return str(s) if len(str(s))>4 else '0'*(4-len(str(s)))+str(s)

fiber = [str(to_4s(x)) for x in fiber]
mjd = [str(x) for x in mjd]
plate = [str(to_4s(x)) for x in plate]

from os.path import exists

spectra = np.zeros((len(catalog),2677))
inv = np.zeros((len(catalog),2677))#only need ivar, flux
has_data = np.zeros(len(catalog))


# In[8]:


import NMF
from NonnegMFPy import nmf


# In[18]:


len(catalog)


# In[193]:


spectra = np.zeros((len(catalog),2677))
inv = np.zeros((len(catalog),2677))#only need ivar, flux
has_data = np.zeros(len(catalog))

def gaussian(x,i,mu,sigma,A):
    return i-A*sigma*np.sqrt(2*np.pi)*np.exp(-(x-mu)**2/2/sigma**2)/(sigma*np.sqrt(2*np.pi))

def gaussian_EW(flux,range,i,mu,sigma,A):
    params,cov = curve_fit(gaussian,x[range],flux[range],p0=(i,mu,sigma,A),maxfev=5000)
    return -params[2]*params[3]*np.sqrt(np.pi*2)

def E_BV(Rest_EW,z): #expected
    return 0.017*(Rest_EW**1.6)*(1+z)**(-0.01)/1.55/(1+z)**1.2

def EW_DIB(E_BV): #expected
    return 1.22*E_BV**0.89

for i in range(len(catalog)):
    path_to_file = f'/Users/Chih-Yuan/Desktop/MA/spec-{plate[i]}-{mjd[i]}-{fiber[i]}.fits'
    if exists(path_to_file):
        has_data[i] = 1
        data = pf.open(f'/Users/Chih-Yuan/Desktop/MA/spec-{plate[i]}-{mjd[i]}-{fiber[i]}.fits')
        log10_wavelength = data[1].data['loglam']
        flux = data[1].data['flux']*gaussian(10**log10_wavelength,1,4200*(1+ABS_redshift[i]),7.74,EW_DIB(E_BV(W_0[i],ABS_redshift[i]))/np.sqrt(np.pi*2)/7.74)
        ivar = data[1].data['ivar']
        Q_redshift = QSO_redshift[i]
        A_redshift = ABS_redshift[i]
        new_flux = NMF_normalization_v2(log10_wavelength,flux,ivar,Q_redshift,NMF_basis)[3] #spectrum_back_median
        new_ivar = NMF_normalization_v2(log10_wavelength,flux,ivar,Q_redshift,NMF_basis)[4] #inv_back_median
        spectra[i] = shift_to_galaxy_rest_frame(log10_wavelength,new_flux,new_ivar,A_redshift,log_rest_frame)[1] #spectrum_restframe
        inv[i] = shift_to_galaxy_rest_frame(log10_wavelength,new_flux,new_ivar,A_redshift,log_rest_frame)[2] #inv_restframe
    else:
        has_data[i] = 0
        spectra[i] = np.zeros(2677)
        inv[i] = np.zeros(2677)


# In[199]:


def composite(flux,REW,min_REW,max_REW):
    spectrum = []
    search = np.where((REW < max_REW) & (REW > min_REW))[0]
    for i_spec in range(len(x)):
        spectrum.append(np.median(flux[search,i_spec]))
    return np.array(spectrum),len(search)

x = 10**log_rest_frame
found = np.array(spectra)[np.where(has_data == 1)]
Rest_EW = catalog[np.where(has_data == 1)]['REST_EW_MGII_2796']
y = composite(found,Rest_EW,0,1000)[0]
z = composite(found,Rest_EW,0,1000)[1]
plt.plot(x,y)
plt.xlim(4000,4400)
plt.ylim(0.99,1.002)
print('The number of spectra currently compiled = ',z)


# In[196]:


hdu1 = pf.PrimaryHDU(spectra)
hdu2 = pf.ImageHDU(inv)
hdu3 = pf.ImageHDU(has_data)
hdu = pf.HDUList([hdu1,hdu2,hdu3])
hdu.writeto('Synthetic_catalog_temp.fits',overwrite=True)


# In[ ]:





# In[138]:


import matplotlib
log_rest_frame = np.arange(np.log10(2700),np.log10(5000),1e-4)
x = 10**log_rest_frame

def single_gaussian(p,x):
    A, mu, sigma, zerop = p	
    return  -1.0 * A * np.exp(-(x - mu)**2 / (2.0 * sigma**2))+zerop

#Making composite spectra
def composite_bins(spectra):
    composite_spectra = np.zeros((len(bins)-1,len(spectra[1])))
    for i in range(len(bins)-1):
        spec = np.where((W_0 > bins[i]) & (W_0 < bins[i+1]))
        spectrum = []
        for i_spec in range(len(spectra[1])):
            spectrum.append(np.nanmedian(spectra[spec,i_spec]))
        composite_spectra[i] = spectrum
    return composite_spectra

def median_rew(REW):
    median = []
    for i in range(len(bins)-1):
        spec = np.where((REW > bins[i]) & (REW < bins[i+1]))
        median.append(np.median(REW[spec]))
    return median

def median_filter(spectra,region,long = 14,short = 7): #Aribitrary values I thought worked the best
    to_be_filtered = np.copy(spectra)
    to_be_filtered[region] = np.nan
    median_fil = []
    for i in range(len(spectra)):
        if np.isnan(to_be_filtered[i]) == True:
            window_length = long
            i = i + 1
        else:
            window_length = short
        spec = to_be_filtered[i-window_length:i+window_length]
        median_fil.append(np.nanmedian(spec[~np.isnan(spec)]))
    filtered_spectra = spectra/np.array(median_fil)
    return filtered_spectra

def bootstrap(times,REW,spectra,min_REW,max_REW): #O(N)
    '''
    times: the amount of iterations to run the bootstrap for
    spectra: the parent dataset used to run the bootstrap
    REW: the data containing the rest equivalent width of the MgII2796 line
    min_REW, max_REW: the minimum and maximum rest equivalent width of the dataset to be used
    '''
    final_spectra = np.zeros((times,len(spectra[1])))
    for i_times in range(times):
        search = np.where((REW > min_REW) & (REW < max_REW))[0]
        data_size = len(spectra[search])
        bootstrapped_spectra = np.zeros((data_size,len(spectra[1])))
        v = np.random.randint(0,data_size, size = data_size)
        bootstrapped_spectra[:,:] = spectra[search,:][v,:]
        composite_bootstrapped_spectra = []
        for i_spec in range(len(spectra[1])):
            composite_bootstrapped_spectra.append(np.nanmedian(bootstrapped_spectra[:,i_spec]))
        final_spectra[i_times] = median_filter(np.array(composite_bootstrapped_spectra),mgii)
    return final_spectra


DIB = np.where((x > 4410) & (x < 4450))[0]

def DIB_gaussian(x,A):
    return 1 - A * np.exp(- (x - 4429.0)**2 / 2 / 6.9**2)

from scipy.optimize import curve_fit
bound = [[0],[1]]
uncontaminated_region = np.where((x > ) & (x < 4349))[0]
def DIB_fitting(spectra,region): #2D spectra array
    params = []
    for i in range(len(spectra)):
        popt, pcov = curve_fit(DIB_gaussian,x[uncontaminated_region],spectra[i][uncontaminated_region],p0=(0.001),method='trf',bounds=bound, maxfev=10000)
        params.append(popt[0] * 6.9 * np.sqrt(2 * np.pi))
    REW_err = np.std(params)
    return(params,REW_err)


# In[176]:


data = pf.open('MA_catalog.fits')
synth_data = pf.open('Synthetic_catalog.fits')
MA = pf.open('DR16_absorber.fits')
spectra = data[0].data
has_data = data[2].data
has_data = np.where(has_data == 1)[0]
spectra = spectra[has_data]
catalog = MA[1].data

s_spectra = synth_data[0].data[np.where(synth_data[2].data == 1)[0]]

QSO_redshift = np.array(catalog['Z_QSO'])[has_data]
ABS_redshift = np.array(catalog['Z_ABS'])[has_data]
W_0 = np.array(catalog['REST_EW_MGII_2796'])[has_data]

s_QSO_redshift = np.array(catalog['Z_QSO'])[np.where(synth_data[2].data == 1)[0]]
s_ABS_redshift = np.array(catalog['Z_ABS'])[np.where(synth_data[2].data == 1)[0]]
s_W_0 = np.array(catalog['REST_EW_MGII_2796'])[np.where(synth_data[2].data == 1)[0]]


# In[177]:


bins = [0.5,1.0,2.0,3.0,4.0,np.amax(W_0)]
composite = composite_bins(spectra)
# s_composite = composite_bins(s_spectra)


# In[186]:


s_composite = np.zeros((len(bins)-1,len(s_spectra[1])))
for i in range(len(bins)-1):
    spec = np.where((s_W_0 > bins[i]) & (s_W_0 < bins[i+1]))
    spectrum = []
    for i_spec in range(len(s_spectra[1])):
        spectrum.append(np.median(s_spectra[spec,i_spec]))
    s_composite[i] = spectrum


# In[184]:


s_spectra = synth_data[0].data[np.where(synth_data[2].data == 1)[0]]


# In[175]:


def composite(flux,REW,min_REW,max_REW):
    spectrum = []
    search = np.where((REW < max_REW) & (REW > min_REW))[0]
    for i_spec in range(len(x)):
        spectrum.append(np.median(flux[search,i_spec]))
    return np.array(spectrum),len(search)

x = 10**log_rest_frame
found = np.array(s_spectra)[np.where(synth_data[2].data == 1)[0]]
Rest_EW = catalog[np.where(synth_data[2].data == 1)[0]]['REST_EW_MGII_2796']
y = composite(found,Rest_EW,0,1000)[0]
z = composite(found,Rest_EW,0,1000)[1]
plt.plot(x,y)
#plt.xlim(4100,4300)
#plt.ylim(0.99,1.001)
print('The number of spectra currently compiled = ',z)


# In[146]:


mgii = np.where((x > 2794) & (x < 2805))
median_filtered_composite = np.zeros_like(composite)
median_filtered_spectra = np.zeros_like(spectra)
for i in range(len(composite)):
    median_filtered_composite[i] = median_filter(composite[i],mgii)


# In[187]:


plt.plot(s_composite[0])


# In[191]:


fig,axs = plt.subplots(5,2,sharex='col',figsize=(15,12),gridspec_kw={'width_ratios': [2.5, 1]})
plt.subplots_adjust(wspace=0,hspace=0)
axs[0,0].set_xlim(2700,4700)
axs[0,1].set_xlim(3950,4050)
axs[0,1].set_ylim(0.99,1.005)
axs[1,1].set_ylim(0.99,1.005)
axs[2,1].set_ylim(0.99,1.005)
axs[3,1].set_ylim(0.95,1.02)
axs[4,1].set_ylim(0.93,1.04)
axs[3,1].set_xticks([4160,4180,4200,4220,4240])
for i in range(3):
    axs[i,1].set_yticks([0.9925,0.995,0.9975,1,1.0025,1.005],[0.9925,0.995,0.9975,1.0000,1.0025,1.005])
for i in range(5):
    axs[i,0].set_ylim(0,1.2)
    axs[i,0].tick_params(direction='in',labelleft=True,labelright=False)
    axs[i,0].set_yticks([0.2,0.4,0.6,0.8,1.0],[0.2,0.4,0.6,0.8,1.0])
    axs[i,1].tick_params(direction='in',left=False,right=True,labelleft=False,labelright=True)
#     axs[i,1].set_yticks([0.92,0.94,0.96,0.98,1.00,1.02],[0.92,0.94,0.96,0.98,1.00,1.02])
    axs[i,0].plot(x,median_filtered_composite[i],color='royalblue')
    axs[i,0].add_patch(matplotlib.patches.Rectangle((4380,0.95), 100, 0.1,linewidth=1,edgecolor='red',fill=False))
    axs[i,1].plot(x,s_composite[i],color='lightcoral',ds='steps')
    axs[i,1].plot(x,composite[i],color='royalblue',ds='steps')
# axs[4,1].plot(x, DIB_gaussian(x,*popt4) - 0.05,color='darkorchid')
# axs[3,1].plot(x, DIB_gaussian(x,*popt3) - 0.03,color='darkorchid')
# axs[2,1].plot(x, DIB_gaussian(x,*popt2) - 0.005,color='darkorchid')
# axs[1,1].plot(x, DIB_gaussian(x,*popt1) - 0.005,color='darkorchid')
# axs[0,1].plot(x, DIB_gaussian(x,*popt0) - 0.005,color='darkorchid')
# fig.add_subplot(111, frameon=False)

plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel("Wavelength [$\AA$]",fontsize=14)
# plt.text(-0.063,0.4, 'Normalized Flux', rotation=90, fontsize=14)
#fig.tight_layout()
fig.subplots_adjust(top=0.8)


# In[ ]:


iterations = 500
bs_spectra = np.zeros((len(bins)-1,iterations,len(spectra[1])))
start_time = time.time()
for i in range(len(bins)-1):
    bs_spectra[i] = bootstrap(iterations,W_0,spectra,bins[i],bins[i+1])
    print('Finished with bin',f'{i+1}')
end_time = time.time()
print('Final time:',end_time - start_time, 'seconds')


# In[ ]:


bs_spectra = np.zeros((len(bins)-1,iterations,len(s_spectra[1])))
start_time = time.time()
for i in range(len(bins)-1):
    bs_spectra[i] = bootstrap(iterations,s_W_0,spectra,bins[i],bins[i+1])
    print('Finished with bin',f'{i+1}')
end_time = time.time()
print('Final time:',end_time - start_time, 'seconds')


# In[ ]:


bound = [[0],[1]]
uncontaminated_region = np.where((x > 4180) & (x < 4220))[0]
orig_spectra = DIB_fitting(composite,uncontaminated_region)
synth_spectra = DIB_fitting(s_composite,uncontaminated_region)
print(orig_spectra_spectra_spectra[0])
print(synth_spectra[0])


# In[ ]:


bins = [0.5,1.0,2.0,3.0,4.0,np.amax(W_0)]
E_BV = np.linspace(0,0.09967488189730966,100)
E_BV_upper = np.linspace(0, 0.158156192364265,100)
E_BV_lower = np.linspace(0,0.06086189176356548,100)

upper = EWDIB(E_BV_upper)[1]
expected = EWDIB(E_BV)[0]
lower = EWDIB(E_BV_lower)[2]

fig = plt.figure(figsize=(9,7))
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()

x_arr = median_rew(W_0)
#these will have new values soon
# orig_median = [0.0029304892873043826, 0.0037817420057991028, 0.008817379643506847, 0.00010449825452783505, 5.050664458859056e-07]
# orig_err = [0.0016494017659980592,0.001335688458852618,0.004165176004070511,0.011629881089316297,0.025365427860354072]
# synth_median = [0.011296453772046798, 0.023939439887524172, 0.050158534226789224, 0.06748070351566694, 0.08414512820754671]
ax2.fill_between(E_BV,upper,lower,color='green',alpha=0.1)
ax2.plot(E_BV,expected,color='green',lw=0.2)
ax2.plot(np.linspace(0,4,100),0.03426337*np.linspace(0,4,100)**0.89,lw=0.5,ls='dashed',color='blue', zorder=15)
ax1.errorbar(x_arr,orig_median,yerr=orig_err,fmt = '.',color='blue',ecolor='blue',marker = '.',markersize=15,capsize=7,label='DIB Signal Strength in CGM',zorder=15)
ax1.errorbar(x_arr,synth_median,yerr=synth_err,fmt = '.',color='darkgreen',ecolor='darkgreen',marker = '.',markersize=15,capsize=7,label='Simulation Based on MW Trend (Lan et al.)',zorder=15)

ax1.set_xscale('log')
ax2.set_xscale('log')

