# Basic Python Libraries:
import sys, os, time, importlib, glob, pdb
import matplotlib, pylab as pl, numpy as np

# Useful Scientific Libraries:
import h5py, astropy, scipy, scipy.stats

# Some Useful Shortcuts:
from astropy import units as un, constants as cons
from numpy import log10 as log

homedir = os.getenv("HOME")+'/'
projectdir = homedir + '/radial_to_rotating_flows/'
simname = 'm11_ad_25_lr_extralight_2_cooling_1'
datadir = projectdir + 'data/'+simname+'/'

# Some Constants:
X=0.7 # Hydrogen mass fraction
gamma=5/3. # Adiabatic index
mu=0.62 # Mean molecular weight

# Class For h5py Dic
class h5py_dic:
    """class for handling hdf5 files"""
    def __init__(self,fs,non_subhalo_inds=None):
        self.dic = {}
        self.fs = fs
    def __getitem__(self,k):
        if type(k)==type((None,)):
            particle,field = k
        else:
            particle = 'PartType0'
            field = k
        if (particle,field) not in self.dic:
            if particle in self.fs[0].keys():                 
                arr = self.fs[0][particle][field][...]                
                for f in self.fs[1:]:
                    new_arr = f[particle][field][...]                    
                    arr = np.append(arr, new_arr,axis=0)                
            else:
                arr = []
            self.dic[(particle,field)] = arr
            print('loaded %s, %s'%(particle, field))
        return self.dic[(particle,field)]

# Class For A Snapshot
class Snapshot:
    """interface class for a single simulation snapshot"""
    zvec = np.array([0.,0.,1.])
    def __init__(self,fn,center,center_velocity, h=0.7,a=1):
        self.h = h
        self.a = a
        self.center = center
        self.center_velocity = center_velocity
        self.f = h5py.File(fn)
        self.dic = h5py_dic([self.f])    
    def TimeInGyr(self): # [Gyr] =[1 Billion Years]
        return self.f['Header'].attrs['Time']
    def NumberOfParticles(self):
        return self.f['Header'].attrs['NumPart_ThisFile']
    def MassesInSunMasses(self,iPartType):
        return 1e10/self.h*self.dic[('PartType%d'%iPartType,'Masses')]
    def MassesInGrams(self,iPartType):
        return (1e10/self.h*self.dic[('PartType%d'%iPartType,'Masses')])*((un.Msun).to('g'))
    def CoordinatesInkpc(self,iPartType): # [kpc]=[3.086e+16 km]
        return (self.dic[('PartType%d'%iPartType,'Coordinates')]-self.center)/self.h*self.a
    def VelocitiesIn_km_s(self,iPartType): # In [km/s]
        return self.dic[('PartType%d'%iPartType,'Velocities')]*self.a**0.5-self.center_velocity
    def VelocitiesIn_kpc_s(self,iPartType): # In [kpc/s]
        return (self.dic[('PartType%d'%iPartType,'Velocities')]*self.a**0.5-self.center_velocity)*((un.km).to('kpc'))
    def RadiusesInkpc(self,iPartType): 
        return ((self.CoordinatesInkpc(iPartType)**2).sum(axis=1))**0.5  
    def TemperaturesInK(self): # For gas particles only
        epsilon = self.dic[('PartType0','InternalEnergy')][:] # Energy per unit mass
        return (un.km**2/un.s**2*cons.m_p/cons.k_B).to('K').value*(2./3*mu)*epsilon         
    def DensityIn_gr_cm_3(self,iPartType): # In [gr/cm^3]
        return ((un.Msun/un.kpc**3).to('g/cm**3')*self.dic[('PartType%d'%iPartType,'Density')]*1e10/self.h*self.h**3/self.a**3)
    def VolumeInkpc_3(self): # In [kpc^3], for gas particles only
        return un.Msun.to('g')*un.cm.to('kpc')**3*self.MassesInSunMasses(0)/self.DensityIn_gr_cm_3(0)
    def nHs(self,iPartType): #In [cm^-3]
        return X*self.DensityIn_gr_cm_3(iPartType)/cons.m_p.to('g').value  
    def cos_thetas(self):
        normed_coords = (self.CoordinatesInkpc(0).T/np.linalg.norm(self.CoordinatesInkpc(0),axis=1)).T
        return np.dot(normed_coords,self.zvec) 
    def vrs(self):
        vs=self.VelocitiesIn_km_s(0) 
        coords=self.CoordinatesInkpc(0) 
        return (vs[:,0]*coords[:,0]+vs[:,1]*coords[:,1]+vs[:,2]*coords[:,2])/self.RadiusesInkpc(0)
    def NeutralHydrogenAbundance(self,iPartType):
        return self.dic[('PartType%d'%iPartType,'NeutralHydrogenAbundance')]
    def SmoothingLength(self,iPartType):
        return self.dic[('PartType%d'%iPartType,'SmoothingLength')]*self.a
    def BoxSize(self):
        return self.f['Header'].attrs['BoxSize']
    def Redshift(self):
        return self.f['Header'].attrs['Redshift']
    def Metallicity(self,iPartType):
        return self.dic[('PartType%d'%iPartType,'Metallicity')]
    def JVectores(self,iPartType): # In [M_sun*kpc^2/s]
        vs=self.VelocitiesIn_kpc_s(iPartType)
        coords=self.CoordinatesInkpc(iPartType)
        return (np.array([coords[:,1]*vs[:,2]-coords[:,2]*vs[:,1],
                          coords[:,2]*vs[:,0]-coords[:,0]*vs[:,2],
                          coords[:,0]*vs[:,1]-coords[:,1]*vs[:,0]])*self.MassesInSunMasses(4)).T
    
