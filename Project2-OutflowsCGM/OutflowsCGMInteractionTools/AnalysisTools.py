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
X = 0.7 # Hydrogen Mass Fraction
gamma = 5/3. # Adiabatic Index
mu = 0.62 # Mean Molecular Weight

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
    def time(self): #In [Gyr] =[1 Billion Years]
        return self.f['Header'].attrs['Time']
    def number_of_particles(self):
        return self.f['Header'].attrs['NumPart_ThisFile']    
    def IDS(self,iPartType): #Particles I.D
        return (self.h*self.dic[('PartType%d'%iPartType,'ParticleIDs')])
    def masses(self,iPartType): #In Sun Masses [M ☉]
        return 1e10/self.h*self.dic[('PartType%d'%iPartType,'Masses')]
    def coords(self,iPartType): #In [kpc] =[3.086e+16 km]
        return (self.dic[('PartType%d'%iPartType,'Coordinates')] - self.center)/self.h*self.a
    def InternalEnergy(self,iPartType): 
        return (self.dic[('PartType%d'%iPartType,'InternalEnergy')])
    def velocities(self,iPartType): #In [km/s]
        return self.dic[('PartType%d'%iPartType,'Velocities')]*self.a**0.5 - self.center_velocity
    def Ts(self,iPartType): #In [K]
        epsilon = self.dic[('PartType%d'%iPartType,'InternalEnergy')][:] #Energy Per Unit Mass
        return (un.km**2/un.s**2 * cons.m_p / cons.k_B).to('K').value * (2./3* mu) * epsilon         
    def rs(self,iPartType): #In [kpc]
        return ((self.coords(iPartType)**2).sum(axis=1))**0.5    
    def rhos(self,iPartType): #In [gr/cm^3]
        return ((un.Msun/un.kpc**3).to('g/cm**3') *
                self.dic[('PartType%d'%iPartType,'Density')] * 1e10 /self.h * self.h**3 / self.a**3)
    def nHs(self): #In [cm^-3]
        return X*self.rhos() / cons.m_p.to('g').value         
    def cos_thetas(self):
        normed_coords = (self.coords().T / np.linalg.norm(self.coords(),axis=1)).T
        return np.dot(normed_coords,self.zvec)    
    def vrs(self):
        vs = self.vs() 
        coords = self.coords() 
        return (vs[:,0]*coords[:,0] + vs[:,1]*coords[:,1] + vs[:,2]*coords[:,2]) / self.rs()
    def neutralHydrogenAbundance(self,iPartType):
        return self.dic[('PartType%d'%iPartType,'NeutralHydrogenAbundance')]
    def SmoothingLength(self,iPartType):
        return self.dic[('PartType%d'%iPartType,'SmoothingLength')]*self.a
    def BoxSize(self):
        return self.f['Header'].attrs['BoxSize']
    def Redshift(self):
        return self.f['Header'].attrs['Redshift']

    
