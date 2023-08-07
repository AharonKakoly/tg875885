 # Import of important libraries in Python:
import sys, os, time, importlib, glob, pdb
import matplotlib, pylab as pl, numpy as np
from numpy import log10 as log
import h5py, astropy, scipy, scipy.stats
from astropy import units as un, constants as cons
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import statistics as s
import random
from os import listdir
from os.path import isfile, join
import MyTools.AnalysisTools as l
import MyTools.HalosData as HD
import MyTools.Verdict as Verdict
import MyTools.Config as c
import math
import matplotlib.patches as mpatches
from astropy import units as un, constants as cons
import pylab as pl
import seaborn as sns
import palettable

# FIRE studio libraries
from OldFireStudio.firestudio.studios.gas_studio import GasStudio
from OldFireStudio.firestudio.studios.star_studio import StarStudio

# Important constants
X=0.7 # Hydrogen mass fraction
h=0.7 # Hubble parameter in units of 100 km/s Mpc^-1

# Useful functions
def ionFractions(z,Ts,nHs,tablefn,element='Mg',ionizationLevel=1):
    # Returns the MgII fraction
    F=h5py.File(tablefn,'r')
    logTs_hm2012=np.array(F[element].attrs['Temperature'])
    lognHs_hm2012=np.array(F[element].attrs['Parameter1'])
    zs_hm2012=np.array(F[element].attrs['Parameter2'])
    ind_z=np.searchsorted(zs_hm2012,z)
    log_f_tab=np.array(F[element])[ionizationLevel,:,ind_z,:]
    func=interpolate.RectBivariateSpline(lognHs_hm2012,logTs_hm2012,log_f_tab)
    res=func.ev(log(nHs),log(Ts))
    F.close()
    return (10**res)

# This part will create the direct for the analysis for specific simulation and specific snapnumber
StampedeSnapDirect='/scratch/projects/xsede/GalaxiesOnFIRE/metal_diffusion/'
SimulationsName=sys.argv[0] # 'm12b' $1
SimulationsNameWithRes=sys.argv[0]+'_res7100' # 'm12b_res7100' 
SnapshotNumber=sys.argv[3] # '050'$3

SnapDirect=StampedeSnapDirect+SimulationsNameWithRes+"/output" # '/scratch/projects/xsede/GalaxiesOnFIRE/metal_diffusion/m12b_res7100/output'
OutputFiles=os.listdir(SnapDirect) # All snapshots in the outputfolder
option1="snapshot_"+SnapshotNumber+".hdf5" # 'snapshot_050.hdf5'
option2="snapdir_"+SnapshotNumber # 'snapdir_050'
if(option1 in OutputFiles):
    FullPath=SnapDirect+"/"+option1 # '/scratch/projects/xsede/GalaxiesOnFIRE/metal_diffusion/m12b_res7100/output/snapshot_050.hdf5'
if(option2 in OutputFiles):
    FullPath=SnapDirect+"/"+option2 # '/scratch/projects/xsede/GalaxiesOnFIRE/metal_diffusion/m12b_res7100/output/snapdir_050'
        
# This block will create MgII projections plots for different velocities in range [-400,400]km/s for different snapshots
# Output: will save 2 files: .png image of the plot and .hdf5 file of the projection
plt.clf()
ListOfSnapshotsParts=[]
if(FullPath.endswith('.hdf5')):
    ListOfSnapshotsParts.append(FullPath)
else:
    ListOfFiles=os.listdir(FullPath)
    for i in ListOfFiles:
        if (i.endswith('.hdf5')):
            ListOfSnapshotsParts.append(FullPath+'/'+i)# Append Only hdf5 Files To The Snapshots Parts List

Snap1=l.Snapshot(ListOfSnapshotsParts[0],[0,0,0],[0,0,0])# SnapName, Not Real Center Coordinates, Not Real Center Velocity
z=Snap1.Redshift()
print("The redshift of the given halo is:",z)
Mvir=HD.HaloMvir(SimulationsName,z)
Rvir=HD.HaloRvir(SimulationsName,z)
CenterCoordinates=HD.HaloCenterCoordinates(SimulationsName,z)
CenterVelocity=HD.HaloCenterVelocity(SimulationsName,z)
afactor=HD.HaloaFactor(SimulationsName,z)

# This part intended to merge all the data from the different snapshot parts (if there are more than 1 part)
TotalNumberOfParticles=0
Coordinates=[]
Temperature=[]
r=[]
rs=[]
Density=[]
NeutralH=[]
GasMasses=[]
BoxesSizes=[]
SmoothingLengths=[]
Velocities=[]
Metallicity=[]
nHs=[]
    
for snapshotname in ListOfSnapshotsParts:
    Snap1=l.Snapshot(snapshotname,CenterCoordinates,CenterVelocity,a=afactor)# SnapName, Real Center Coordinates, Real Center Velocity

    NumberOfParticlesOfOneSnapshot=Snap1.number_of_particles()# 0-Gas,1-High Resolution Dark Matter,2-Dummy Particles,3-Dummy Particles,4-Stars,5-Black Holes
    TotalNumberOfParticles=TotalNumberOfParticles+NumberOfParticlesOfOneSnapshot[0]

    MassesOfOneSnapshotGasParticles=Snap1.masses(0)# The Masses Of The Gas Particles
    GasMasses.extend(MassesOfOneSnapshotGasParticles)

    CoordinatesOfOneSnapShot=Snap1.coords(0)# The Coordinates Of The Gas Particles
    Coordinates.extend(CoordinatesOfOneSnapShot)

    TemperatureOfOneSnapShot=Snap1.Ts()# The Temperature Of The Gas Particles
    Temperature.extend(TemperatureOfOneSnapShot)

    rOfOneSnapShot=Snap1.rs()# The r Of The Gas Particles
    r.extend(rOfOneSnapShot)
        
    DensityOfOneSnapShot=Snap1.rhos()# The Density Of The Gas Particles
    Density.extend(DensityOfOneSnapShot)

    NeutralHOfOneSnapshot=Snap1.neutralHydrogenAbundance(0)# Returns The Neutral Hydrogen Fraction
    NeutralH.extend(NeutralHOfOneSnapshot)
        
    SmoothingLengthOfOneSnapshot=Snap1.SmoothingLength()
    SmoothingLengths.extend(SmoothingLengthOfOneSnapshot)
        
    VelocitiesOfOneSnapshot=Snap1.vs()
    Velocities.extend(VelocitiesOfOneSnapshot)
        
    MetallicityOfOneSnapshot=Snap1.Metallicity()
    Metallicity.extend(MetallicityOfOneSnapshot)
        
    nHsOfOneSnapshot=Snap1.nHs()
    nHs.extend(nHsOfOneSnapshot)
    
ConversionTables='/home1/08289/tg875885/MgIIAnalysis/hm2012_hr.h5'
MgII_Mg_Fractions=ionFractions(z,Temperature,nHs,ConversionTables)
    
Coordinates=np.array(Coordinates)
Temperature=np.array(Temperature)
r=np.array(r)
Density=np.array(Density)
NeutralH=np.array(NeutralH)
GasMasses=np.array(GasMasses)
SmoothingLengths=np.array(SmoothingLengths)
Velocities=np.array(Velocities)
Metallicity=np.array(Metallicity)
nHs=np.array(nHs)
MgII_Mg_Fractions=np.array(MgII_Mg_Fractions)
    
Vzs=[]
for t in Velocities:
    Vzs.append(t[2])
Vzs=np.array(Vzs)
    
MgFractions=[]
for t in Metallicity:
    MgFractions.append(t[6])
MgFractions=np.array(MgFractions)
    
# Bins: [-400,-390] , [-390,-380] , ... , [380,390] , [390,400]
Bins=[[-400, -390], [-390, -380], [-380, -370], [-370, -360], [-360, -350], [-350, -340], [-340, -330], [-330, -320], [-320, -310], [-310, -300], [-300, -290], [-290, -280], [-280, -270], [-270, -260], [-260, -250], [-250, -240], [-240, -230], [-230, -220], [-220, -210], [-210, -200], [-200, -190], [-190, -180], [-180, -170], [-170, -160], [-160, -150], [-150, -140], [-140, -130], [-130, -120], [-120, -110], [-110, -100], [-100, -90], [-90, -80], [-80, -70], [-70, -60], [-60, -50], [-50, -40], [-40, -30], [-30, -20], [-20, -10], [-10, 0], [0, 10], [10, 20], [20, 30], [30, 40], [40, 50], [50, 60], [60, 70], [70, 80], [80, 90], [90, 100], [100, 110], [110, 120], [120, 130], [130, 140], [140, 150], [150, 160], [160, 170], [170, 180], [180, 190], [190, 200], [200, 210], [210, 220], [220, 230], [230, 240], [240, 250], [250, 260], [260, 270], [270, 280], [280, 290], [290, 300], [300, 310], [310, 320], [320, 330], [330, 340], [340, 350], [350, 360], [360, 370], [370, 380], [380, 390], [390, 400]]
VzBins=[[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
MgFractionsBins=[[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
MgII_Mg_FractionsBins=[[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
CoordinatesBins=[[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
GasMassesBins=[[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
SmoothingLengthsBins=[[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
      
for t in range(0,len(Vzs)):
    if(Vzs[t]>-400 and Vzs[t]<-390):
        VzBins[0].append(Vzs[t])
        MgFractionsBins[0].append(MgFractions[t])
        MgII_Mg_FractionsBins[0].append(MgII_Mg_Fractions[t])
        CoordinatesBins[0].append(Coordinates[t])
        GasMassesBins[0].append(GasMasses[t])
        SmoothingLengthsBins[0].append(SmoothingLengths[t])
    if(Vzs[t]>-390 and Vzs[t]<-380):
        VzBins[1].append(Vzs[t])
        MgFractionsBins[1].append(MgFractions[t])
        MgII_Mg_FractionsBins[1].append(MgII_Mg_Fractions[t])
        CoordinatesBins[1].append(Coordinates[t])
        GasMassesBins[1].append(GasMasses[t])
        SmoothingLengthsBins[1].append(SmoothingLengths[t])
    if(Vzs[t]>-380 and Vzs[t]<-370):
        VzBins[2].append(Vzs[t])
        MgFractionsBins[2].append(MgFractions[t])
        MgII_Mg_FractionsBins[2].append(MgII_Mg_Fractions[t])
        CoordinatesBins[2].append(Coordinates[t])
        GasMassesBins[2].append(GasMasses[t])
        SmoothingLengthsBins[2].append(SmoothingLengths[t])
    if(Vzs[t]>-370 and Vzs[t]<-360):
        VzBins[3].append(Vzs[t])
        MgFractionsBins[3].append(MgFractions[t])
        MgII_Mg_FractionsBins[3].append(MgII_Mg_Fractions[t])
        CoordinatesBins[3].append(Coordinates[t])
        GasMassesBins[3].append(GasMasses[t])
        SmoothingLengthsBins[3].append(SmoothingLengths[t])
    if(Vzs[t]>-360 and Vzs[t]<-350):
        VzBins[4].append(Vzs[t])
        MgFractionsBins[4].append(MgFractions[t])
        MgII_Mg_FractionsBins[4].append(MgII_Mg_Fractions[t])
        CoordinatesBins[4].append(Coordinates[t])
        GasMassesBins[4].append(GasMasses[t])
        SmoothingLengthsBins[4].append(SmoothingLengths[t])
    if(Vzs[t]>-350 and Vzs[t]<-340):
        VzBins[5].append(Vzs[t])
        MgFractionsBins[5].append(MgFractions[t])
        MgII_Mg_FractionsBins[5].append(MgII_Mg_Fractions[t])
        CoordinatesBins[5].append(Coordinates[t])
        GasMassesBins[5].append(GasMasses[t])
        SmoothingLengthsBins[5].append(SmoothingLengths[t])
    if(Vzs[t]>-340 and Vzs[t]<-330):
        VzBins[6].append(Vzs[t])
        MgFractionsBins[6].append(MgFractions[t])
        MgII_Mg_FractionsBins[6].append(MgII_Mg_Fractions[t])
        CoordinatesBins[6].append(Coordinates[t])
        GasMassesBins[6].append(GasMasses[t])
        SmoothingLengthsBins[6].append(SmoothingLengths[t])
    if(Vzs[t]>-330 and Vzs[t]<-320):
        VzBins[7].append(Vzs[t])
        MgFractionsBins[7].append(MgFractions[t])
        MgII_Mg_FractionsBins[7].append(MgII_Mg_Fractions[t])
        CoordinatesBins[7].append(Coordinates[t])
        GasMassesBins[7].append(GasMasses[t])
        SmoothingLengthsBins[7].append(SmoothingLengths[t])
    if(Vzs[t]>-320 and Vzs[t]<-310):
        VzBins[8].append(Vzs[t])
        MgFractionsBins[8].append(MgFractions[t])
        MgII_Mg_FractionsBins[8].append(MgII_Mg_Fractions[t])
        CoordinatesBins[8].append(Coordinates[t])
        GasMassesBins[8].append(GasMasses[t])
        SmoothingLengthsBins[8].append(SmoothingLengths[t])
    if(Vzs[t]>-310 and Vzs[t]<-300):
        VzBins[9].append(Vzs[t])
        MgFractionsBins[9].append(MgFractions[t])
        MgII_Mg_FractionsBins[9].append(MgII_Mg_Fractions[t])
        CoordinatesBins[9].append(Coordinates[t])
        GasMassesBins[9].append(GasMasses[t])
        SmoothingLengthsBins[9].append(SmoothingLengths[t])
    if(Vzs[t]>-300 and Vzs[t]<-290):
        VzBins[10].append(Vzs[t])
        MgFractionsBins[10].append(MgFractions[t])
        MgII_Mg_FractionsBins[10].append(MgII_Mg_Fractions[t])
        CoordinatesBins[10].append(Coordinates[t])
        GasMassesBins[10].append(GasMasses[t])
        SmoothingLengthsBins[10].append(SmoothingLengths[t])
    if(Vzs[t]>-290 and Vzs[t]<-280):
        VzBins[11].append(Vzs[t])
        MgFractionsBins[11].append(MgFractions[t])
        MgII_Mg_FractionsBins[11].append(MgII_Mg_Fractions[t])
        CoordinatesBins[11].append(Coordinates[t])
        GasMassesBins[11].append(GasMasses[t])
        SmoothingLengthsBins[11].append(SmoothingLengths[t])
    if(Vzs[t]>-280 and Vzs[t]<-270):
        VzBins[12].append(Vzs[t])
        MgFractionsBins[12].append(MgFractions[t])
        MgII_Mg_FractionsBins[12].append(MgII_Mg_Fractions[t])
        CoordinatesBins[12].append(Coordinates[t])
        GasMassesBins[12].append(GasMasses[t])
        SmoothingLengthsBins[12].append(SmoothingLengths[t])
    if(Vzs[t]>-270 and Vzs[t]<-260):
        VzBins[13].append(Vzs[t])
        MgFractionsBins[13].append(MgFractions[t])
        MgII_Mg_FractionsBins[13].append(MgII_Mg_Fractions[t])
        CoordinatesBins[13].append(Coordinates[t])
        GasMassesBins[13].append(GasMasses[t])
        SmoothingLengthsBins[13].append(SmoothingLengths[t])
    if(Vzs[t]>-260 and Vzs[t]<-250):
        VzBins[14].append(Vzs[t])
        MgFractionsBins[14].append(MgFractions[t])
        MgII_Mg_FractionsBins[14].append(MgII_Mg_Fractions[t])
        CoordinatesBins[14].append(Coordinates[t])
        GasMassesBins[14].append(GasMasses[t])
        SmoothingLengthsBins[14].append(SmoothingLengths[t])
    if(Vzs[t]>-250 and Vzs[t]<-240):
        VzBins[15].append(Vzs[t])
        MgFractionsBins[15].append(MgFractions[t])
        MgII_Mg_FractionsBins[15].append(MgII_Mg_Fractions[t])
        CoordinatesBins[15].append(Coordinates[t])
        GasMassesBins[15].append(GasMasses[t])
        SmoothingLengthsBins[15].append(SmoothingLengths[t])
    if(Vzs[t]>-240 and Vzs[t]<-230):
        VzBins[16].append(Vzs[t])
        MgFractionsBins[16].append(MgFractions[t])
        MgII_Mg_FractionsBins[16].append(MgII_Mg_Fractions[t])
        CoordinatesBins[16].append(Coordinates[t])
        GasMassesBins[16].append(GasMasses[t])
        SmoothingLengthsBins[16].append(SmoothingLengths[t])
    if(Vzs[t]>-230 and Vzs[t]<-220):
        VzBins[17].append(Vzs[t])
        MgFractionsBins[17].append(MgFractions[t])
        MgII_Mg_FractionsBins[17].append(MgII_Mg_Fractions[t])
        CoordinatesBins[17].append(Coordinates[t])
        GasMassesBins[17].append(GasMasses[t])
        SmoothingLengthsBins[17].append(SmoothingLengths[t])
    if(Vzs[t]>-220 and Vzs[t]<-210):
        VzBins[18].append(Vzs[t])
        MgFractionsBins[18].append(MgFractions[t])
        MgII_Mg_FractionsBins[18].append(MgII_Mg_Fractions[t])
        CoordinatesBins[18].append(Coordinates[t])
        GasMassesBins[18].append(GasMasses[t])
        SmoothingLengthsBins[18].append(SmoothingLengths[t])
    if(Vzs[t]>-210 and Vzs[t]<-200):
        VzBins[19].append(Vzs[t])
        MgFractionsBins[19].append(MgFractions[t])
        MgII_Mg_FractionsBins[19].append(MgII_Mg_Fractions[t])
        CoordinatesBins[19].append(Coordinates[t])
        GasMassesBins[19].append(GasMasses[t])
        SmoothingLengthsBins[19].append(SmoothingLengths[t])
    if(Vzs[t]>-200 and Vzs[t]<-190):
        VzBins[20].append(Vzs[t])
        MgFractionsBins[20].append(MgFractions[t])
        MgII_Mg_FractionsBins[20].append(MgII_Mg_Fractions[t])
        CoordinatesBins[20].append(Coordinates[t])
        GasMassesBins[20].append(GasMasses[t])
        SmoothingLengthsBins[20].append(SmoothingLengths[t])
    if(Vzs[t]>-190 and Vzs[t]<-180):
        VzBins[21].append(Vzs[t])
        MgFractionsBins[21].append(MgFractions[t])
        MgII_Mg_FractionsBins[21].append(MgII_Mg_Fractions[t])
        CoordinatesBins[21].append(Coordinates[t])
        GasMassesBins[21].append(GasMasses[t])
        SmoothingLengthsBins[21].append(SmoothingLengths[t])
    if(Vzs[t]>-180 and Vzs[t]<-170):
        VzBins[22].append(Vzs[t])
        MgFractionsBins[22].append(MgFractions[t])
        MgII_Mg_FractionsBins[22].append(MgII_Mg_Fractions[t])
        CoordinatesBins[22].append(Coordinates[t])
        GasMassesBins[22].append(GasMasses[t])
        SmoothingLengthsBins[22].append(SmoothingLengths[t])
    if(Vzs[t]>-170 and Vzs[t]<-160):
        VzBins[23].append(Vzs[t])
        MgFractionsBins[23].append(MgFractions[t])
        MgII_Mg_FractionsBins[23].append(MgII_Mg_Fractions[t])
        CoordinatesBins[23].append(Coordinates[t])
        GasMassesBins[23].append(GasMasses[t])
        SmoothingLengthsBins[23].append(SmoothingLengths[t])
    if(Vzs[t]>-160 and Vzs[t]<-150):
        VzBins[24].append(Vzs[t])
        MgFractionsBins[24].append(MgFractions[t])
        MgII_Mg_FractionsBins[24].append(MgII_Mg_Fractions[t])
        CoordinatesBins[24].append(Coordinates[t])
        GasMassesBins[24].append(GasMasses[t])
        SmoothingLengthsBins[24].append(SmoothingLengths[t])
    if(Vzs[t]>-150 and Vzs[t]<-140):
        VzBins[25].append(Vzs[t])
        MgFractionsBins[25].append(MgFractions[t])
        MgII_Mg_FractionsBins[25].append(MgII_Mg_Fractions[t])
        CoordinatesBins[25].append(Coordinates[t])
        GasMassesBins[25].append(GasMasses[t])
        SmoothingLengthsBins[25].append(SmoothingLengths[t])
    if(Vzs[t]>-140 and Vzs[t]<-130):
        VzBins[26].append(Vzs[t])
        MgFractionsBins[26].append(MgFractions[t])
        MgII_Mg_FractionsBins[26].append(MgII_Mg_Fractions[t])
        CoordinatesBins[26].append(Coordinates[t])
        GasMassesBins[26].append(GasMasses[t])
        SmoothingLengthsBins[26].append(SmoothingLengths[t])
    if(Vzs[t]>-130 and Vzs[t]<-120):
        VzBins[27].append(Vzs[t])
        MgFractionsBins[27].append(MgFractions[t])
        MgII_Mg_FractionsBins[27].append(MgII_Mg_Fractions[t])
        CoordinatesBins[27].append(Coordinates[t])
        GasMassesBins[27].append(GasMasses[t])
        SmoothingLengthsBins[27].append(SmoothingLengths[t])
    if(Vzs[t]>-120 and Vzs[t]<-110):
        VzBins[28].append(Vzs[t])
        MgFractionsBins[28].append(MgFractions[t])
        MgII_Mg_FractionsBins[28].append(MgII_Mg_Fractions[t])
        CoordinatesBins[28].append(Coordinates[t])
        GasMassesBins[28].append(GasMasses[t])
        SmoothingLengthsBins[28].append(SmoothingLengths[t])
    if(Vzs[t]>-110 and Vzs[t]<-100):
        VzBins[29].append(Vzs[t])
        MgFractionsBins[29].append(MgFractions[t])
        MgII_Mg_FractionsBins[29].append(MgII_Mg_Fractions[t])
        CoordinatesBins[29].append(Coordinates[t])
        GasMassesBins[29].append(GasMasses[t])
        SmoothingLengthsBins[29].append(SmoothingLengths[t])
    if(Vzs[t]>-100 and Vzs[t]<-90):
        VzBins[30].append(Vzs[t])
        MgFractionsBins[30].append(MgFractions[t])
        MgII_Mg_FractionsBins[30].append(MgII_Mg_Fractions[t])
        CoordinatesBins[30].append(Coordinates[t])
        GasMassesBins[30].append(GasMasses[t])
        SmoothingLengthsBins[30].append(SmoothingLengths[t])
    if(Vzs[t]>-90 and Vzs[t]<-80):
        VzBins[31].append(Vzs[t])
        MgFractionsBins[31].append(MgFractions[t])
        MgII_Mg_FractionsBins[31].append(MgII_Mg_Fractions[t])
        CoordinatesBins[31].append(Coordinates[t])
        GasMassesBins[31].append(GasMasses[t])
        SmoothingLengthsBins[31].append(SmoothingLengths[t])
    if(Vzs[t]>-80 and Vzs[t]<-70):
        VzBins[32].append(Vzs[t])
        MgFractionsBins[32].append(MgFractions[t])
        MgII_Mg_FractionsBins[32].append(MgII_Mg_Fractions[t])
        CoordinatesBins[32].append(Coordinates[t])
        GasMassesBins[32].append(GasMasses[t])
        SmoothingLengthsBins[32].append(SmoothingLengths[t])
    if(Vzs[t]>-70 and Vzs[t]<-60):
        VzBins[33].append(Vzs[t])
        MgFractionsBins[33].append(MgFractions[t])
        MgII_Mg_FractionsBins[33].append(MgII_Mg_Fractions[t])
        CoordinatesBins[33].append(Coordinates[t])
        GasMassesBins[33].append(GasMasses[t])
        SmoothingLengthsBins[33].append(SmoothingLengths[t])
    if(Vzs[t]>-60 and Vzs[t]<-50):
        VzBins[34].append(Vzs[t])
        MgFractionsBins[34].append(MgFractions[t])
        MgII_Mg_FractionsBins[34].append(MgII_Mg_Fractions[t])
        CoordinatesBins[34].append(Coordinates[t])
        GasMassesBins[34].append(GasMasses[t])
        SmoothingLengthsBins[34].append(SmoothingLengths[t])
    if(Vzs[t]>-50 and Vzs[t]<-40):
        VzBins[35].append(Vzs[t])
        MgFractionsBins[35].append(MgFractions[t])
        MgII_Mg_FractionsBins[35].append(MgII_Mg_Fractions[t])
        CoordinatesBins[35].append(Coordinates[t])
        GasMassesBins[35].append(GasMasses[t])
        SmoothingLengthsBins[35].append(SmoothingLengths[t])
    if(Vzs[t]>-40 and Vzs[t]<-30):
        VzBins[36].append(Vzs[t])
        MgFractionsBins[36].append(MgFractions[t])
        MgII_Mg_FractionsBins[36].append(MgII_Mg_Fractions[t])
        CoordinatesBins[36].append(Coordinates[t])
        GasMassesBins[36].append(GasMasses[t])
        SmoothingLengthsBins[36].append(SmoothingLengths[t])
    if(Vzs[t]>-30 and Vzs[t]<-20):           
        VzBins[37].append(Vzs[t])
        MgFractionsBins[37].append(MgFractions[t])
        MgII_Mg_FractionsBins[37].append(MgII_Mg_Fractions[t])
        CoordinatesBins[37].append(Coordinates[t])
        GasMassesBins[37].append(GasMasses[t])
        SmoothingLengthsBins[37].append(SmoothingLengths[t])
    if(Vzs[t]>-20 and Vzs[t]<-10):
        VzBins[38].append(Vzs[t])
        MgFractionsBins[38].append(MgFractions[t])
        MgII_Mg_FractionsBins[38].append(MgII_Mg_Fractions[t])
        CoordinatesBins[38].append(Coordinates[t])
        GasMassesBins[38].append(GasMasses[t])
        SmoothingLengthsBins[38].append(SmoothingLengths[t])
    if(Vzs[t]>-10 and Vzs[t]<0):
        VzBins[39].append(Vzs[t])
        MgFractionsBins[39].append(MgFractions[t])
        MgII_Mg_FractionsBins[39].append(MgII_Mg_Fractions[t])
        CoordinatesBins[39].append(Coordinates[t])
        GasMassesBins[39].append(GasMasses[t])
        SmoothingLengthsBins[39].append(SmoothingLengths[t])
    if(Vzs[t]>0 and Vzs[t]<10):
        VzBins[40].append(Vzs[t])
        MgFractionsBins[40].append(MgFractions[t])
        MgII_Mg_FractionsBins[40].append(MgII_Mg_Fractions[t])
        CoordinatesBins[40].append(Coordinates[t])
        GasMassesBins[40].append(GasMasses[t])
        SmoothingLengthsBins[40].append(SmoothingLengths[t])
    if(Vzs[t]>10 and Vzs[t]<20):
        VzBins[41].append(Vzs[t])
        MgFractionsBins[41].append(MgFractions[t])
        MgII_Mg_FractionsBins[41].append(MgII_Mg_Fractions[t])
        CoordinatesBins[41].append(Coordinates[t])
        GasMassesBins[41].append(GasMasses[t])
        SmoothingLengthsBins[41].append(SmoothingLengths[t])
    if(Vzs[t]>20 and Vzs[t]<30):
        VzBins[42].append(Vzs[t])
        MgFractionsBins[42].append(MgFractions[t])
        MgII_Mg_FractionsBins[42].append(MgII_Mg_Fractions[t])
        CoordinatesBins[42].append(Coordinates[t])
        GasMassesBins[42].append(GasMasses[t])
        SmoothingLengthsBins[42].append(SmoothingLengths[t])
    if(Vzs[t]>30 and Vzs[t]<40):
        VzBins[43].append(Vzs[t])
        MgFractionsBins[43].append(MgFractions[t])
        MgII_Mg_FractionsBins[43].append(MgII_Mg_Fractions[t])
        CoordinatesBins[43].append(Coordinates[t])
        GasMassesBins[43].append(GasMasses[t])
        SmoothingLengthsBins[43].append(SmoothingLengths[t])
    if(Vzs[t]>40 and Vzs[t]<50):
        VzBins[44].append(Vzs[t])
        MgFractionsBins[44].append(MgFractions[t])
        MgII_Mg_FractionsBins[44].append(MgII_Mg_Fractions[t])
        CoordinatesBins[44].append(Coordinates[t])
        GasMassesBins[44].append(GasMasses[t])
        SmoothingLengthsBins[44].append(SmoothingLengths[t])
    if(Vzs[t]>50 and Vzs[t]<60):
        VzBins[45].append(Vzs[t])
        MgFractionsBins[45].append(MgFractions[t])
        MgII_Mg_FractionsBins[45].append(MgII_Mg_Fractions[t])
        CoordinatesBins[45].append(Coordinates[t])
        GasMassesBins[45].append(GasMasses[t])
        SmoothingLengthsBins[45].append(SmoothingLengths[t])
    if(Vzs[t]>60 and Vzs[t]<70):
        VzBins[46].append(Vzs[t])
        MgFractionsBins[46].append(MgFractions[t])
        MgII_Mg_FractionsBins[46].append(MgII_Mg_Fractions[t])
        CoordinatesBins[46].append(Coordinates[t])
        GasMassesBins[46].append(GasMasses[t])
        SmoothingLengthsBins[46].append(SmoothingLengths[t])
    if(Vzs[t]>70 and Vzs[t]<80):
        VzBins[47].append(Vzs[t])
        MgFractionsBins[47].append(MgFractions[t])
        MgII_Mg_FractionsBins[47].append(MgII_Mg_Fractions[t])
        CoordinatesBins[47].append(Coordinates[t])
        GasMassesBins[47].append(GasMasses[t])
        SmoothingLengthsBins[47].append(SmoothingLengths[t])
    if(Vzs[t]>80 and Vzs[t]<90):
        VzBins[48].append(Vzs[t])
        MgFractionsBins[48].append(MgFractions[t])
        MgII_Mg_FractionsBins[48].append(MgII_Mg_Fractions[t])
        CoordinatesBins[48].append(Coordinates[t])
        GasMassesBins[48].append(GasMasses[t])
        SmoothingLengthsBins[48].append(SmoothingLengths[t])
    if(Vzs[t]>90 and Vzs[t]<100):
        VzBins[49].append(Vzs[t])
        MgFractionsBins[49].append(MgFractions[t])
        MgII_Mg_FractionsBins[49].append(MgII_Mg_Fractions[t])
        CoordinatesBins[49].append(Coordinates[t])
        GasMassesBins[49].append(GasMasses[t])
        SmoothingLengthsBins[49].append(SmoothingLengths[t])
    if(Vzs[t]>100 and Vzs[t]<110):
        VzBins[50].append(Vzs[t])
        MgFractionsBins[50].append(MgFractions[t])
        MgII_Mg_FractionsBins[50].append(MgII_Mg_Fractions[t])
        CoordinatesBins[50].append(Coordinates[t])
        GasMassesBins[50].append(GasMasses[t])
        SmoothingLengthsBins[50].append(SmoothingLengths[t])
    if(Vzs[t]>110 and Vzs[t]<120):
        VzBins[51].append(Vzs[t])
        MgFractionsBins[51].append(MgFractions[t])
        MgII_Mg_FractionsBins[51].append(MgII_Mg_Fractions[t])
        CoordinatesBins[51].append(Coordinates[t])
        GasMassesBins[51].append(GasMasses[t])
        SmoothingLengthsBins[51].append(SmoothingLengths[t])
    if(Vzs[t]>120 and Vzs[t]<130):
        VzBins[52].append(Vzs[t])
        MgFractionsBins[52].append(MgFractions[t])
        MgII_Mg_FractionsBins[52].append(MgII_Mg_Fractions[t])
        CoordinatesBins[52].append(Coordinates[t])
        GasMassesBins[52].append(GasMasses[t])
        SmoothingLengthsBins[52].append(SmoothingLengths[t])
    if(Vzs[t]>130 and Vzs[t]<140):
        VzBins[53].append(Vzs[t])
        MgFractionsBins[53].append(MgFractions[t])
        MgII_Mg_FractionsBins[53].append(MgII_Mg_Fractions[t])
        CoordinatesBins[53].append(Coordinates[t])
        GasMassesBins[53].append(GasMasses[t])
        SmoothingLengthsBins[53].append(SmoothingLengths[t])
    if(Vzs[t]>140 and Vzs[t]<150):
        VzBins[54].append(Vzs[t])
        MgFractionsBins[54].append(MgFractions[t])
        MgII_Mg_FractionsBins[54].append(MgII_Mg_Fractions[t])
        CoordinatesBins[54].append(Coordinates[t])
        GasMassesBins[54].append(GasMasses[t])
        SmoothingLengthsBins[54].append(SmoothingLengths[t])
    if(Vzs[t]>150 and Vzs[t]<160):
        VzBins[55].append(Vzs[t])
        MgFractionsBins[55].append(MgFractions[t])
        MgII_Mg_FractionsBins[55].append(MgII_Mg_Fractions[t])
        CoordinatesBins[55].append(Coordinates[t])
        GasMassesBins[55].append(GasMasses[t])
        SmoothingLengthsBins[55].append(SmoothingLengths[t])
    if(Vzs[t]>160 and Vzs[t]<170):
        VzBins[56].append(Vzs[t])
        MgFractionsBins[56].append(MgFractions[t])
        MgII_Mg_FractionsBins[56].append(MgII_Mg_Fractions[t])
        CoordinatesBins[56].append(Coordinates[t])
        GasMassesBins[56].append(GasMasses[t])
        SmoothingLengthsBins[56].append(SmoothingLengths[t])
    if(Vzs[t]>170 and Vzs[t]<180):
        VzBins[57].append(Vzs[t])
        MgFractionsBins[57].append(MgFractions[t])
        MgII_Mg_FractionsBins[57].append(MgII_Mg_Fractions[t])
        CoordinatesBins[57].append(Coordinates[t])
        GasMassesBins[57].append(GasMasses[t])
        SmoothingLengthsBins[57].append(SmoothingLengths[t])
    if(Vzs[t]>180 and Vzs[t]<190):
        VzBins[58].append(Vzs[t])
        MgFractionsBins[58].append(MgFractions[t])
        MgII_Mg_FractionsBins[58].append(MgII_Mg_Fractions[t])
        CoordinatesBins[58].append(Coordinates[t])
        GasMassesBins[58].append(GasMasses[t])
        SmoothingLengthsBins[58].append(SmoothingLengths[t])
    if(Vzs[t]>190 and Vzs[t]<200):
        VzBins[59].append(Vzs[t])
        MgFractionsBins[59].append(MgFractions[t])
        MgII_Mg_FractionsBins[59].append(MgII_Mg_Fractions[t])
        CoordinatesBins[59].append(Coordinates[t])
        GasMassesBins[59].append(GasMasses[t])
        SmoothingLengthsBins[59].append(SmoothingLengths[t])
    if(Vzs[t]>200 and Vzs[t]<210):
        VzBins[60].append(Vzs[t])
        MgFractionsBins[60].append(MgFractions[t])
        MgII_Mg_FractionsBins[60].append(MgII_Mg_Fractions[t])
        CoordinatesBins[60].append(Coordinates[t])
        GasMassesBins[60].append(GasMasses[t])
        SmoothingLengthsBins[60].append(SmoothingLengths[t])
    if(Vzs[t]>210 and Vzs[t]<220):
        VzBins[61].append(Vzs[t])
        MgFractionsBins[61].append(MgFractions[t])
        MgII_Mg_FractionsBins[61].append(MgII_Mg_Fractions[t])
        CoordinatesBins[61].append(Coordinates[t])
        GasMassesBins[61].append(GasMasses[t])
        SmoothingLengthsBins[61].append(SmoothingLengths[t])
    if(Vzs[t]>220 and Vzs[t]<230):
        VzBins[62].append(Vzs[t])
        MgFractionsBins[62].append(MgFractions[t])
        MgII_Mg_FractionsBins[62].append(MgII_Mg_Fractions[t])
        CoordinatesBins[62].append(Coordinates[t])
        GasMassesBins[62].append(GasMasses[t])
        SmoothingLengthsBins[62].append(SmoothingLengths[t])
    if(Vzs[t]>230 and Vzs[t]<240):
        VzBins[63].append(Vzs[t])
        MgFractionsBins[63].append(MgFractions[t])
        MgII_Mg_FractionsBins[63].append(MgII_Mg_Fractions[t])
        CoordinatesBins[63].append(Coordinates[t])
        GasMassesBins[63].append(GasMasses[t])
        SmoothingLengthsBins[63].append(SmoothingLengths[t])
    if(Vzs[t]>240 and Vzs[t]<250):
        VzBins[64].append(Vzs[t])
        MgFractionsBins[64].append(MgFractions[t])
        MgII_Mg_FractionsBins[64].append(MgII_Mg_Fractions[t])
        CoordinatesBins[64].append(Coordinates[t])
        GasMassesBins[64].append(GasMasses[t])
        SmoothingLengthsBins[64].append(SmoothingLengths[t])
    if(Vzs[t]>250 and Vzs[t]<260):
        VzBins[65].append(Vzs[t])
        MgFractionsBins[65].append(MgFractions[t])
        MgII_Mg_FractionsBins[65].append(MgII_Mg_Fractions[t])
        CoordinatesBins[65].append(Coordinates[t])
        GasMassesBins[65].append(GasMasses[t])
        SmoothingLengthsBins[65].append(SmoothingLengths[t])
    if(Vzs[t]>260 and Vzs[t]<270):
        VzBins[66].append(Vzs[t])
        MgFractionsBins[66].append(MgFractions[t])
        MgII_Mg_FractionsBins[66].append(MgII_Mg_Fractions[t])
        CoordinatesBins[66].append(Coordinates[t])
        GasMassesBins[66].append(GasMasses[t])
        SmoothingLengthsBins[66].append(SmoothingLengths[t])
    if(Vzs[t]>270 and Vzs[t]<280):
        VzBins[67].append(Vzs[t])
        MgFractionsBins[67].append(MgFractions[t])
        MgII_Mg_FractionsBins[67].append(MgII_Mg_Fractions[t])
        CoordinatesBins[67].append(Coordinates[t])
        GasMassesBins[67].append(GasMasses[t])
        SmoothingLengthsBins[67].append(SmoothingLengths[t])
    if(Vzs[t]>280 and Vzs[t]<290):
        VzBins[68].append(Vzs[t])
        MgFractionsBins[68].append(MgFractions[t])
        MgII_Mg_FractionsBins[68].append(MgII_Mg_Fractions[t])
        CoordinatesBins[68].append(Coordinates[t])
        GasMassesBins[68].append(GasMasses[t])
        SmoothingLengthsBins[68].append(SmoothingLengths[t])
    if(Vzs[t]>290 and Vzs[t]<300):
        VzBins[69].append(Vzs[t])
        MgFractionsBins[69].append(MgFractions[t])
        MgII_Mg_FractionsBins[69].append(MgII_Mg_Fractions[t])
        CoordinatesBins[69].append(Coordinates[t])
        GasMassesBins[69].append(GasMasses[t])
        SmoothingLengthsBins[69].append(SmoothingLengths[t])
    if(Vzs[t]>300 and Vzs[t]<310):
        VzBins[70].append(Vzs[t])
        MgFractionsBins[70].append(MgFractions[t])
        MgII_Mg_FractionsBins[70].append(MgII_Mg_Fractions[t])
        CoordinatesBins[70].append(Coordinates[t])
        GasMassesBins[70].append(GasMasses[t])
        SmoothingLengthsBins[70].append(SmoothingLengths[t])
    if(Vzs[t]>310 and Vzs[t]<320):
        VzBins[71].append(Vzs[t])
        MgFractionsBins[71].append(MgFractions[t])
        MgII_Mg_FractionsBins[71].append(MgII_Mg_Fractions[t])
        CoordinatesBins[71].append(Coordinates[t])
        GasMassesBins[71].append(GasMasses[t])
        SmoothingLengthsBins[71].append(SmoothingLengths[t])
    if(Vzs[t]>320 and Vzs[t]<330):
        VzBins[72].append(Vzs[t])
        MgFractionsBins[72].append(MgFractions[t])
        MgII_Mg_FractionsBins[72].append(MgII_Mg_Fractions[t])
        CoordinatesBins[72].append(Coordinates[t])
        GasMassesBins[72].append(GasMasses[t])
        SmoothingLengthsBins[72].append(SmoothingLengths[t])
    if(Vzs[t]>330 and Vzs[t]<340):
        VzBins[73].append(Vzs[t])
        MgFractionsBins[73].append(MgFractions[t])
        MgII_Mg_FractionsBins[73].append(MgII_Mg_Fractions[t])
        CoordinatesBins[73].append(Coordinates[t])
        GasMassesBins[73].append(GasMasses[t])
        SmoothingLengthsBins[73].append(SmoothingLengths[t])
    if(Vzs[t]>340 and Vzs[t]<350):
        VzBins[74].append(Vzs[t])
        MgFractionsBins[74].append(MgFractions[t])
        MgII_Mg_FractionsBins[74].append(MgII_Mg_Fractions[t])
        CoordinatesBins[74].append(Coordinates[t])
        GasMassesBins[74].append(GasMasses[t])
        SmoothingLengthsBins[74].append(SmoothingLengths[t])
    if(Vzs[t]>350 and Vzs[t]<360):
        VzBins[75].append(Vzs[t])
        MgFractionsBins[75].append(MgFractions[t])
        MgII_Mg_FractionsBins[75].append(MgII_Mg_Fractions[t])
        CoordinatesBins[75].append(Coordinates[t])
        GasMassesBins[75].append(GasMasses[t])
        SmoothingLengthsBins[75].append(SmoothingLengths[t])
    if(Vzs[t]>360 and Vzs[t]<370):
        VzBins[76].append(Vzs[t])
        MgFractionsBins[76].append(MgFractions[t])
        MgII_Mg_FractionsBins[76].append(MgII_Mg_Fractions[t])
        CoordinatesBins[76].append(Coordinates[t])
        GasMassesBins[76].append(GasMasses[t])
        SmoothingLengthsBins[76].append(SmoothingLengths[t])
    if(Vzs[t]>370 and Vzs[t]<380):
        VzBins[77].append(Vzs[t])
        MgFractionsBins[77].append(MgFractions[t])
        MgII_Mg_FractionsBins[77].append(MgII_Mg_Fractions[t])
        CoordinatesBins[77].append(Coordinates[t])
        GasMassesBins[77].append(GasMasses[t])
        SmoothingLengthsBins[77].append(SmoothingLengths[t])
    if(Vzs[t]>380 and Vzs[t]<390):
        VzBins[78].append(Vzs[t])
        MgFractionsBins[78].append(MgFractions[t])
        MgII_Mg_FractionsBins[78].append(MgII_Mg_Fractions[t])
        CoordinatesBins[78].append(Coordinates[t])
        GasMassesBins[78].append(GasMasses[t])
        SmoothingLengthsBins[78].append(SmoothingLengths[t])
    if(Vzs[t]>390 and Vzs[t]<400):
        VzBins[79].append(Vzs[t])
        MgFractionsBins[79].append(MgFractions[t])
        MgII_Mg_FractionsBins[79].append(MgII_Mg_Fractions[t])
        CoordinatesBins[79].append(Coordinates[t])
        GasMassesBins[79].append(GasMasses[t])
        SmoothingLengthsBins[79].append(SmoothingLengths[t])
    
VzBinsNp=[]
MgFractionsBinsNp=[]
MgII_Mg_FractionsBinsNp=[]
CoordinatesBinsNp=[]
GasMassesBinsNp=[]
SmoothingLengthsBinsNp=[]
    
for t in range(0,80):
    VzBinsNp.append(np.array(VzBins[t]))
    MgFractionsBinsNp.append(np.array(MgFractionsBins[t]))
    MgII_Mg_FractionsBinsNp.append(np.array(MgII_Mg_FractionsBins[t]))
    CoordinatesBinsNp.append(np.array(CoordinatesBins[t]))
    GasMassesBinsNp.append(np.array(GasMassesBins[t]))
    SmoothingLengthsBinsNp.append(np.array(SmoothingLengthsBins[t]))
        
VzBinsNp=np.array(VzBinsNp)
MgFractionsBinsNp=np.array(MgFractionsBinsNp)
MgII_Mg_FractionsBinsNp=np.array(MgII_Mg_FractionsBinsNp)
CoordinatesBinsNp=np.array(CoordinatesBinsNp)
GasMassesBinsNp=np.array(GasMassesBinsNp)
SmoothingLengthsBinsNp=np.array(SmoothingLengthsBinsNp)
    
print("The total number of gas particles is:",TotalNumberOfParticles)
print("r min is:",r.min())
print("r max is:",r.max())
          
snapdir=SnapDirect
iSnapshot=SnapshotNumber
r_max=200
z_width=200
filedir='/home1/08289/tg875885/radial_to_rotating_flows/Aharon/'
SnapDirect=SnapDirect.replace("/","-")
SnapDirect=SnapDirect.replace(".hdf5","")
projection_output_filename=SnapDirect
    
for t in range(0,80):
    # Create input dictionary for FIRE studio from snapshot and snapdict
    studiodic = {}
    studiodic['Coordinates']    = CoordinatesBinsNp[t]
    studiodic['Masses']         = GasMassesBinsNp[t]*MgFractionsBinsNp[t]*MgII_Mg_FractionsBinsNp[t]
    studiodic['BoxSize']        = Snap1.BoxSize()
    studiodic['SmoothingLength']= SmoothingLengthsBinsNp[t] 

    # Create N_HI projection 
    mystudio=GasStudio(
                snapdir, 
                snapnum=int(iSnapshot),
                snapdict=studiodic,
                datadir=filedir,
                frame_half_width=r_max,
                frame_depth=z_width,
                quantity_name='Masses',
                take_log_of_quantity=False, 
                galaxy_extractor=False,
                pixels=1200,
                single_image='Density',
                overwrite=True,
                savefig=False,      
                use_hsml=True,
                intermediate_file_name=projection_output_filename,
                )

    # This part intended to plot the HI projection with Firestudio and save .png and .hdf5 files
    # .hdf5 file will contain the following things: 
    # A grid of column densities, The side length/the positions the column densities are at, The redshift, The simulation halo    mass (its viral mass),
    # The simulation name, The path to the file name should contain reference to the simulation itself and to the snapshot number. 
    fig,ax=plt.subplots()
    NHImap,_=mystudio.projectImage([])
    NHImap+=log((X*un.Msun*un.pc**-2/(24*cons.m_p)/1e10*h).to('cm**-2').value) # Units fix
    NHImap=NHImap.T
    Xs=np.linspace(-r_max,r_max,NHImap.shape[0])
    Ys=np.linspace(-r_max,r_max,NHImap.shape[1])

    # This part prepares the graph
    plt.rcParams['mathtext.fontset']='dejavuserif'
    plt.pcolormesh(Xs,Ys,NHImap,cmap='viridis')#,norm=pl.Normalize(6))
    cbar=pl.colorbar()
    plt.xlabel("X [kpc]",fontsize=12,fontname="serif")
    plt.ylabel("Y [kpc]",fontsize=12,fontname="serif")
    title="z= "+str(round(z,2))+"   ;   M= "+str(Mvir)+r" $M_{☉}$"+"\n"+"Bin: "+str(Bins[t])+" "+r"$\frac{km}{s}$"
    plt.title(title)
    a_circle=plt.Circle((0,0),Rvir,fill=False,color='black') # Draw white circle in Rvir of the halo
    ax.add_patch(a_circle) # Adds the circle to the plot
    plt.savefig('/home1/08289/tg875885/radial_to_rotating_flows/Aharon/OutpotsMgII/'+FullPath+'_Bins_'+str(Bins[t])+'.png')
    np.savez_compressed('/home1/08289/tg875885/radial_to_rotating_flows/Aharon/OutpotsMgII/'+FullPath+'_Bins_'+str(Bins[t])+'_NMgII'+'_Vz='+str((Bins[t][0]+Bins[t][1])/2)+'.npz',NHImap,dtype=float)

    plt.show()
    plt.clf()
    plt.cla()
    plt.close()
print("DONE")
####################################################################################################################################################