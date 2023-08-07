import numpy as np
import math
def FindClosest(lst,K): 
    return lst[min(range(len(lst)),key=lambda i: abs(lst[i]-K))]
# DataOfHalos=[[ID(1)  hostHalo(2)	numSubStruct(3)	Mvir(4)	npart(5)	Xc(6)	Yc(7)	Zc(8)	VXc(9)	VYc(10)	VZc(11)	Rvir(12)
# Rmax(13)	r2(14)	mbp_offset(15)	com_offset(16)	Vmax(17)	v_esc(18)	sigV(19)	lambda(20)	lambdaE(21)	Lx(22)	Ly(23)
# Lz(24)	b(25)	c(26)	Eax(27)	Eay(28)	Eaz(29)	Ebx(30)	Eby(31)	Ebz(32)	Ecx(33)	Ecy(34)	Ecz(35)	ovdens(36)	nbins(37)	fMhires(38)	
# Ekin(39)	Epot(40)	SurfP(41)	Phi0(42)	cNFW(43)	mbp_Vx(44)	mbp_Vy(45)	mbp_Vz(46)	mbp_x(47)	mbp_y(48)	mbp_z(49)	
# com_x(50)	com_y(51)	com_z(52)	n_gas(53)	M_gas(54)	lambda_gas(55)	lambdaE_gas(56)	Lx_gas(57)	Ly_gas(58)	Lz_gas(59)	b_gas(60)
# c_gas(61)	Eax_gas(62)	Eay_gas(63)	Eaz_gas(64)	Ebx_gas(65)	Eby_gas(66)	Ebz_gas(67)	Ecx_gas(68)	Ecy_gas(69)	Ecz_gas(70)	Ekin_gas(71	
# Epot_gas(72)	n_star(73)	M_star(74)	lambda_star(75)	lambdaE_star(76)	Lx_star(77)	Ly_star(78)	Lz_star(79)	b_star(80)	c_star(81)
# Eax_star(82)	Eay_star(83)	Eaz_star(84)	Ebx_star(85)	Eby_star(86)	Ebz_star(87)	Ecx_star(88)	Ecy_star(89)	
# Ecz_star(90)	Ekin_star(91)	Epot_star(92)],...]

m11d='/home1/08289/tg875885/radial_to_rotating_flows/Aharon/HalosData/m11d.dat'
m11h='/home1/08289/tg875885/radial_to_rotating_flows/Aharon/HalosData/m11h.dat'
m11i='/home1/08289/tg875885/radial_to_rotating_flows/Aharon/HalosData/m11i.dat'
m12b='/home1/08289/tg875885/radial_to_rotating_flows/Aharon/HalosData/m12b.dat'
m12c='/home1/08289/tg875885/radial_to_rotating_flows/Aharon/HalosData/m12c.dat'
m12f='/home1/08289/tg875885/radial_to_rotating_flows/Aharon/HalosData/m12f.dat'
m12i='/home1/08289/tg875885/radial_to_rotating_flows/Aharon/HalosData/m12i.dat'
m12r='/home1/08289/tg875885/radial_to_rotating_flows/Aharon/HalosData/m12r.dat'
m12w='/home1/08289/tg875885/radial_to_rotating_flows/Aharon/HalosData/m12w.dat'

def HaloCenterCoordinates(Simultaion,z):
    if(Simultaion=='m11d'):
        A=m11d
    if(Simultaion=='m11h'):
        A=m11h
    if(Simultaion=='m11i'):
        A=m11i
    if(Simultaion=='m12b'):
        A=m12b
    if(Simultaion=='m12c'):
        A=m12c
    if(Simultaion=='m12f'):
        A=m12f
    if(Simultaion=='m12i'):
        A=m12i
    if(Simultaion=='m12r'):
        A=m12r
    if(Simultaion=='m12w'):
        A=m12w
    DataOfHalos=np.loadtxt(A)
    ListOfRedshifts=[]
    for i in range (0,len(DataOfHalos)):
        ListOfRedshifts.append(DataOfHalos[i][1])
    ListOfRedshifts=np.array(ListOfRedshifts)
    R=FindClosest(ListOfRedshifts,z)
    print("The closet redshift to the given redshift is:",R)
    IndexOfRedshift=np.where(DataOfHalos==R)[0][0]
    RelevantHaloData=DataOfHalos[IndexOfRedshift]
    Xc=RelevantHaloData[7]
    Yc=RelevantHaloData[8]
    Zc=RelevantHaloData[9]
    c=[Xc,Yc,Zc]
    print("The center coordinates of the given halo is: c=",c)
    return c
    
def HaloCenterVelocity(Simultaion,z):   
    if(Simultaion=='m11d'):
        A=m11d
    if(Simultaion=='m11h'):
        A=m11h
    if(Simultaion=='m11i'):
        A=m11i
    if(Simultaion=='m12b'):
        A=m12b
    if(Simultaion=='m12c'):
        A=m12c
    if(Simultaion=='m12f'):
        A=m12f
    if(Simultaion=='m12i'):
        A=m12i
    if(Simultaion=='m12r'):
        A=m12r
    if(Simultaion=='m12w'):
        A=m12w
    DataOfHalos=np.loadtxt(A)
    ListOfRedshifts=[]
    for i in range (0,len(DataOfHalos)):
        ListOfRedshifts.append(DataOfHalos[i][1])
    ListOfRedshifts=np.array(ListOfRedshifts)
    R=FindClosest(ListOfRedshifts,z)
    IndexOfRedshift=np.where(DataOfHalos==R)[0][0]
    RelevantHaloData=DataOfHalos[IndexOfRedshift]
    VXc=RelevantHaloData[10]
    VYc=RelevantHaloData[11]
    VZc=RelevantHaloData[12]
    Vc=[VXc,VYc,VZc]
    print("The velocity of the center of the given halo is: Vc=",Vc)
    return Vc
                   
def HaloRvir(Simultaion,z):
    if(Simultaion=='m11d'):
        A=m11d
    if(Simultaion=='m11h'):
        A=m11h
    if(Simultaion=='m11i'):
        A=m11i
    if(Simultaion=='m12b'):
        A=m12b
    if(Simultaion=='m12c'):
        A=m12c
    if(Simultaion=='m12f'):
        A=m12f
    if(Simultaion=='m12i'):
        A=m12i
    if(Simultaion=='m12r'):
        A=m12r
    if(Simultaion=='m12w'):
        A=m12w
    DataOfHalos=np.loadtxt(A)
    ListOfRedshifts=[]
    for i in range (0,len(DataOfHalos)):
        ListOfRedshifts.append(DataOfHalos[i][1])
    ListOfRedshifts=np.array(ListOfRedshifts)
    R=FindClosest(ListOfRedshifts,z)
    IndexOfRedshift=np.where(DataOfHalos==R)[0][0]
    RelevantHaloData=DataOfHalos[IndexOfRedshift]
    Rvir=RelevantHaloData[13]/(1+z)
    print("R_vir of the given halo is:",Rvir) 
    return Rvir
          
def HaloaFactor(Simultaion,z):
    if(Simultaion=='m11d'):
        A=m11d
    if(Simultaion=='m11h'):
        A=m11h
    if(Simultaion=='m11i'):
        A=m11i
    if(Simultaion=='m12b'):
        A=m12b
    if(Simultaion=='m12c'):
        A=m12c
    if(Simultaion=='m12f'):
        A=m12f
    if(Simultaion=='m12i'):
        A=m12i
    if(Simultaion=='m12r'):
        A=m12r
    if(Simultaion=='m12w'):
        A=m12w
    DataOfHalos=np.loadtxt(A)
    ListOfRedshifts=[]
    for i in range (0,len(DataOfHalos)):
        ListOfRedshifts.append(DataOfHalos[i][1])
    ListOfRedshifts=np.array(ListOfRedshifts)
    R=FindClosest(ListOfRedshifts,z)
    a=(1+R)**-1
    print("a factor of the given halo is:",a)
    return a

def HaloMvir(Simultaion,z):
    if(Simultaion=='m11d'):
        A=m11d
    if(Simultaion=='m11h'):
        A=m11h
    if(Simultaion=='m11i'):
        A=m11i
    if(Simultaion=='m12b'):
        A=m12b
    if(Simultaion=='m12c'):
        A=m12c
    if(Simultaion=='m12f'):
        A=m12f
    if(Simultaion=='m12i'):
        A=m12i
    if(Simultaion=='m12r'):
        A=m12r
    if(Simultaion=='m12w'):
        A=m12w
    DataOfHalos=np.loadtxt(A)
    ListOfRedshifts=[]
    for i in range (0,len(DataOfHalos)):
        ListOfRedshifts.append(DataOfHalos[i][1])
    ListOfRedshifts=np.array(ListOfRedshifts)
    R=FindClosest(ListOfRedshifts,z)      
    IndexOfRedshift=np.where(DataOfHalos==R)[0][0]
    RelevantHaloData=DataOfHalos[IndexOfRedshift]
    Mvir=RelevantHaloData[5] 
    SignificantDigits=2
    RoundedMass=round(Mvir,SignificantDigits-int(math.floor(math.log10(abs(Mvir))))-1)
    SientificNotation="{:.2e}".format(RoundedMass)
    print("M_vir of the given halo is:",SientificNotation)
    return SientificNotation

def Snapnum(Simultaion,z):
    if(Simultaion=='m11d'):
        A=m11d
    if(Simultaion=='m11h'):
        A=m11h
    if(Simultaion=='m11i'):
        A=m11i
    if(Simultaion=='m12b'):
        A=m12b
    if(Simultaion=='m12c'):
        A=m12c
    if(Simultaion=='m12f'):
        A=m12f
    if(Simultaion=='m12i'):
        A=m12i
    if(Simultaion=='m12r'):
        A=m12r
    if(Simultaion=='m12w'):
        A=m12w
    DataOfHalos=np.loadtxt(A)
    ListOfRedshifts=[]
    for i in range (0,len(DataOfHalos)):
        ListOfRedshifts.append(DataOfHalos[i][1])
    ListOfRedshifts=np.array(ListOfRedshifts)
    R=FindClosest(ListOfRedshifts,z)      
    IndexOfRedshift=np.where(DataOfHalos==R)[0][0]
    RelevantHaloData=DataOfHalos[IndexOfRedshift]
    Snapnum=RelevantHaloData[0] 
    return Snapnum