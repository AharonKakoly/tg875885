#this notebook creates an HI and H2 projection of a FIRE snapshot
#the H2 calculation relies on 
#1) SKIRT RT calculation for the snapshot to determine the LW flux at each particle
#2) FIRE_to_SKIRT run to find the AMR cell in the SKIRT output corresponding to each FIRE particle[
#3) use of the NH2_tools that are just some tabulated NH2 values for a given density, LW flux, total column, and metallicity

#load some libraries and set directories for the data
import numpy as np
import matplotlib
import time, importlib, sys
import pylab as pl, numpy as np, glob, pdb, scipy, scipy.stats
from numpy import log10 as log
from astropy.cosmology import Planck15 as cosmo
from astropy import units as un, constants as cons
# FIRE studio libraries
sys.path.append('./FIRE_studio')
import abg_python
import abg_python.snapshot_utils
import firestudio
from firestudio.studios.gas_studio import GasStudio
import FIRE_files as ff
import simulationFiles
import os
import h5py

# halo_name = 'A4' #halo name, can be either A1, A2, A4, or A8
# iSnapshot = 88 #snapshot number, can be either 67, 88, 120 or 172, corresponding (I think) to redshifts of 5, 4, 3, and 2.
for halo_name in ['A1','A2','A4','A8']:
    for iSnapshot in [67,77,88,102,120,142]:
        resolution = 'HR'   #these parameters are not important
        simgroup = 'Daniel' #^

        # downsample = int(1e3) #use appropriate value if using downsampled data
        downsample = int(1e0) #use appropriate value if using downsampled data

        if halo_name == 'A1':
            simname = 'h206'
        elif halo_name == 'A2':
            simname = 'h29'
        elif halo_name == 'A4':
            simname = 'h113'
        elif halo_name == 'A8':
            simname = 'h2'

        if iSnapshot > 99:
            snap_id = str(iSnapshot)
        elif iSnapshot > 9:
            snap_id = '0' + str(iSnapshot)
        else:
            snap_id = '00' + str(iSnapshot)

        # with h5py.File('skirt_box_sizes.hdf5','r') as f:
        with h5py.File('skirt_box_sizes_temp.hdf5','r') as f: #uses the *temp.hdf5 version where A1_88 and A4_88 are set to 50 until i fix it
            skirt_box_size = f[halo_name+"_"+snap_id][()] #the size of the box on which SKIRT computation was run, in kpc

        snapshot_dir = '/mnt/home/chayward/firesims/fire2/MassiveFIRE/'+halo_name+'_res33000/' #directory for the FIRE snapshot data
        AHF_dir = '../ceph/anglesd_m13/'+simname+'_HR_sn1dy300ro100ss/halo_00000_smooth.dat' #directory for the smooth AHF file
        SKIRT_indices_dir = '../ceph/SKIRT_to_FIRE/SKIRT_indices_downsample_'+str(np.log10(downsample))+'_'+halo_name+'_'+snap_id+'.npy'
        # SKIRT_ILW_dir = './SKIRT_data/50 kpc/'+halo_name+'/'+snap_id+'/I_LW_SKIRT_downsample_'+str(np.log10(downsample))+'.npy'
        SKIRT_ILW_dir = '../SKIRT_data/R_vir/'+halo_name+'/'+snap_id+'/I_LW_SKIRT_downsample_'+str(np.log10(downsample))+'.npy'

        limit_to_50kpc = 1 #set these to 1
        skirt = 1          #^1
        mcrt = 0           #set this to 0



        #load snapshot and SKIRT data
        simFilesDic = simulationFiles.simulationFiles_dic[(simname,resolution,simgroup)]
        simFilesDic['AHF_fn'] = AHF_dir
        simFilesDic['snapshotDir'] = snapshot_dir
        filedir = './results'
        snapdir_name = ('','snapdir')[simFilesDic['snapshot_in_directory']] 
        meta = ff.Snapshot_meta(simname,simgroup,resolution,iSnapshot=iSnapshot,**simulationFiles.simulationFiles_dic[(simname,resolution,simgroup)])        
        snapshot = ff.Snapshot(meta,pr=False,loadAll=False)
        snapdict = abg_python.snapshot_utils.openSnapshot(            
            simulationFiles.simulationFiles_dic[(simname,resolution,simgroup)]['snapshotDir'],
            snapnum = iSnapshot,ptype=0,cosmological=1,snapdir_name=snapdir_name,
            header_only=False,keys_to_extract=['SmoothingLength'])

        r_max = skirt_box_size
        z_width = r_max

        #create input dictionary for FIRE studio
        studiodic = {}

        #optional - rotate snapshot
        # prof = ff.Snapshot_profiler(snapshot)
        # if proj_ax == 'z':
        #     rotated_coords = snapshot.rotated_vector(jvec = prof.central_jvec(),vector_str='coords',edge_on=False)
        # elif proj_ax == 'y':
        #     rotated_coords = snapshot.rotated_vector(jvec = prof.central_jvec(),vector_str='coords',edge_on=True)
        # elif proj_ax == 'x':
        #     rotated_coords = snapshot.rotated_vector(jvec = prof.central_jvec(),vector_str='coords',edge_on=True,rot_j = np.pi/2)

        coords = snapshot.coords()

        #filter particles within skirt box (not necessarily 50 kpc)
        if limit_to_50kpc:

            xc = coords[:,0]
            yc = coords[:,1]
            zc = coords[:,2]
            # ii = np.where((np.abs(xc)<50)&(np.abs(yc)<50)&(np.abs(zc)<50))
            SKIRT_box_size = skirt_box_size
            ii = np.where((np.abs(xc)<SKIRT_box_size)&(np.abs(yc)<SKIRT_box_size)&(np.abs(zc)<SKIRT_box_size))
            coords = coords[ii[0],:]

            studiodic['Coordinates'] =  coords
            # studiodic['Coordinates'] =  rotated_coords
            studiodic['Masses'] =  snapshot.HImasses()[ii[0]]
            studiodic['BoxSize'] = snapdict['BoxSize']
            studiodic['SmoothingLength'] = snapdict['SmoothingLength'][ii[0]]
            studiodic['StarCoordinates'] =  snapshot.coords(iPartType=4)
            studiodic['StarMasses'] =  snapshot.masses(iPartType=4)
            studiodic['StarDist'] =  snapshot.rs(iPartType=4)
            H2_masses = snapshot.H2masses(mcrt=mcrt,skirt=skirt,ILW_SKIRT_path=SKIRT_ILW_dir)[ii[0]]
            H2_masses_no_skirt = snapshot.H2masses(mcrt=mcrt,skirt=0,ILW_SKIRT_path=SKIRT_ILW_dir)[ii[0]]
        else:
            studiodic['Coordinates'] =  coords
            # studiodic['Coordinates'] =  rotated_coords
            studiodic['Masses'] =  snapshot.HImasses() 
            studiodic['BoxSize'] = snapdict['BoxSize']
            studiodic['SmoothingLength'] = snapdict['SmoothingLength']
            studiodic['StarCoordinates'] =  snapshot.coords(iPartType=4)
            studiodic['StarMasses'] =  snapshot.masses(iPartType=4)
            studiodic['StarDist'] =  snapshot.rs(iPartType=4)
            H2_masses = snapshot.H2masses(mcrt,skirt,ILW_SKIRT_path=SKIRT_ILW_dir)

        r_vir = snapshot.sim.rvir.value

        stellar_prof_bin_num = 1000
        r_stellar_prof = np.linspace(0,0.2*r_vir,num=stellar_prof_bin_num)

        stellar_mass_prof = np.zeros_like(r_stellar_prof)
        for i in range(0,stellar_prof_bin_num):
            iii = np.where(studiodic['StarDist']<r_stellar_prof[i])
            stellar_mass_prof[i] = np.sum(studiodic['StarMasses'][iii])

        #find stellar hals mass radius
        r_stellar_1_2 = r_stellar_prof[np.max(np.where(stellar_mass_prof<0.5*stellar_mass_prof[stellar_mass_prof.size-1]))] #stellar half mass radius



        #create projection

        projection_output_filename = 'example_jonathan' #not important

        if not os.path.isdir('./results'):
            os.mkdir('./results')
        if not os.path.isdir('./results/Plots'):
            os.mkdir('./results/Plots')

        #HI
        mystudio = GasStudio(
         snapdir = None, 
         snapnum = iSnapshot,
         snapdict = studiodic,
         datadir = filedir,
         frame_half_width = r_max,
         frame_depth = z_width,
         quantity_name = 'Masses',
         take_log_of_quantity = False, 
         galaxy_extractor = False,
         pixels=1200,
         single_image='Density',
         overwrite = True,
         savefig=False,      
         use_hsml=True,
         intermediate_file_name=projection_output_filename,
         )
        NHImap, _ = mystudio.projectImage([])
        NHImap += log((0.7*un.Msun*un.pc**-2/cons.m_p/1e10*snapshot.sim.h).to('cm**-2').value) #fix units
        XsNHI = np.linspace(-r_max,r_max,NHImap.shape[0])
        YsNHI = np.linspace(-r_max,r_max,NHImap.shape[1])



        #create input dictionary for FIRE studio
        # studiodic = {}
        # studiodic['Coordinates'] =  rotated_coords
        # studiodic['Coordinates'] =  snapshot.coords()
        studiodic['Masses'] =  H2_masses
        studiodic['BoxSize'] = snapdict['BoxSize']
        # studiodic['SmoothingLength'] = snapdict['SmoothingLength']


        #H2 
        mystudio = GasStudio(
         snapdir = None, 
         snapnum = iSnapshot,
         snapdict = studiodic,
         datadir = filedir,
         frame_half_width = r_max,
         frame_depth = z_width,
         quantity_name = 'Masses',
         take_log_of_quantity = False, 
         galaxy_extractor = False,
         pixels=1200,
         single_image='Density',
         overwrite = True,
         savefig=False,      
         use_hsml=True,
         intermediate_file_name=projection_output_filename,
         )
        NH2map, _ = mystudio.projectImage([])
        NH2map += log((0.7*un.Msun*un.pc**-2/cons.m_p/1e10*snapshot.sim.h).to('cm**-2').value) #fix units
        Xs = np.linspace(-r_max,r_max,NH2map.shape[0])
        Ys = np.linspace(-r_max,r_max,NH2map.shape[1])

        #H2_no_SKIRT

        studiodic['Masses'] =  H2_masses_no_skirt

        mystudio = GasStudio(
         snapdir = None, 
         snapnum = iSnapshot,
         snapdict = studiodic,
         datadir = filedir,
         frame_half_width = r_max,
         frame_depth = z_width,
         quantity_name = 'Masses',
         take_log_of_quantity = False, 
         galaxy_extractor = False,
         pixels=1200,
         single_image='Density',
         overwrite = True,
         savefig=False,      
         use_hsml=True,
         intermediate_file_name=projection_output_filename,
         )
        NH2map_no_skirt, _ = mystudio.projectImage([])
        NH2map_no_skirt += log((0.7*un.Msun*un.pc**-2/cons.m_p/1e10*snapshot.sim.h).to('cm**-2').value) #fix units
        Xs_no_skirt = np.linspace(-r_max,r_max,NH2map_no_skirt.shape[0])
        Ys_no_skirt = np.linspace(-r_max,r_max,NH2map_no_skirt.shape[1])


        #ILW

        ILW_ = snapshot.ILW_skirt(ILW_SKIRT_path = SKIRT_ILW_dir)[ii[0]]

        studiodic['Masses'] =  ILW_

        mystudio = GasStudio(
         snapdir = None, 
         snapnum = iSnapshot,
         snapdict = studiodic,
         datadir = filedir,
         frame_half_width = r_max,
         frame_depth = z_width,
         quantity_name = 'Masses',
         take_log_of_quantity = False, 
         galaxy_extractor = False,
         pixels=1200,
         single_image='Density',
         overwrite = True,
         savefig=False,      
         use_hsml=True,
         intermediate_file_name=projection_output_filename,
         )
        ILW_map, _ = mystudio.projectImage([])
        # ILW_map += log((0.7*un.Msun*un.pc**-2/cons.m_p/1e10*snapshot.sim.h).to('cm**-2').value) #fix units
        Xs_ILW = np.linspace(-r_max,r_max,ILW_map.shape[0])
        Ys_ILW = np.linspace(-r_max,r_max,ILW_map.shape[1])



        #plot projection

        r_vir = snapshot.sim.rvir.value

        fig = matplotlib.pyplot.figure(figsize = [3*6.4, 4.8], dpi = 100)
        # fig = matplotlib.pyplot.figure()
        ax = fig.subplots(1,3)
        # mystudio.plotImage(ax,[])
        ax[0].pcolor(Xs,Ys,NH2map)
        c = ax[0].pcolormesh(Xs,Ys,NH2map, cmap='viridis',vmin=10,vmax=22)
        # c = ax.pcolormesh(Xs,Ys,NH2map, cmap='PiYG',vmin=10,vmax=22)
        # c = ax.pcolormesh(Xs,Ys,NH2map, cmap='afmhot',vmin=10,vmax=22)
        cbar = fig.colorbar(c,ax=ax[0])
        # matplotlib.pyplot.title("log$\;N_{\mathregular{H}_2}\; [\mathregular{cm}^{-2}]$", fontsize=18)
        ax[0].set_title("log$\;N_{\mathregular{H}_2}\; [\mathregular{cm}^{-2}]$", fontsize=18)
        ax[0].tick_params(labelsize=16) 
        ax[0].set_xlabel('$x\;$ [kpc]',fontsize=16)
        ax[0].set_ylabel('$y\;$ [kpc]',fontsize=16)
        ax[0].set_xlim((-r_vir,r_vir))
        ax[0].set_ylim((-r_vir,r_vir))
        detec_lim = 14+np.log(2)

        # matplotlib.pyplot.contour(Xs,Ys,NH2map, levels=[14+np.log(2)], colors='white', linewidths = 0.8)
        ax[0].contour(Xs,Ys,NH2map, levels=[14+np.log(2)], colors='white', linewidths = 0.8)
        # matplotlib.pyplot.contour(Xs,Ys,NH2map, levels=[14+np.log(2)], linewidths = 0.8)


        r_gal = np.ones_like(NH2map)
        for i in range(0,Xs.size):
            for j in range(0,Ys.size):
                r_gal[i,j] = (Xs[i]**2+Ys[j]**2)**0.5

        ax[0].contour(Xs,Ys,r_gal, levels=[r_vir], colors='white')
        # matplotlib.pyplot.contour(Xs,Ys,r_gal, levels=[2.3], linewidths = 0.8)
        # matplotlib.pyplot.contour(Xs,Ys,r_gal, levels=[2.3], linewidths = 1, colors='white')
        ax[0].contour(Xs,Ys,r_gal, levels=[r_stellar_1_2], linewidths = 0.8)

        ax[1].pcolor(XsNHI,YsNHI,NHImap)
        c = ax[1].pcolormesh(XsNHI,YsNHI,NHImap, cmap='viridis',vmin=16,vmax=22)
        # c = ax.pcolormesh(Xs,Ys,NH2map, cmap='PiYG',vmin=10,vmax=22)
        # c = ax.pcolormesh(Xs,Ys,NH2map, cmap='afmhot',vmin=10,vmax=22)
        cbar = fig.colorbar(c,ax=ax[1])
        # matplotlib.pyplot.title("log$\;N_{\mathregular{H}_2}\; [\mathregular{cm}^{-2}]$", fontsize=18)
        ax[1].set_title("log$\;N_{\mathregular{HI}}\; [\mathregular{cm}^{-2}]$", fontsize=18)
        ax[1].tick_params(labelsize=16) 
        ax[1].set_xlabel('$x\;$ [kpc]',fontsize=16)
        ax[1].set_ylabel('$y\;$ [kpc]',fontsize=16)
        ax[1].set_xlim((-r_vir,r_vir))
        ax[1].set_ylim((-r_vir,r_vir))
        detec_lim = 14+np.log(2)

        # matplotlib.pyplot.contour(Xs,Ys,NH2map, levels=[14+np.log(2)], colors='white', linewidths = 0.8)
        # ax[0].contour(Xs,Ys,NH2map, levels=[14+np.log(2)], colors='white', linewidths = 0.8)
        # matplotlib.pyplot.contour(Xs,Ys,NH2map, levels=[14+np.log(2)], linewidths = 0.8)

        # r_vir = snapshot.sim.rvir.value
        # r_gal = np.ones_like(NH2map)

        ax[1].contour(Xs,Ys,r_gal, levels=[r_vir], colors='white')
        # matplotlib.pyplot.contour(Xs,Ys,r_gal, levels=[2.3], linewidths = 0.8)
        # matplotlib.pyplot.contour(Xs,Ys,r_gal, levels=[2.3], linewidths = 1, colors='white')
        ax[1].contour(Xs,Ys,r_gal, levels=[r_stellar_1_2], linewidths = 0.8)

        #---
        ax[2].pcolor(Xs_ILW,Ys_ILW,ILW_map)
        c = ax[2].pcolormesh(Xs_ILW,Ys_ILW,ILW_map, cmap='viridis', vmin=4, vmax=10)
        cbar = fig.colorbar(c,ax=ax[2])
        ax[2].set_title("log$\;I_{\mathregular{LW}}$", fontsize=18)
        ax[2].tick_params(labelsize=16) 
        ax[2].set_xlabel('$x\;$ [kpc]',fontsize=16)
        ax[2].set_ylabel('$y\;$ [kpc]',fontsize=16)
        ax[2].set_xlim((-r_vir,r_vir))
        ax[2].set_ylim((-r_vir,r_vir))


        fig.savefig('../ceph/cgm_H2_figs/'+halo_name+"_"+snap_id+"_map.png")


        # detectable covering factor

        detection_limit = log(2)+14
        resol = int(1e3)
        rs = np.zeros([Xs.size,Ys.size])
        for i in range(0,Xs.size):
            for j in range(0,Ys.size):
                rs[i,j] = (Xs[i]**2 + Ys[j]**2)**0.5

        r_range = np.linspace(0,2**0.5*r_max,resol)

        i_rs = np.searchsorted(r_range,rs)

        i_NH2_detec = NH2map>detection_limit
        covering_factor = np.zeros(r_range.size-1)


        bin_size = np.zeros(r_range.size-1)
        bin_value = np.zeros(r_range.size-1)
        for i in range(0,resol-1):
            bin_size[i] = np.count_nonzero(i_rs == i+1)
            bin_value[i] = np.count_nonzero(np.logical_and(i_rs == i+1 , i_NH2_detec))

        covering_factor = bin_value/bin_size

        bin_size = np.zeros(r_range.size-1)
        conversion_bin_value = np.zeros(r_range.size-1)

        i_molecular = NHImap<log(4)+NH2map

        for i in range(0,resol-1):
            bin_size[i] = np.count_nonzero(i_rs == i+1)
            conversion_bin_value[i] = np.count_nonzero(np.logical_and(i_rs == i+1 , i_molecular))

        conversion_factor = np.zeros(r_range.size-1)

        conversion_factor = conversion_bin_value/bin_size
        #plot detectable and covering together

        fig3 = matplotlib.pyplot.figure()
        ax3 = fig3.subplots()
        # ax2.plot(r_range[:-1],conversion_factor)

        # ax3.plot(r_range[:-1],covering_factor, label = '$\mathregular{N}_{\mathregular{H_2}}>2\cdot10^{14}\; \mathregular{cm}^{-2}$')
        # ax3.plot(r_range[:-1],conversion_factor, label = '$\mathregular{2N}_{\mathregular{H_2}}>\mathregular{N}_{\mathregular{HI}}$')
        ax3.plot(r_range[:-1]/r_vir,covering_factor, label = '$\mathregular{N}_{\mathregular{H_2}}>2\cdot10^{14}\; \mathregular{cm}^{-2}$')
        ax3.plot(r_range[:-1]/r_vir,conversion_factor, label = '$\mathregular{2N}_{\mathregular{H_2}}>\mathregular{N}_{\mathregular{HI}}$')
        # matplotlib.pyplot.xlim([0,30])
        matplotlib.pyplot.xlim([0,0.6])
        matplotlib.pyplot.ylim([0,1.1])
        # matplotlib.pyplot.xlabel('r (kpc)', fontsize = 18)
        matplotlib.pyplot.xlabel('$\\rho/R_{\mathregular{vir}}$', fontsize = 18)
        matplotlib.pyplot.ylabel('$\mathregular{f}_{\mathregular{covering}}$',  fontsize = 18)
        matplotlib.pyplot.grid('True')
        matplotlib.pyplot.legend(prop = {'size':12})

        i_detec = np.argmax(covering_factor < 0.5)
        r_detec = r_range[i_detec]

        # ax3.plot([r_detec,r_detec],[0,covering_factor[i_detec]], 'k-',linestyle = '--')
        # ax3.plot([0,r_detec],[covering_factor[i_detec],covering_factor[i_detec]], 'k-',linestyle = '--')
        ax3.plot([r_detec/r_vir,r_detec/r_vir],[0,covering_factor[i_detec]], 'k-',linestyle = '--')
        ax3.plot([0,r_detec/r_vir],[covering_factor[i_detec],covering_factor[i_detec]], 'k-',linestyle = '--')

        ax3.plot([r_stellar_1_2/r_vir,r_stellar_1_2/r_vir],[0,2], 'k-',linestyle = ':', lw=2, c="red")

        ax4 = ax3.twiny()

        new_tick_locations = np.array([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

        def tick_function(X):
            V = X * r_vir
            return ["%.0f" % z for z in V]

        ax4.set_xlim(ax3.get_xlim())
        ax4.set_xticks(new_tick_locations)
        ax4.set_xticklabels(tick_function(new_tick_locations))
        ax4.set_xlabel('$\\rho \; (\mathregular{kpc})$', fontsize = 18)


        matplotlib.pyplot.show()



        fig3.savefig('../ceph/cgm_H2_figs/'+halo_name+"_"+snap_id+"_prof.png",dpi=100)



        #plot projection no skirt

        r_vir = snapshot.sim.rvir.value

        fig = matplotlib.pyplot.figure(figsize = [3*6.4, 4.8], dpi = 100)
        # fig = matplotlib.pyplot.figure()
        ax = fig.subplots(1,3)
        # mystudio.plotImage(ax,[])
        ax[0].pcolor(Xs,Ys,NH2map_no_skirt)
        c = ax[0].pcolormesh(Xs,Ys,NH2map_no_skirt, cmap='viridis',vmin=10,vmax=22)
        # c = ax.pcolormesh(Xs,Ys,NH2map, cmap='PiYG',vmin=10,vmax=22)
        # c = ax.pcolormesh(Xs,Ys,NH2map, cmap='afmhot',vmin=10,vmax=22)
        cbar = fig.colorbar(c,ax=ax[0])
        # matplotlib.pyplot.title("log$\;N_{\mathregular{H}_2}\; [\mathregular{cm}^{-2}]$", fontsize=18)
        ax[0].set_title("log$\;N_{\mathregular{H}_2}\; [\mathregular{cm}^{-2}]$", fontsize=18)
        ax[0].tick_params(labelsize=16) 
        ax[0].set_xlabel('$x\;$ [kpc]',fontsize=16)
        ax[0].set_ylabel('$y\;$ [kpc]',fontsize=16)
        ax[0].set_xlim((-r_vir,r_vir))
        ax[0].set_ylim((-r_vir,r_vir))
        detec_lim = 14+np.log(2)

        # matplotlib.pyplot.contour(Xs,Ys,NH2map, levels=[14+np.log(2)], colors='white', linewidths = 0.8)
        ax[0].contour(Xs,Ys,NH2map_no_skirt, levels=[14+np.log(2)], colors='white', linewidths = 0.8)
        # matplotlib.pyplot.contour(Xs,Ys,NH2map, levels=[14+np.log(2)], linewidths = 0.8)


        r_gal = np.ones_like(NH2map_no_skirt)
        for i in range(0,Xs.size):
            for j in range(0,Ys.size):
                r_gal[i,j] = (Xs[i]**2+Ys[j]**2)**0.5

        ax[0].contour(Xs,Ys,r_gal, levels=[r_vir], colors='white')
        # matplotlib.pyplot.contour(Xs,Ys,r_gal, levels=[2.3], linewidths = 0.8)
        # matplotlib.pyplot.contour(Xs,Ys,r_gal, levels=[2.3], linewidths = 1, colors='white')
        ax[0].contour(Xs,Ys,r_gal, levels=[r_stellar_1_2], linewidths = 0.8)

        ax[1].pcolor(XsNHI,YsNHI,NHImap)
        c = ax[1].pcolormesh(XsNHI,YsNHI,NHImap, cmap='viridis',vmin=16,vmax=22)
        # c = ax.pcolormesh(Xs,Ys,NH2map, cmap='PiYG',vmin=10,vmax=22)
        # c = ax.pcolormesh(Xs,Ys,NH2map, cmap='afmhot',vmin=10,vmax=22)
        cbar = fig.colorbar(c,ax=ax[1])
        # matplotlib.pyplot.title("log$\;N_{\mathregular{H}_2}\; [\mathregular{cm}^{-2}]$", fontsize=18)
        ax[1].set_title("log$\;N_{\mathregular{HI}}\; [\mathregular{cm}^{-2}]$", fontsize=18)
        ax[1].tick_params(labelsize=16) 
        ax[1].set_xlabel('$x\;$ [kpc]',fontsize=16)
        ax[1].set_ylabel('$y\;$ [kpc]',fontsize=16)
        ax[1].set_xlim((-r_vir,r_vir))
        ax[1].set_ylim((-r_vir,r_vir))
        detec_lim = 14+np.log(2)

        # matplotlib.pyplot.contour(Xs,Ys,NH2map, levels=[14+np.log(2)], colors='white', linewidths = 0.8)
        # ax[0].contour(Xs,Ys,NH2map, levels=[14+np.log(2)], colors='white', linewidths = 0.8)
        # matplotlib.pyplot.contour(Xs,Ys,NH2map, levels=[14+np.log(2)], linewidths = 0.8)

        # r_vir = snapshot.sim.rvir.value
        # r_gal = np.ones_like(NH2map)

        ax[1].contour(Xs,Ys,r_gal, levels=[r_vir], colors='white')
        # matplotlib.pyplot.contour(Xs,Ys,r_gal, levels=[2.3], linewidths = 0.8)
        # matplotlib.pyplot.contour(Xs,Ys,r_gal, levels=[2.3], linewidths = 1, colors='white')
        ax[1].contour(Xs,Ys,r_gal, levels=[r_stellar_1_2], linewidths = 0.8)

        #---
        ax[2].pcolor(Xs_ILW,Ys_ILW,ILW_map)
        c = ax[2].pcolormesh(Xs_ILW,Ys_ILW,ILW_map, cmap='viridis', vmin=4, vmax=10)
        cbar = fig.colorbar(c,ax=ax[2])
        ax[2].set_title("log$\;I_{\mathregular{LW}}$", fontsize=18)
        ax[2].tick_params(labelsize=16) 
        ax[2].set_xlabel('$x\;$ [kpc]',fontsize=16)
        ax[2].set_ylabel('$y\;$ [kpc]',fontsize=16)
        ax[2].set_xlim((-r_vir,r_vir))
        ax[2].set_ylim((-r_vir,r_vir))


        fig.savefig('../ceph/cgm_H2_figs/'+halo_name+"_"+snap_id+"_map_no_skirt.png")


        # detectable covering factor no skirt

        detection_limit = log(2)+14
        resol = int(1e3)
        rs = np.zeros([Xs.size,Ys.size])
        for i in range(0,Xs.size):
            for j in range(0,Ys.size):
                rs[i,j] = (Xs[i]**2 + Ys[j]**2)**0.5

        r_range = np.linspace(0,2**0.5*r_max,resol)

        i_rs = np.searchsorted(r_range,rs)

        i_NH2_detec = NH2map_no_skirt>detection_limit
        covering_factor_no_skirt = np.zeros(r_range.size-1)


        bin_size = np.zeros(r_range.size-1)
        bin_value = np.zeros(r_range.size-1)
        for i in range(0,resol-1):
            bin_size[i] = np.count_nonzero(i_rs == i+1)
            bin_value[i] = np.count_nonzero(np.logical_and(i_rs == i+1 , i_NH2_detec))

        covering_factor_no_skirt = bin_value/bin_size

        bin_size = np.zeros(r_range.size-1)
        conversion_bin_value = np.zeros(r_range.size-1)

        i_molecular = NHImap<log(4)+NH2map_no_skirt

        for i in range(0,resol-1):
            bin_size[i] = np.count_nonzero(i_rs == i+1)
            conversion_bin_value[i] = np.count_nonzero(np.logical_and(i_rs == i+1 , i_molecular))

        conversion_factor_no_skirt = np.zeros(r_range.size-1)

        conversion_factor_no_skirt = conversion_bin_value/bin_size
        #plot detectable and covering together

        fig3 = matplotlib.pyplot.figure()
        ax3 = fig3.subplots()
        # ax2.plot(r_range[:-1],conversion_factor)

        # ax3.plot(r_range[:-1],covering_factor, label = '$\mathregular{N}_{\mathregular{H_2}}>2\cdot10^{14}\; \mathregular{cm}^{-2}$')
        # ax3.plot(r_range[:-1],conversion_factor, label = '$\mathregular{2N}_{\mathregular{H_2}}>\mathregular{N}_{\mathregular{HI}}$')
        ax3.plot(r_range[:-1]/r_vir,covering_factor_no_skirt, label = '$\mathregular{N}_{\mathregular{H_2}}>2\cdot10^{14}\; \mathregular{cm}^{-2}$')
        ax3.plot(r_range[:-1]/r_vir,conversion_factor_no_skirt, label = '$\mathregular{2N}_{\mathregular{H_2}}>\mathregular{N}_{\mathregular{HI}}$')
        # matplotlib.pyplot.xlim([0,30])
        matplotlib.pyplot.xlim([0,0.6])
        matplotlib.pyplot.ylim([0,1.1])
        # matplotlib.pyplot.xlabel('r (kpc)', fontsize = 18)
        matplotlib.pyplot.xlabel('$\\rho/R_{\mathregular{vir}}$', fontsize = 18)
        matplotlib.pyplot.ylabel('$\mathregular{f}_{\mathregular{covering}}$',  fontsize = 18)
        matplotlib.pyplot.grid('True')
        matplotlib.pyplot.legend(prop = {'size':12})

        i_detec = np.argmax(covering_factor_no_skirt < 0.5)
        r_detec = r_range[i_detec]

        # ax3.plot([r_detec,r_detec],[0,covering_factor[i_detec]], 'k-',linestyle = '--')
        # ax3.plot([0,r_detec],[covering_factor[i_detec],covering_factor[i_detec]], 'k-',linestyle = '--')
        ax3.plot([r_detec/r_vir,r_detec/r_vir],[0,covering_factor_no_skirt[i_detec]], 'k-',linestyle = '--')
        ax3.plot([0,r_detec/r_vir],[covering_factor_no_skirt[i_detec],covering_factor_no_skirt[i_detec]], 'k-',linestyle = '--')

        ax3.plot([r_stellar_1_2/r_vir,r_stellar_1_2/r_vir],[0,2], 'k-',linestyle = ':', lw=2, c="red")

        ax4 = ax3.twiny()

        new_tick_locations = np.array([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

        def tick_function(X):
            V = X * r_vir
            return ["%.0f" % z for z in V]

        ax4.set_xlim(ax3.get_xlim())
        ax4.set_xticks(new_tick_locations)
        ax4.set_xticklabels(tick_function(new_tick_locations))
        ax4.set_xlabel('$\\rho \; (\mathregular{kpc})$', fontsize = 18)


        matplotlib.pyplot.show()



        fig3.savefig('../ceph/cgm_H2_figs/'+halo_name+"_"+snap_id+"_prof_no_skirt.png",dpi=100)







        with h5py.File('../ceph/cgm_H2_data/'+halo_name+"_"+snap_id+"_map.hdf5",'w') as f:
            f.create_dataset("HI",data=NHImap)
            f.create_dataset("x_HI",data=XsNHI)
            f.create_dataset("y_HI",data=YsNHI)
            f.create_dataset("H2",data=NH2map)
            f.create_dataset("H2",data=NH2map_no_skirt)
            f.create_dataset("x_H2",data=Xs)
            f.create_dataset("y_H2",data=Ys)
            f.create_dataset("ILW",data=ILW_map)
            f.create_dataset("x_ILW",data=Xs_ILW)
            f.create_dataset("y_ILW",data=Ys_ILW)

            f.create_dataset("r_prof_kpc",data=r_range)
            f.create_dataset("detec_prof",data=covering_factor)
            f.create_dataset("conversion_prof",data=conversion_factor)
            f.create_dataset("detec_prof_no_skirt",data=covering_factor_no_skirt)
            f.create_dataset("conversion_prof_no_skirt",data=conversion_factor_no_skirt)
            f.create_dataset("r_vir_kpc",data=r_vir)
            f.create_dataset("r_stellar_1_2_kpc",data=r_stellar_1_2)

