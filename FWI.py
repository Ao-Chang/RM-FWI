import os
import sys
import datetime
import numpy as np
import scipy.stats as st
import scipy.spatial as sp
import matplotlib
matplotlib.use('agg')
import matplotlib.pylab as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import scipy.interpolate as interpolate

fpth = os.path.abspath(os.path.join('rmwspy'))
sys.path.append(fpth)
from random_mixing_whittaker_shannon import *
from basics_fwi import *
from numpy import linalg as LA
from mpl_toolkits.axes_grid1 import make_axes_locatable




class SonicWaveModel(NonLinearProblemTemplate):
    def __init__(self, domain, model, frequencies, Src_ids, Src_tags, Src_amps, use=None, data=None, communicator=None, testFieldConsistency=False):

        self.domain = domain
        self.model = model
        self.frequencies = frequencies
        self.Src_ids = Src_ids
        self.Src_tags = Src_tags
        self.Src_amps = Src_amps
        self.use = use   # mask which data from model output to be used
        self.data = data # the corresponding data
        self.communicator = communicator # MPI communicator, if None no MPI is used.
        if communicator is None:
            self.isMPIRoot = True
            self.useMPI=False
            self.testFieldConsistency
        else:
            self.isMPIRoot = communicator.Get_rank() == 0
            self.useMPI= communicator.Get_size() > 1
            self.testFieldConsistency=testFieldConsistency # this tests if all MPI ranks have the same values. Use this for debugging only. 
    
    def runFieldConsistencyTest(self, fields):
            """
            this checks fields as the same value on all MPI ranks.
            rank=0 is used as a reference
            """
            if self.communicator is not None:
                from mpi4py import MPI
                testfields=np.copy(fields)
                self.communicator.Bcast(testfields, root=0) 
                n=np.linalg.norm(fields)
                e=np.linalg.norm(fields-testfields)
                print("Consistence check: rank %s: difference of field to rank 0: %e (max=%e)."%(self.communicator.Get_rank(), e, n))
                if e < 1e-10* n:
                    errorcode=0
                else:
                    errorcode=1
                errorcode=self.communicator.allreduce(errorcode, op=MPI.MAX)
                if errorcode >0:
                    raise ValueError("Inconsistent random fields across ranks detected.")
                
    def objective_function(self, prediction):
        print("prediction dim=",prediction.shape)
        return LA.norm((np.log10(prediction/self.data)),axis=1)
        # if prediction.ndim == 1:
        #     return LA.norm(self.data - prediction)
        # elif prediction.ndim == 2:
        #     return LA.norm((self.data - prediction), axis=1)
        # elif prediction.ndim == 3:
        #     obs3d = np.atleast_3d(self.data).reshape(-1, 1, 1)
        #     return LA.norm((obs3d - prediction), axis=0)
    
    def allforwards(self, fields):
        """
        this runs all the forward models for the nfields realizations fields.
        and returns an array out[nfields,ndata] where ndata is the number of observations
        calculated by the forward model with out[i,:] being the output observations for
        field fields[i] (i=0,...,nfields-1)
        
        """
        ResultType=complex

        if self.testFieldConsistency: self.runFieldConsistencyTest(fields)
        nfields=fields.shape[0]
        nfrq=len(self.frequencies)
        ndata=self.data.shape[0]
        out = np.empty((nfields, ndata), dtype=ResultType)
        # this how it is done if there is no MPI involved:
        if not self.useMPI:
            for ifield in range(nfields):
                result=self.forward(self.frequencies, self.Src_amps, fields[ifield]) # this is for all frequencies, sources, observations
                out[ifield] = result[self.use]     # we grab all the observations we need marked by use
                
        else:
            # this is shape of the return array of the model:
            data_shape=(len(self.Src_ids),  len(Receiver_ids))
            # this is portion of the work load of each rank:
            portionWork=(nfields*nfrq)//self.communicator.Get_size()
            if not (nfields*nfrq)%comm.Get_size() ==0 : portionWork+=1 # in case we have lost some work in the splitting
            # the results are first collected in this array:
            myResults=np.empty((portionWork,) +  data_shape, dtype=ResultType)
            for ifield in range(nfields):  # loop over fields
                for ifrq in range(nfrq):   # loop over frequency
                    i=ifield*nfrq+ifrq     # index of result in an virtual array of length (nfields*nfrq) 
                    irank=i//portionWork  # which rank should work on this index
                    if irank == self.communicator.Get_rank(): # if I am the rank, lets do it:
                        myResults[i%portionWork]=self.forward([ self.frequencies[ifrq] ], self.Src_amps[:,ifrq:ifrq+1], fields[ifield])[0]
            
            # we collect the big array of the results which is then copied to all ranks:
            results=np.empty( (self.communicator.Get_size(),portionWork) + data_shape, dtype=ResultType)
            self.communicator.Allgather(myResults, results)
            # now we need to remove the unused bits:
            results=results.reshape( (self.communicator.Get_size()*portionWork, )+data_shape)[:nfields*nfrq]
            # now this is reshaped to get the first dimension to be the number of data fields:
            results=results.reshape( (nfields,nfrq)+data_shape)
            # it assumed here that all ranks have the same values in the fields array:
            for ifield in range(nfields):
                out[ifield] = results[ifield][self.use]     # we grab all the observations we need marked by use
            # just checking Consistence:
            if self.testFieldConsistency: self.runFieldConsistencyTest(out)
        return out

    def marginal_transformation(self, T):
        ##marginal for random field
        return st.gamma.ppf(st.norm.cdf(T), 4, 1500, 200)
        ## marginal for sinus field
        #return st.gennorm.ppf(st.norm.cdf(T), 3, 3000, 500)

    def forward(self, frequencies, Src_amps, field):        
        # transform marginal
        t = self.marginal_transformation(field)
        

        x = self.domain.getX()
              
        # if not self.useMPI: #this test can create chaos on MPI:
        #     assert inf(T)>0
        #     assert inf(S)>0
        if self.communicator:
            print("Rank: %d solves for frequencies %s"%(self.communicator.Get_rank(), frequencies))
        v=mapToDomain(domain, t, Resolution, origin=(PaddingX, PaddingZ))
        responses = self.model.runSurvey(frequencies, v, self.Src_ids, Src_amps, self.Src_tags)

        nlvals_at_x = responses # abs is just used in this pumping example
        return nlvals_at_x

def rSquare(estimations, measureds):
	""" Compute the coefficient of determination of random data. 
   This metric gives the level of confidence about the model used to model data"""
	SEE = (abs( np.array(measureds) - np.array(estimations) )**2 ).sum()
	mMean = (np.array(measureds)).sum() / float(len(measureds))
	dErr = (abs(mMean - measureds)**2).sum()

	return 1 - (SEE/dErr)



if __name__ == "__main__":
    
    # import MPI module:
    # if failed some substitutes are set so this still runs on a single MPI rank
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        isMPIRoot = comm.Get_rank() == 0
        if isMPIRoot: print("MPI with %s ranks started."%comm.Get_size())
    except ImportError:
        comm=None
        isMPIRoot = True
    
    start = datetime.datetime.now()

    if isMPIRoot:
       ROOT='/scratch/uqacha25/interval_fwi_mpi'
    
    numWSNodes=8 # number of Whittaker-Shannon interpolation nodes (normally 8)

    # read survey:
    survey=np.load("data_random.npy")
    Nx=survey.item().get('gridx')
    Ny=survey.item().get('gridz')
    Width=survey.item().get('width')
    Depth=survey.item().get('depth')
    # get data for inversion
    Source = survey.item().get('source_loc')
    src_tags=survey.item().get('sourcetags')
    Sourcetags = src_tags
    if isMPIRoot: print("%s Source found."%(len(Source)) )
    
    
    # same domain is generated on all MPI ranks:
    domain = Rectangle(Nx, Ny, l0=Width, l1=Depth, diracPoints=Source, diracTags=Sourcetags, order=1, fullOrder=True)
    # if isMPIRoot: print("Gmsh msh file read from %s"%(GMESHFN))
    X=domain.getX()
     
    model = SeismicWaveFrequency2DModel(domain)

    #PML
    Resolution=10*U.m
    RefinementFactor=2

    PaddingCellsX=15   
    PaddingCellsZ=10
    PaddingX=PaddingCellsX*Resolution
    PaddingZ=PaddingCellsZ*Resolution

    xmin=inf(X[0])
    xmax=sup(X[0])
    zmin=inf(X[1])
    zmax=sup(X[1])
    model.setpmlx(PaddingX,xmax)    
    model.setpmlz(PaddingZ,zmax)

    Receiver_ids = survey.item().get('nreceivers')
    Receiver_loc = survey.item().get('receiver')
    if isMPIRoot: print("Receiver :", Receiver_ids, "(", [ Receiver_loc[r] for r in Receiver_ids], ")")

    model.setReceivers(Receiver_loc, Receiver_ids)

    frequencies = survey.item().get('freq')
    Src_ids = survey.item().get('nsource')
    Src_tags = [ Sourcetags[s] for s in Src_ids]
    if isMPIRoot: print("Sources found  :", Src_tags, flush=True)
    Src_amps = survey.item().get('amplitude')
    Data = survey.item().get('signal') # This is just to make frequency the first dimension
    use = np.where(np.isnan(Data) == False)
    data = Data[use]
    D=[]

    numFrq=len(frequencies)
    if isMPIRoot: print("%s frequencies found."%numFrq)
    # Best efficiency is achieved when  numFrq * (numWSNodes-1) equals comm.Get_size() or is a multiple thereof.
    # (numWSNodes-1) is used as periodicity of interpolation nodes is use: 
    if comm is not None:
        if not (numFrq * (numWSNodes-1))%comm.Get_size() == 0:
            if isMPIRoot: print("INFORMATION: number of ranks (=%s) should be multiple of %s."%(comm.Get_size(),numFrq * (numWSNodes-1)))
        
    # initialize pumping model
    my_model = SonicWaveModel(domain, model, frequencies=frequencies, Src_ids=Src_ids, Src_tags=Src_tags, Src_amps=Src_amps, use=use, data=data, communicator=comm, testFieldConsistency=False)

    nFields = 100
    cmod = '1.0 Sph(50)' # this is for random field
    #cmod = '1.0 Sph(12)' # this is for sinus field 
    # This makes sure that all RMWSCondSim are identical. The seed argument needs to be the same on all MPI ranks.
    np.random.seed(345)
    # initialize Random Mixing Whittaker-Shannon
    # INFO: this is running on all MPI ranks but only the fields generated on comm.Get_rank()==0 are used!
    CS = RMWSCondSim(my_model,
                    domainsize = (200, 50),
                    covmod = cmod,
                    nFields = nFields,
                    p_on_circle = numWSNodes,
                    optmethod = 'circleopt',
                    minObj = 0.001,    
                    maxiter = 50,
                    )

    # run RMWS
    CS()

    # save the fields:
    # note that they are in standard normal
    # To avoid overwriting the file this is done on the MPI root rank only:  
    if isMPIRoot:
        print('cs.fields',CS.finalFields)
    # to get a scatter plot of data vs sim 
    # we need to run the forward model again using them
    # Again this is only done on the MPI rank with comm.Get_rank()==0
    all_sim_data = my_model.allforwards(CS.finalFields)
    np.save(os.path.join(ROOT,'sim_data.npy'), all_sim_data)
    if isMPIRoot:
        for i in range(nFields):
            sim_data=all_sim_data[i]
            r_value=rSquare(sim_data, data)
            
            plt.figure(figsize=(6,6))
            plt.scatter(abs(data), abs(sim_data))
            plt.plot(abs(data), abs(data), c='orange') 
            plt.xlabel('Observed data', fontsize=14)
            plt.ylabel('Predicted data', fontsize=14)
            plt.title(r'$R^2={:.3f}$'.format(r_value), fontsize=20)
            plt.xlim((0,16))
            plt.ylim((0,16))
            plt.tight_layout()
            plt.savefig('scatter_{}.png'.format(i))
            plt.clf()
            plt.close()
            # also plot the simulated field
            T = my_model.marginal_transformation(CS.finalFields[i])
            D.append(T)
            plt.figure(figsize=(8,4))
            ax = plt.gca()
            im = ax.imshow(T.T, origin='lower',cmap=plt.get_cmap('jet'),extent=(0, Width-2*PaddingX, -Depth+PaddingZ, 0),vmin=1500,vmax=3500)
            plt.gca().xaxis.set_ticks_position('top')
            plt.gca().xaxis.set_label_position('top')
            plt.xlabel('Distance (m)', fontsize=14)
            plt.ylabel('Depth (m)', fontsize=14)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2%", pad=0.2)
            clb = plt.colorbar(im, cax=cax)
            tick_locator = ticker.MaxNLocator(nbins=5)
            clb.locator = tick_locator
            clb.update_ticks()
            plt.tight_layout()
            plt.savefig('sim_{}.png'.format(i), bbox_inches='tight', pad_inches=0)
            plt.clf()
            plt.close()
        D=np.array(D)
        np.save(os.path.join(ROOT,'field_velocity(wrong).npy'),D)
        mean=np.mean(D, axis=0)
        plt.figure(figsize=(8,4))
        # plt.title("Mean", fontsize=14)
        plt.imshow(mean.T, origin='lower',cmap=plt.get_cmap('jet'),extent=(0, Width-2*PaddingX, -Depth+PaddingZ, 0))
        plt.gca().xaxis.set_ticks_position('top')
        plt.gca().xaxis.set_label_position('top')
        plt.xlabel('Distance (m)', fontsize=14)
        plt.ylabel('Depth (m)', fontsize=14)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.2)
        clb = plt.colorbar(im, cax=cax)
        tick_locator = ticker.MaxNLocator(nbins=5)
        clb.locator = tick_locator
        clb.update_ticks()
        clb.set_label("v (m/s)", fontsize=16)
        plt.tight_layout()
        plt.savefig('mean.png', bbox_inches='tight', pad_inches=0)
        plt.clf()
        plt.close()

        sta_dev=np.std(D, axis=0)
        plt.figure(figsize=(8,4))
        # plt.title("Standard Deviation", fontsize=14)
        plt.imshow(sta_dev.T, origin='lower',cmap=plt.get_cmap('jet'),extent=(0, Width-2*PaddingX, -Depth+PaddingZ, 0))
        plt.gca().xaxis.set_ticks_position('top')
        plt.gca().xaxis.set_label_position('top')
        plt.xlabel('Distance (m)', fontsize=14)
        plt.ylabel('Depth (m)', fontsize=14)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.2)
        clb = plt.colorbar(im, cax=cax)
        tick_locator = ticker.MaxNLocator(nbins=5)
        clb.locator = tick_locator
        clb.update_ticks()
        clb.set_label("v (m/s)", fontsize=16)
        plt.tight_layout()
        plt.savefig('standard deviation.png', bbox_inches='tight', pad_inches=0)
        plt.clf()
        plt.close()


    end = datetime.datetime.now()
    print('time needed:', end - start)




# # seperate frequency############ 
            # data_array=np.array(data)
            # sim_array=np.array(sim_data)
            # data_new=data.reshape(len(frequencies), len(Source), len(Receiver_loc))
            # sim_data_new=sim_data.reshape(len(frequencies), len(Source), len(Receiver_loc))
            # np.save(os.path.join(ROOT,'data_obe%s.npy'%i),data_new)
            # np.save(os.path.join(ROOT,'data_pre%s.npy'%i),sim_data_new)

            # for j in range (len(frequencies)):
            #   x=abs(data_new[j,:,:])
            #   y=abs(sim_data_new[j,:,:])
            #   plt.figure()
            #   plt.scatter(x,y)
            #   plt.plot(x1,y1,c='orange')
            #   plt.xlabel('data')
            #   plt.ylabel('sim')
            #   plt.title('r_value = {:.3f},intercept={:.3f},frequency={:.3f}'.format(r_value,intercept,frequencies[j]))
            #   plt.savefig('scatter_%s(0729)%s.png'%(i,j))
            #   plt.clf()
            #   plt.close()
            #############################
            #frequency with different colors####################
            # xs=[abs(data_new[i,:,:]) for i in range (len(frequencies))]
            # ys=[abs(sim_data_new[j,:,:]) for j in range (len(frequencies))]
            # cs=cm.rainbow(np.linspace(0,1,len(frequencies)))
            # groups=("15","18","20","21.5","25","27","30","32","35")
            # y2=x1
            # gradient, intercept_imag, r_imag, p_imag, std_err_imag=st.linregress(data.imag, sim_data.imag)
            # x2=data.imag
            # y2=gradient*x2+intercept_imag
            # plt.figure()
            # for x, y, c, group in zip (xs, ys, cs, groups):
            #   plt.scatter(x, y ,s=20, color=c, label=group)
            # plt.scatter(x, y, color=cs,label=frequencies)
            # plt.legend(loc="best", title="frequency")
            # plt.plot(x1, y1, c='orange')
            # plt.xlabel('data')
            # plt.ylabel('sim')
            # plt.title('r_value = {:.3f},intercept={:.3f}'.format(r_value,intercept))
            # plt.savefig('scatter_{}(0807).png'.format(i))
            # plt.clf()
            # plt.close()
            ##########################################