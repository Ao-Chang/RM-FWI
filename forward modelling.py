from __future__ import print_function,division
from esys.escript import *
from RMFWI import *
from esys.finley import Rectangle
from esys.escript.linearPDEs import LinearPDE,SolverOptions
from esys.escript.pdetools import Locator
from esys.escript import unitsSI as U
from esys.weipa import saveSilo
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
matplotlib.use('agg')
import matplotlib.pylab as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import numpy as np
import math 
import os
from esys.finley import ReadMesh, ReadGmsh
import scipy.stats as st

# import h5py

def setupPDE(domain):
	"""
	used t setup all ERT PDEs
	"""
	pde=LinearPDE(domain, numEquations=1, isComplex=True)
	pde.setSymmetryOn()
	optionsG=pde.getSolverOptions()
	optionsG.setSolverMethod(SolverOptions.DIRECT)

	#optionsG.setSolverMethod(SolverOptions.PCG)
	optionsG.setTolerance(1e-8)
	if True and hasFeature('trilinos'):
		optionsG.setPackage(SolverOptions.TRILINOS)
		#optionsG.setPreconditioner(SolverOptions.AMG)
		#optionsG.setTrilinosParameter("problem:type", "Poisson-3D")
		optionsG.setTrilinosParameter("verbosity", "none")
		optionsG.setTrilinosParameter("number of equations", 1)
		optionsG.setTrilinosParameter("problem: symmetric", True)
		optionsG.setTrilinosParameter("smoother: pre or post", "both")
	return pde

class SeismicWaveFrequency2DModel(object):
	
	def __init__(self, domain):
		
		self.domain=domain
		self.pdeK=setupPDE(domain)
						
	def getDomain(self):
		return self.pdeK.getDomain()

	def setbeta(self, locx, x, pml_layer):
		return (abs(locx-x)/pml_layer)**2

	def setpmlx(self,p_x,xmax):
		self.p_x=p_x
		self.p_xx=xmax-self.p_x

	def setpmlz(self, p_z,zmax):
		self.p_z=p_z
		self.p_zz=zmax-self.p_z

	def PMLBoundary(self, npml, para, X, frequencies, n, velocity):
		self.omega=frequencies*2*np.pi
		self.k=self.omega/velocity
		self.eta=n*velocity
		beta_1=self.setbeta(npml,X,npml)
		beta_2=self.setbeta(para,X,npml)

		direction_1=X-npml
		direction_2=X-para

		mask1=whereNegative(direction_1)
		mask2=wherePositive(direction_2)
			
		beta=beta_1*mask1+beta_2*mask2+0*(1-mask1-mask2)
		gama=1-1j*self.eta*beta/self.omega
		
		return gama
	

	def setReceivers(self, recorder_location, NRecorders):
		"""
		sets receivers
		
		:receiver_locations: location of receivers
		"""
		self.rec_ids=NRecorders
		self.RecorderLocator=Locator(Solution(self.getDomain()), recorder_location)  
	   

	def runSurvey(self, frequencies, velocity, sources, source_amplitudes, source_tags):

		X=self.domain.getX()
		# initialize return array:

		response=np.empty((len(frequencies), len(sources), len(self.rec_ids)), dtype=complex)
		for q in range(len(frequencies)):
			gamax=self.PMLBoundary(self.p_x, self.p_xx,X[0],frequencies[q],0.2,velocity)
			gamaz=self.PMLBoundary(self.p_z, X[1],X[1],frequencies[q],0.2,velocity)
			D =-self.k**2*gamax*gamaz
			self.pdeK.setValue(D=D)
			A=self.pdeK.createCoefficient("A")
			A[0,0]=gamaz/gamax
			A[1,1]=gamax/gamaz
			self.pdeK.setValue(A=A) 
			for o in range(len(sources)):
				print("solving for frequency %e Hz and source %s"%(frequencies[q],source_tags[o]), flush=True)
				src=Scalar(1j*0.,DiracDeltaFunctions(self.getDomain()))
				src.setTaggedValue(source_tags[o], source_amplitudes[o,q])
				self.pdeK.setValue(y_dirac=src)
				p=self.pdeK.getSolution()
				lp=self.RecorderLocator(p)
				for s in range(len(self.rec_ids)):
					response[q,o,s]=lp[s]
		return response

	
if __name__ == '__main__':
	
	source_amplitudes= 8.+1j*3 # m^3/sec
	Frequencies = np.arange(5.,15.,1.) # Hz
	# print("frequencies", Frequencies)

	VelocityFile='randomfield.npy'
	# VelocityFile=None
	if VelocityFile:
		vel = np.load(VelocityFile)
		velocity_data=vel
	else:
		velocity_data=getMarmousi()	
	print("Found velocity data over grid %s"%(velocity_data.shape,))

	##set up field with sinus wave 
	# y=np.linspace(0, 2000, 200)
	# x=np.linspace(0, 500, 50)
	# xx,yy=np.meshgrid(x,y)
	# v0=3000
	# dv=800
	# z=0.025# wavelength 2*pi/f
	# vel=v0+dv*np.sin(z*(500-xx))*np.sin(z*(1000-yy))
	
	
	Resolution=10*U.m # = x[1]-x[0]
	RefinementFactor=2
	#number of source and receivers
	NSources=21
	NRecorders=61
	#distance of source and recerivers from boundary
	SourceOffsetCell=40
	OffsetCell=30
	#PML
	PaddingCellsX=15 
	PaddingCellsZ=10
	PaddingX=PaddingCellsX*Resolution
	PaddingZ=PaddingCellsZ*Resolution

	CoreWidth=velocity_data.shape[0]*Resolution 
	CoreDepth=velocity_data.shape[1]*Resolution 
	Width=CoreWidth+2*PaddingX                  
	Depth=CoreDepth+PaddingZ              
   
	SourceSpacing=math.floor((velocity_data.shape[0]-2*SourceOffsetCell)/(NSources-1))*Resolution
	SourceOffset=(PaddingCellsX+SourceOffsetCell)*Resolution
	Surface=Depth
	sources = [ (SourceOffset + SourceSpacing * n, Surface) for n in range(NSources) ]
	sourceTags = ["src%s"%n for n in range(NSources) ]
	RecorderSpacing=math.floor((velocity_data.shape[0]-2*OffsetCell)/(NRecorders-1))*Resolution
	Offset=(PaddingCellsX+OffsetCell)*Resolution    
	recorder_locations=[(Offset+RecorderSpacing*j, Surface) for j in range(NRecorders)]

	nex=(velocity_data.shape[0]+PaddingCellsX*2)*RefinementFactor
	nez=(velocity_data.shape[1]+PaddingCellsZ)*RefinementFactor
	domain=Rectangle(nex,nez,l0=Width,l1=Depth, diracPoints=sources, diracTags=sourceTags, order=1, fullOrder=True)
	X=domain.getX()
	
	# print("shape of velocity field = %s"%(velocity_data.shape,))
	# print("FEM gid =  %s x %s"%(nex, nez))

	v=mapToDomain(domain, velocity_data, Resolution, origin=(PaddingX, PaddingZ))
	# saveSilo("velocity", v=v)


	xmin=inf(X[0])
	xmax=sup(X[0])
	zmin=inf(X[1])
	zmax=sup(X[1])
	
	
	model=SeismicWaveFrequency2DModel(domain)
	model.setpmlx(PaddingX,xmax)	
	model.setpmlz(PaddingZ,zmax)		
	
	Src_ids = [ i for i in range(NSources) ]
	Src_tags = [ sourceTags[s] for s in Src_ids]
	Rcv_ids = [ i for i in range(NRecorders) ]
	Rcv_loc = [ recorder_locations[r] for r in Rcv_ids] 
	Src_amps = np.full((len(Src_tags), len(Frequencies)), source_amplitudes)

	# print("Receivers :", [ recorder_locations[r] for r in Rcv_ids] )
	# print("Source :", [sources[r] for r in Src_ids])
	
	model.setReceivers(Rcv_loc, Rcv_ids)
		
	responses=model.runSurvey(Frequencies, v, Src_ids, Src_amps, Src_tags)
	
	
	# print(responses)
	
	np.save("data_random",{"gridx": nex,
			"gridz": nez,
			"width": Width,
			"depth": Depth,
			"freq": Frequencies,
			"nsource": Src_ids,
			"source_loc": sources,
			"sourcetags": Src_tags,
			"nreceivers": Rcv_ids,
			"receiver": recorder_locations,
			"amplitude": Src_amps,
			"signal": responses})
	

	plt.figure(figsize=(8,4))
	ax = plt.gca()
	im = ax.imshow(velocity_data.T, origin='lower', cmap='jet',  extent=(0,Resolution*velocity_data.shape[0],-Resolution*velocity_data.shape[1],0), vmin=1500, vmax=3500)
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
	plt.savefig("v(random field).png", bbox_inches='tight', pad_inches=0)
	plt.clf()
	plt.close()

	# true_rcv_offsets=[ x[0] for x in model.RecorderLocator.getX()]
	# print("true receiver offsets = ", true_rcv_offsets )
	# print("true receiver offsets range = ", min(true_rcv_offsets), max(true_rcv_offsets) )

	print("all done")
