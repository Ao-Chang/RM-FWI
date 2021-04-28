#-------------------------------------------------------------------------------
# Name:        Random Mixing using Whittaker-Shannon Interpolation
#
# Author:      Dr.-Ing. S. Hoerning
#
# Created:     10.01.2019, Centre for Natural Gas, EAIT,
#			   The University of Queensland, Brisbane, QLD, Australia
#-------------------------------------------------------------------------------

import os
import sys
import numpy as np
import matplotlib
import matplotlib.pylab as plt
import scipy.stats as st
import scipy.spatial as sp
import itertools as it
import spectralsim as Specsim
import covariancefunction as covfun


# class to update dictionaries
class Bunch(object):
	def __init__(self, adict):
		self.__dict__.update(adict)


class NonLinearProblemTemplate(object):
	"""
	template for nonlinear problem definition
	"""

	def objective_function(self, prediction):
		raise NotImplementedError

	def allforwards(self, nf, fields):
		"""
		default excection of all forward models.
		Please overwrite this method for threading MPI evaluation.
		
		"""
		for k in range(len(fields.shape[0])):
			out[k] = self.forward(field[k])
	
	def forward(self, field):
		"""
		Overwrite this function to define the foreard models
		
		:param field: realization of phyiscal propetrties
		:type field: numpy array
		:rtype: numpy array
		:returns: values of prediction 
		"""
		raise NotImplementedError


		
# Random Mixing using Whittaker-Shannon interpolation
class RMWSCondSim(object):
	def __init__(self,
				 nonlinearproblem,
				 domainsize=(50,50),           # domainsize
				 covmod='0.01 Nug(0.0) + 0.99 Exp(3.5)',     # spatial covariance model
				 nFields=10,                # number of fields to simulate
				 cp=None,                   # conditioning point coordinates
				 cv=None,                   # conditioning point values
				 le_cp=None,                # <= inequality conditioning point coordinates
				 le_cv=None,                # <= inequality conditioning point values
				 ge_cp=None,                # >= inequality conditioning point coordinates
				 ge_cv=None,                # >= inequality conditioning point values                 
				 optmethod='no_nl_constraints', # optimization method: 'circleopt' or 'no_nl_constraints'
				 p_on_circle=8,				# discretization for the circle
				 minObj=None,               # stopping criteria: min objective function value
				 maxiter=None,              # maximum number of iterations for optimization
				 maxbadcount=10,            # max number of consecutive iteration with less than frac_imp -> stopping criteria
				 frac_imp=0.9975,           # 0.25%

				 ):

		assert isinstance(nonlinearproblem, NonLinearProblemTemplate)
		self.nonlinearproblem = nonlinearproblem

		self.domainsize = list(domainsize)
		self.nFields = nFields
		self.finalFields = []      
		self.covmod = covmod      
		self.method = optmethod
		self.minObj = minObj
		self.maxiter = maxiter
		self.maxbadcount = maxbadcount
		self.frac_imp = frac_imp
		self.p_on_circle = p_on_circle
		
		if cp is None:
			if len(self.domainsize) == 3:
				self.cp = np.atleast_3d(np.array([])).reshape(0,3).astype('int')
			elif len(self.domainsize) == 2:
				self.cp = np.atleast_2d(np.array([])).reshape(0,2).astype('int')
			else:
				self.cp = np.atleast_1d(np.array([])).reshape(0,1).astype('int')
			self.cv = np.array([])
		else:
			self.cp = np.array(cp).astype('int')
			self.cv = np.array(cv)

		if le_cp is None:
			if len(self.domainsize) == 3:
				self.le_cp = np.atleast_3d(np.array([])).reshape(0,3).astype('int')
			elif len(self.domainsize) == 2:
				self.le_cp = np.atleast_2d(np.array([])).reshape(0,2).astype('int')
			else:
				self.le_cp = np.atleast_1d(np.array([])).reshape(0,1).astype('int')
			self.le_cv = np.array([])
		else:
			self.le_cp = np.array(le_cp).astype('int')
			self.le_cv = np.array(le_cv)

		if ge_cp is None:
			if len(self.domainsize) == 3:
				self.ge_cp = np.atleast_3d(np.array([])).reshape(0,3).astype('int')
			elif len(self.domainsize) == 2:
				self.ge_cp = np.atleast_2d(np.array([])).reshape(0,2).astype('int')
			else:
				self.ge_cp = np.atleast_1d(np.array([])).reshape(0,1).astype('int')
			self.ge_cv = np.array([])
		else:
			self.ge_cp = np.array(ge_cp).astype('int')
			self.ge_cv = np.array(ge_cv)

		# simulate unconditional random fields as input for RMWS
		if self.method == 'no_nl_constraints':
			self.n_uncondFields = [np.min((np.max((self.cp.shape[0] + self.le_cp.shape[0] + self.ge_cp.shape[0], 500)), 5000))]
		elif self.method == 'circleopt':
			self.n_uncondFields = [np.min((np.max((self.cp.shape[0] + self.le_cp.shape[0] + self.ge_cp.shape[0], 5000)), 12000))]  
		else:
			raise Exception('Wrong method!')
			
		# self.n_uncondFields = [np.min((np.max((self.cp.shape[0] + self.le_cp.shape[0] + self.ge_cp.shape[0], 5000)), 10000))]
		self.spsim = Specsim.spectral_random_field(domainsize=self.domainsize, covmod=self.covmod)      
		self.uncondFields = np.empty(self.n_uncondFields + self.domainsize, dtype=('float32')) 
		for i in range(self.n_uncondFields[0]):
			f = self.spsim.simnew()
			self.uncondFields[i] = (f - f.mean())/np.std(f)

		self.n_inc_fac = int(np.max([5,(self.cp.shape[0] + self.le_cp.shape[0] + self.ge_cp.shape[0])/2.]))

		# if inequalities -> calculate conditional covariance matrix and
		# conditional mean which are necessary to calculate the conditional 
		# Gaussian pdf of the inequalities
		if ((self.le_cp.shape[0] != 0) | (self.ge_cp.shape[0] != 0)):
			m = np.concatenate((self.ge_cp,self.le_cp,self.cp))
			dm = sp.distance_matrix(m,m)
			self.ineq_cv = np.copy(np.concatenate((self.ge_cv,self.le_cv)))

			cov = covfun.Covariogram(dm,model=self.covmod)
			cov11 = cov[:self.ineq_cv.shape[0],:self.ineq_cv.shape[0]]
			cov22 = cov[self.ineq_cv.shape[0]:,self.ineq_cv.shape[0]:]
			cov12 = cov[:self.ineq_cv.shape[0],self.ineq_cv.shape[0]:]
			cov21 = cov12.T

			self.cov_cond = cov11 - np.tensordot(cov12,np.tensordot(np.linalg.inv(cov22),cov21,axes=1),axes=1)
			self.inv_covcond = np.linalg.inv(self.cov_cond)
			self.cond_mu = np.tensordot(np.tensordot(cov12,np.linalg.inv(cov22),axes=1),self.cv,axes=1)

	def __call__(self,):		
		# loop over number of required conditional fields
		for simno in range(0,self.nFields):

			#print(simno)

			# if inequalities are present -> replace them by equalities
			# using MCMC (Metropolis-Hastings Random Walk, MHRW_inequality)
			if ((self.le_cp.shape[0] != 0) | (self.ge_cp.shape[0] != 0)):
				self.MHRW_inequality()

			# no inequalities
			else: 
				self.ineq_cv = np.array([])

			# merge equalities and to equalities transformed inequalities
			self.cp_total = np.concatenate((self.ge_cp,self.le_cp,self.cp))
			self.cv_total = np.concatenate((self.ineq_cv,self.cv)) 

			# find weights for the low norm field, quasi interpolation
			self.ix, self.jx = self.generate_indicies()
			
			# if there are no linear constraints
			if self.cp_total.shape[0] == 0:
				self.norm_inner = 0.0
				self.inner_field = np.zeros(self.uncondFields[0].shape)
				numberOfFields = 0

			else:
				weights, self.norm_inner, numberOfFields = self.find_low_norm_weights()
				selectedFields = self.uncondFields[self.random_index(self.ix, numberOfFields)]
				self.inner_field = self.calc_field(weights, selectedFields)

			# find a high norm, homogeneous solution           
			self.filter_indicies(numberOfFields)        # filter indicies to avoid double usage    
			numberCondPoints = self.cv_total.shape[0]
			dof = 1     # don't change this for now

			if self.method == 'no_nl_constraints':
				numberHomogFields = 1
			elif self.method == 'circleopt':
				numberHomogFields = 2   
			else:
				raise Exception('Wrong method!')

			index_gen = self.index_gen(self.jx, numberCondPoints + dof)
			# dict for the homogeneous solution fields
			homogargs = {   'dof':dof, 
							'numberHomogFields':numberHomogFields, 
							'numberCondPoints':numberCondPoints, 
							'index_gen':index_gen
						}

			# transform dict to object
			homogargs = Bunch(homogargs)

			# generate first set of homogeneous fields and add to object
			homogargs = self.generate_homogeneous_fields(homogargs)

			# this is RM without non-linear constraints
			if self.method == 'no_nl_constraints':
				args = {'homogargs':homogargs}
				args = Bunch(args)
				finalField = self.getFinalField(self.noNLconstraints, args)
			
			# with non-linear constraints use RMWS
			elif self.method == 'circleopt':
				# dict for the non-linear constraints   
				nlvar = {   'counter':0,  
							'objmin':self.minObj, 
								}			  
				nlvar = Bunch(nlvar)

				# dict for Whittaker-Shannon
				circlevars = {  'discr':self.p_on_circle, 
								'usf':60
							 }
				circlevars = Bunch(circlevars)

				# dict that combines all other dicts
				args = {    'homogargs':homogargs, 
							'nlvar':nlvar, 
							'circlevars':circlevars
						}
				args = Bunch(args)


				finalField, updatedargs, final_obj= self.getFinalField(self.circleopt, args)


			self.finalFields.append(finalField)
			# np.savetxt('final_objective%s'%simno,final_obj)
		self.finalFields = np.array(self.finalFields)
		print('\n Simulation terminated!')

	def MHRW_inequality(self,):  
		# Metropolis-Hasting Random Walk to tranform
		# the inequalities to equalities, taking the conditional covariance
		# and the conditional mean into account 
		# NEEDS IMPROVEMENT     
		LL = []
		# draw a first guess from the corresponding truncated normals
		for ineqcons in range(self.ge_cv.shape[0]):
			self.ineq_cv[ineqcons] =  st.truncnorm.rvs(self.ge_cv[ineqcons],np.inf,0,1,1)
		for ineqcons in range(self.ge_cv.shape[0],self.ineq_cv.shape[0]):
			self.ineq_cv[ineqcons] = st.truncnorm.rvs(-np.inf,self.le_cv[ineqcons-self.ge_cv.shape[0]],0,1,1)
		LL.append(self.pdf_gauss_ineq(self.ineq_cv))
		Lbest = LL[0]

		mcmc = []
		mcmcsteps = np.min((np.max((self.ineq_cv.shape[0]**3, 20000)), 30000))

		for mc in range(mcmcsteps):
			ineq_fulfilled = False
			while ineq_fulfilled == False:
				can = np.copy(self.ineq_cv)
				can += np.random.uniform(-0.01,0.01,can.shape[0])
				# check that values are within the valid truncated distribution
				if (((can[:self.ge_cv.shape[0]] > self.ge_cv).all()) & ((can[self.ge_cv.shape[0]:] < self.le_cv).all())):
					ineq_fulfilled = True

			LL.append(self.pdf_gauss_ineq(can))

			u = np.random.random()
			if u < min([1.,LL[mc+1]/Lbest]):
				self.ineq_cv = np.copy(can)
				Lbest = LL[mc+1]
				mcmc.append(can)

		self.ineq_cv = mcmc[-1]

	def random_index(self, inds, n):
		return inds[:n]

	def get_at_cond_locations(self, data, cp):
		assert cp.ndim > 1
		dimensions = list(map(lambda x: cp[:,x], range(cp.ndim)))
		fullslice = [slice(None,None)] 
		if data.ndim > cp.ndim:
			return data[ tuple(fullslice + dimensions) ].T
		else:
			return data[ tuple(dimensions) ].T

	def find_low_norm_weights(self,):
		# number of fields used when minimizing norm
		n = self.cp.shape[0] + self.le_cp.shape[0] + self.ge_cp.shape[0] 

		#print( '\n Find low norm solution')
		norm_inner = 666
		while norm_inner > 0.1:

			# increase number of fields used
			n += self.n_inc_fac

			if n > self.n_uncondFields[0]:
				ix,jx = self.add_uncondFields(nF=[1000])       
			
			selectedFields = self.uncondFields[self.random_index(self.ix, n)]
			A = self.get_at_cond_locations(selectedFields, self.cp_total)

			# singular value decomposition
			U,S,V = np.linalg.svd(A)
			c = np.dot(self.cv_total,U)

			# using svd you get directly the solution with the lowest norm
			# but it only works for equalities, thats why we had to transform
			# the inequalities in advance
			norm_inner = np.sum((c/S)**2)

		s = np.sum((c/S)*V.T[:,:S.shape[0]],axis=1)
		return (s, norm_inner, n)

	def add_uncondFields(self,nF=[100]):
	
		addField = np.empty(nF + self.domainsize, dtype=('float32'))
		for i in range(nF[0]):
			s = self.spsim.simnew()
			addField[i] = (s - s.mean())/s.std()
		# add the new fields to the old ones
		self.uncondFields = np.concatenate((self.uncondFields,addField))
		# update self.n_uncondFields
		self.n_uncondFields = [self.uncondFields.shape[0]]

		ix = np.arange(self.ix.max()+1,self.ix.max()+1+nF[0])
		np.random.shuffle(ix)
		self.ix = np.concatenate((self.ix, ix))

		jx = np.arange(self.jx.max()+1,self.jx.max()+1+nF[0])
		np.random.shuffle(jx)
		self.jx = np.concatenate((self.jx, jx))

		return (ix, jx)

	def sim_uncondFields(self,nF=[100]):
		uncondFields = np.empty(nF + self.domainsize, dtype=('float32'))
		for i in range(nF[0]):
			s = self.spsim.simnew()
			uncondFields[i] = (s - s.mean())/s.std()

		return uncondFields

	def generate_indicies(self,):
		ix = np.arange(0,self.n_uncondFields[0])
		jx = np.arange(0,self.n_uncondFields[0])
		np.random.shuffle(ix)
		np.random.shuffle(jx)
		return (ix, jx)

	def calc_field(self, weights, fields):
		return np.tensordot(weights, fields, axes=1)

	def index_gen(self, inds, n):
		indit = iter(inds)
		while True:
			res = list(it.islice(indit, n))
			if len(res) < n:
				ix, jx = self.add_uncondFields(nF=[2000])
				indit = iter(jx)
				yield list(it.islice(indit, n))
			else:
				yield res

	def solve_homog_eqs(self, x):
		n = x.numberCondPoints + x.dof

		indx = next(x.index_gen)
		selectedFields = self.uncondFields[indx]
		
		A = self.get_at_cond_locations(selectedFields, self.cp_total)
		Alhs = A[:x.numberCondPoints, :x.numberCondPoints]
		AlhsInv = np.linalg.inv(Alhs)
		Arhs = A[:x.numberCondPoints, x.numberCondPoints:]
		sol = np.dot(AlhsInv, Arhs).T
		# add negative identity
		idd = np.identity(n-x.numberCondPoints)*-1.
		solidd = np.hstack((sol,idd))
		betas_norm = self.normalize_homogweights(solidd.flatten())
		homogfield = self.calc_field(betas_norm, selectedFields)
		return homogfield

	def generate_homogeneous_fields(self, x):
		homogFields = []
		for i in range(x.numberHomogFields):
			homogfield = self.solve_homog_eqs(x)
			homogFields.append(homogfield)
		homogFields = np.array(homogFields)
		x.homogfields = homogFields
		return x

	def noNLconstraints(self, args):
		hargs = args.homogargs
		klam = (1.- self.norm_inner)**0.5
		finalField = self.inner_field + klam*hargs.homogfields[0]
		return finalField

	def normalize_homogweights(self, weights):  
		betas = weights/np.dot(weights,weights)**0.5		
		return betas

	def filter_indicies(self, n):
		self.jx = np.array(list(set(self.jx).difference(set(self.ix[:n]))))
		np.random.shuffle(self.jx)

	def getFinalField(self, method, args):
		return method(args)

	def get_points_on_circle(self, discr, usf):
		t = np.linspace(0, np.pi*2,(usf*discr)-(usf-1))
		return t

	def get_samplepoints_on_circle(self, discr):
		t_s = np.linspace(0,np.pi*2,discr)
		xsample = np.array((np.cos(t_s),np.sin(t_s)))
		return xsample

	def get_normfield_at_samplepoints(self, i, x, hargs):
		homogfield = self.calc_field(x, hargs.homogfields)
		normField = self.normalize_with_innerField(homogfield)
		return normField
		
	def normalize_with_innerField(self, homogfield):
		klam = (1.- self.norm_inner)**0.5                
		normField = self.inner_field + klam*homogfield
		return normField

	def circleopt(self, args):
		cargs = args.circlevars
		nlargs = args.nlvar
		hargs = args.homogargs

		# because from here only one new field is required
		hargs.numberHomogFields = 1

		xsample = self.get_samplepoints_on_circle(cargs.discr)
		circlediscr = self.get_points_on_circle(cargs.discr, cargs.usf)

		obj = 6666666666666.
		notoptimal = True
		badcount = 0
		opt_obj=np.array([])

		while notoptimal:
			nlargs.counter += 1 

			normFields = []
			for i,x in enumerate(xsample.T[:-1]):
				# calculate all normalized fields at samplepoints
				normFields.append(self.get_normfield_at_samplepoints(i, x, hargs))
			normFields = np.array(normFields)

			# call the forward model using the normalized fields
			self.nlvals = self.nonlinearproblem.allforwards(normFields)

			# add the first one which is the same as the last (cyclic, i.e. same angle) 
			self.nlvals = np.vstack((self.nlvals, self.nlvals[-1]))
		
			# interpolate values from samplepoints on the circle using Whittaker-Shannon interpolation        
			intp_nlvals = []
			for nlv in range(len(self.nonlinearproblem.data)):
				# wrap it around from -2pi to 4pi to avoid funny boundary effects
				x = self.nlvals[:,nlv]
				x = np.concatenate(((x[:-1]),x))
				x = np.concatenate((x,self.nlvals[:,nlv][1:]))
				intp_nlval = self.dofftint(cargs.usf,x)
				intp_nlvals.append(np.array(intp_nlval))
			intp_nlvals = np.array(intp_nlvals).T
	 
			# get objective function from interpolated values
			objinter = self.nonlinearproblem.objective_function(intp_nlvals)

			# check the dimensions
			assert len(objinter) > 1, ('Objective function needs to return {} values!'.format(circlediscr.shape[0]))

			# find optimal solution from interpolated objective function
			ix = np.where(objinter == objinter.min())[0][0]
			xsopt = np.array((np.cos(circlediscr[ix]),np.sin(circlediscr[ix])))

			# and run the forward model for these weights again to obtain
			# the real (non-interpolated) objective function value
			normField = self.get_normfield_at_samplepoints(0, xsopt, hargs) 
			opt_nlvals = self.nonlinearproblem.allforwards(normField.reshape((1, )+ normField.shape))     
			
			# real objective function value at the optimal angle
			curobj = self.nonlinearproblem.objective_function(opt_nlvals)
			
			# check the dimensions
			assert len(curobj) == 1, ('Objective function needs to return ONE value!')

			print('\r', curobj, end='')
			sys.stdout.flush()

			if curobj < obj:
				if curobj/obj > self.frac_imp:
					badcount += 1
				else:
					badcount = 0
				opt_obj=np.concatenate((opt_obj,curobj))
				obj = curobj
				curhomogfield = self.calc_field(xsopt, hargs.homogfields)
				normfield = self.normalize_with_innerField(curhomogfield)    

				# create new field
				hargs = self.generate_homogeneous_fields(hargs)	
				# add current best			
				hargs.homogfields = np.array((curhomogfield, hargs.homogfields[0]))

			else:
				badcount += 1
				opt_obj=np.concatenate((opt_obj,obj))
				# no improvement so stick to previous best and a new one
				curhomogfield = hargs.homogfields[0]
				hargs = self.generate_homogeneous_fields(hargs)
				hargs.homogfields = np.array((curhomogfield, hargs.homogfields[0]))

			# check whether objective function is smaller than predefined minimum
			if obj < nlargs.objmin:
				notoptimal = False  
				finalField = self.normalize_with_innerField(curhomogfield)
				print('\n Stopping criteria reached!')

			# check if we need too many iterations and stop after maxiter
			elif nlargs.counter == self.maxiter:
				notoptimal = False  
				finalField = self.normalize_with_innerField(curhomogfield)
				print('\n Number of max model runs exceeded! --> Take current best solution!')

			elif badcount >= self.maxbadcount:
				notoptimal = False  
				finalField = self.normalize_with_innerField(curhomogfield)
				print('\n Too small improvements in last %i consecutive iterations! --> Take current best solution!'%badcount)
		
		return finalField, args, opt_obj

	def dofftint(self, usf, x):
		n = x.shape[0] 
		res = np.fft.fft(x)

		z = np.zeros(n*usf).astype(complex)
		
		z[:n//2] = res[:n//2]
		z[-n//2:] = res[-n//2:]

		if np.dtype(x[0]) == complex:
			ans = np.fft.ifft(z)*usf
		else:
			ans = np.real(np.fft.ifft(z))*usf

		norig = (n + 2)/3 
		if norig % 2 != 0:
			raise ValueError('Input must be even!')
		horst = np.int(usf * (norig-1))
		ans = ans[horst:horst+horst+1]

		return ans

	def pdf_gauss_ineq(self, x):
		dim = self.cov_cond.shape[0]
		pdf = -0.5 * (np.sum(np.tensordot(x-self.cond_mu, self.inv_covcond,axes=1)*(x-self.cond_mu)))
		maxl = 500
		pdf = np.where(pdf < - maxl, -maxl, pdf)
		pdf = np.where(pdf > maxl, maxl, pdf)

		return np.exp(pdf)

