#!/usr/bin/python
from __future__ import division
from __future__ import print_function
import numpy as np
import numpy.linalg as la
import math
import tensorflow as tf
import scipy.io 

class Generator(object):
    def __init__(self,A,**kwargs):
        self.A = A
        M,N = A.shape
        vars(self).update(kwargs)
        self.x_ = tf.placeholder( tf.float32,(N,None),name='x' )
        self.y_ = tf.placeholder( tf.float32,(M,None),name='y' )
        self.s_ = tf.placeholder( tf.float32,(N,None),name='s' )

class TFGenerator(Generator):
    def __init__(self,**kwargs):
        Generator.__init__(self,**kwargs)
    def __call__(self,sess):
        'generates y,x pair for training'
        return sess.run( ( self.ygen_,self.xgen_, self.sgen_) )

class NumpyGenerator(Generator):
    def __init__(self,**kwargs):
        Generator.__init__(self,**kwargs)

    def __call__(self,sess):
        'generates y,x pair for training'
        return self.p.genYX(self.nbatches,self.nsubprocs)


def bernoulli_gaussian_trial(M=250,N=500,L=1000,pnz=.05,pri=.5,SNR=40):

    matrix = scipy.io.loadmat('randommatrix.mat')
    A=matrix['matrix'].astype(np.float32) 
	
    A_ = tf.constant(A,name='A')
    prob = TFGenerator(A=A,A_=A_,pnz=pnz,SNR=SNR)
    prob.name = 'Bernoulli-Gaussian, random A'

    temp1=tf.random_uniform( (N,L) ) ;
    bernoulli_ = tf.to_float(temp1 < pnz)
    sgen_= tf.to_float(temp1 < pnz*pri)*tf.ones((N,L));
    xgen_ = bernoulli_ * tf.random_normal( (N,L) )
    noise_var = pnz*N/M * math.pow(10., -SNR / 10.)
    ygen_ = tf.matmul( A_,xgen_) + tf.random_normal( (M,L),stddev=math.sqrt( noise_var ) )


    temp2=np.random.uniform( 0,1,(N,L));
    prob.sval =  (temp2< pnz*pri)*np.ones((N,L));
    prob.xval = ((temp2<pnz) * np.random.normal(0,1,(N,L))).astype(np.float32)
    prob.yval = np.matmul(A,prob.xval) + np.random.normal(0,math.sqrt( noise_var ),(M,L))
    temp3=np.random.uniform( 0,1,(N,L));
    prob.sinit =  (temp3< pnz*pri)*np.ones((N,L));
    prob.xinit = ((temp3<pnz) * np.random.normal(0,1,(N,L))).astype(np.float32)
    prob.yinit = np.matmul(A,prob.xinit) + np.random.normal(0,math.sqrt( noise_var ),(M,L))
    prob.sgen_ = sgen_
    prob.xgen_ = xgen_
    prob.ygen_ = ygen_
    prob.noise_var = noise_var

    return prob

def random_access_problem(which=1):
    import raputil as ru
    if which == 1:
        opts = ru.Problem.scenario1()
    else:
        opts = ru.Problem.scenario2()

    p = ru.Problem(**opts)
    x1 = p.genX(1)
    y1 = p.fwd(x1)
    A = p.S
    M,N = A.shape
    nbatches = int(math.ceil(1000 /x1.shape[1]))
    prob = NumpyGenerator(p=p,nbatches=nbatches,A=A,opts=opts,iid=(which==1))
    if which==2:
        prob.maskX_ = tf.expand_dims( tf.constant( (np.arange(N) % (N//2) < opts['Nu']).astype(np.float32) ) , 1)

    _,prob.noise_var = p.add_noise(y1)

    unused = p.genYX(nbatches) # for legacy reasons -- want to compare against a previous run
    (prob.yval, prob.xval) = p.genYX(nbatches)
    (prob.yinit, prob.xinit) = p.genYX(nbatches)
    import multiprocessing as mp
    prob.nsubprocs = mp.cpu_count()
    return prob
