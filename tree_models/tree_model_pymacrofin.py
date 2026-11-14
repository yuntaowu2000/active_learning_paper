import warnings
warnings.filterwarnings('ignore')
from PyMacroFin.model import macro_model
import numpy as np
import pandas as pd
import time
import PyMacroFin.utilities as util

def define_model(npoints):
    m = macro_model(name='2trees')

    m.set_endog(['q1','q2'], init=[1,1])
    m.set_value(['k1', 'k2'], init=[1,1])
    m.prices = ['q1', 'q2']
    m.set_state(['z'])

    m.params.add_parameter('gamma',5)
    m.params.add_parameter('rho',0.05)
    m.params.add_parameter('mu_y1',0.01)
    m.params.add_parameter('sig_y1',0.01)
    m.params.add_parameter('mu_y2',0.02)
    m.params.add_parameter('sig_y2',0.02)

    m.equation('q1_z = d(q1,z)')
    m.equation('q2_z = d(q2,z)')
    m.equation('q1_zz = d(q1,z,z)')
    m.equation('q2_zz = d(q2,z,z)')
    m.equation('k1_z = d(k1,z)')
    m.equation('k2_z = d(k2,z)')
    m.equation('k1_zz = d(k1,z,z)')
    m.equation('k2_zz = d(k2,z,z)')
    
    m.equation('mu_z_geo = mu_y1 - (z * mu_y1 + (1-z) * mu_y2) + (z * sig_y1 + (1-z) * sig_y2) * (z * sig_y1 + (1-z) * sig_y2 - sig_y1)')
    m.equation('sig_z_geo = sig_y1 - (z * sig_y1 + (1-z) * sig_y2)')
    m.equation('mu_z_ari = mu_z_geo * z')
    m.equation('sig_z_ari = sig_z_geo * z')
    m.equation('mu_1minusz_ari  = -mu_z_ari')
    m.equation('sig_1minusz_ari = -sig_z_ari')
    m.equation('mu_1minusz_geo  = mu_1minusz_ari/(1-z)')
    m.equation('sig_1minusz_geo = sig_1minusz_ari/(1-z)')
    m.equation('mu_q1 = (q1_z * mu_z_ari + 0.5 * q1_zz * sig_z_ari**2)/q1')
    m.equation('mu_q2 = (q2_z * mu_z_ari + 0.5 * q2_zz * sig_z_ari**2)/q2')
    m.equation('sig_q1 = (q1_z * sig_z_ari)/q1')
    m.equation('sig_q2 = (q2_z * sig_z_ari)/q2')
    m.equation('r = rho + gamma * (z*sig_y1 + (1-z)*sig_y2) - gamma * (1 + gamma) / 2 * ((z*sig_y1)**2 + ((1-z)*sig_y2)**2)')
    m.equation('zeta1 = gamma * z * sig_y1')
    m.equation('zeta2 = gamma * (1-z) * sig_y2')
    m.equation('mu_k1 = mu_z_geo - mu_q1 + sig_q1 *(sig_q1 - sig_z_geo)')
    m.equation('mu_k2 = mu_1minusz_geo - mu_q2 + sig_q2 *(sig_q2 - sig_1minusz_geo)')
    m.equation('sig_k1 = sig_z_geo - sig_q1')
    m.equation('sig_k2 = sig_1minusz_geo - sig_q2')

    m.endog_equation('q1-z/k1')
    m.endog_equation('q2-(1-z)/k2')

    # consider the value function r(x)F(x,t)= u(x) +mu(x)F_x+sig(x)^2/2 F_xx + F_t
    # Note that since there is no value variable, the HJB equations here are not used
    m.hjb_equation('mu','z','mu_z_ari') # this sets mu(x) associated to e, using mue
    m.hjb_equation('sig','z','sig_z_ari') # this sets sig(x) associated to e, using sige
    m.hjb_equation('u','k1',0)
    m.hjb_equation('u','k2',0)
    m.hjb_equation('r','k1',"mu_k1")
    m.hjb_equation('r','k2',"mu_k2")


    m.options.ignore_HJB_loop = False
    m.options.max_iter_outer_static = 10
    m.options.min_hjb1 = 2
    m.options.min_hjb2 = 2
    m.options.max_hjb = 5
    m.options.import_guess = False
    m.options.guess_function = False
    m.options.inner_plot = False
    m.options.outer_plot = False
    m.options.final_plot = True
    m.options.n0 = npoints
    m.options.start0 = 0.01
    m.options.end0 = 0.99
    m.options.inner_solver = 'newton-raphson'
    m.options.parallel = False
    m.options.return_solution = True
    m.options.save_solution = True
    return m

npoints = 100
tic = time.time()
m = define_model(npoints)
df = m.run()
toc = time.time()
print('elapsed time: {}'.format(toc-tic))