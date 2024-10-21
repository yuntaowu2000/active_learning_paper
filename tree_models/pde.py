# """
# Author: Zhouzhou Gu and Goutham Gopalakrishna
# This block pde operators used in the model
# """
from tree_models.para import *
import torch
import random

class Training_pde(Environments):

    def __init__(self, para_dict):
        super().__init__(para_dict)
    
    def get_derivs_1order(self,y_pred,x):
        """ Returns the first order derivatives for scalar
        """
        dy_dx = torch.autograd.grad(y_pred, x,
                            create_graph=True,
                            allow_unused=True,
                            grad_outputs=torch.ones_like(y_pred))[0]
        return dy_dx ## Return 'automatic' gradient.
    
    def batch_jacobian(func, x, create_graph=False):
        # create_graph by default was set to be False, for calculating equilibrium variables
        # x in shape (Batch, Length)
        # define the _func_sum_ function due to the peculiar feature about outputs
        #   For m dim with batch size B, with n dim input, it will return B * n * B * m
        # option for jacobian: vectorize = True (experimental stage)
        def _func_sum(x):
            return func(x).sum(dim=0)
        return torch.autograd.functional.jacobian(_func_sum, x, create_graph=create_graph).permute(1,0,2)
    
    def loss_fun_Net1(self,vh,sigq,ETA,Y):
        # This funciton uses for loop to calculate Jacobian and hessian
        # This function is used for Net1 parameterization

        N_bat       = Y.shape[0]
        eta = ETA.clone()
        y   = Y.clone()
        eta.requires_grad_(True)
        y.requires_grad_(True)

        index_sym   = 0
        index_last  = self.n_pop-1


        eta_last    = 1 - torch.sum(eta,dim=1).unsqueeze(1)

        # sigq_pred =  torch.ones_like(sigq(eta,y)) * sig
        sigq_pred =  sigq(eta,y)

        # ref solution in complete market
        # phi         = torch.ones_like(vh(eta,y)) * self.rho_h + (self.gam - 1) * self.beta * 0.02 - 1/2 * self.gam * (self.gam-1) * self.sig**2

        # Construct the consumption wealth ration vectors
        phi_vec     = torch.zeros((N_bat,self.n_pop))
        phi_vec[:,0]= (vh(eta,y)[:,0])
        # phi_vec[:,0]= torch.ones_like(vh(eta,y)[:,0]) * rho_h
       
        # because everything is symmetric, we only need to compute 1 asset and do swaps
        for j in range(1,self.n_pop-1):
            ind         = (0,j)
            indx        = (j,0)
            eta_swapped = eta.clone()
            eta_swapped[:,ind] = eta[:,indx].clone()
            phi_vec[:,j]= vh(eta_swapped,y)[:,0]
            # phi_vec[:,j]= torch.ones_like(vh(eta_swapped,y)[:,0]) * rho_h

        phi_vec[:,self.n_pop-1] = vh(eta,y)[:,1]
        # phi_vec[:,self.n_pop-1] = torch.ones_like(vh(eta,y)[:,1]) * self.rho_e

        phi         = phi_vec
        # phi         = torch.ones_like(phi_vec)* self.rho_h + (self.gam - 1) * self.beta * 0.02 - 1/2 * self.gam * (self.gam-1) * self.sig**2

        omega       = phi * torch.cat((eta,eta_last),axis = 1)
        q_pred      = (y+1/phi_fric)/(torch.sum(omega,dim=1).unsqueeze(1) + 1/phi_fric)
        c_pred      = omega * q_pred
        xi          = c_pred**(-self.gam)
        
        #recursive utility
        a_e = eta_last*q_pred
        V_e = vh(eta,y)[:,1]**(-self.gam)*a_e**(1-self.gam)/(1-self.gam)
        if ies>1.0:
            denominator = ((1-self.gam)*V_e)**(1/(1-self.gam))
            g_e = (1-self.gam)/(1-1/ies)*(c_pred[:,self.n_pop-1]/denominator)**(1-1/ies)
            f_e = -self.rho_e *(1-self.gam)/(1-1/ies)*V_e + g_e
            ge_Ve = self.get_derivs_1order(g_e,V_e).unsqueeze(1)
        else:
            f_e = (1-self.gam)*self.rho_e*V_e*(torch.log(c_pred[:,self.n_pop-1])-1/(1-self.gam)*torch.log((1-self.gam)*V_e))
            fe_Ve = self.get_derivs_1order(f_e,V_e).unsqueeze(1)
        fe_ce = xi[:,-1] #note that f_c = v_a by envelope theorem
        

        iota_pred = (q_pred-1)/phi_fric
        invest_fun = torch.log(iota_pred*phi_fric+1)/phi_fric
        drift_K_geo        = invest_fun-delta_K
        # dxi_dy for solving the risk allocation
        dxi_dy_vec  = torch.zeros((N_bat,self.n_pop,1))
        M_mat       = torch.zeros((N_bat,self.n_pop,self.n_pop))
        H_mat_rand  = torch.zeros((N_bat,self.n_pop,self.n_pop))
        H_mat_last  = torch.zeros((N_bat,self.n_pop,self.n_pop))
        H_mat_price = torch.zeros((N_bat,self.n_pop,self.n_pop))

        # do the automatic differentiation for price, and extend the dimension
        dq_deta     = self.get_derivs_1order(q_pred,eta).unsqueeze(2)
        dq_dy       = self.get_derivs_1order(q_pred,y).unsqueeze(2)

        # do the automatic differentiation for SDF
        dxi_i_deta    = self.get_derivs_1order(xi[:,0].unsqueeze(1),eta)
        dxi_i_dy      = self.get_derivs_1order(xi[:,0].unsqueeze(1),y)
        M_mat[:,0,0:self.n_pop -1]  = dxi_i_deta * eta * (sigq_pred **2)/(xi[:,0].unsqueeze(1))
        dxi_dy_vec[:,0,:]           = dxi_i_dy
        for j in range(1,self.n_pop):
            dxi_j_deta  = self.get_derivs_1order(xi[:,j].unsqueeze(1),eta)
            dxi_j_dy    = self.get_derivs_1order(xi[:,j].unsqueeze(1),y)

            M_mat[:,j,0:self.n_pop -1]   = dxi_j_deta * eta * (sigq_pred **2)/(xi[:,j].unsqueeze(1))
            dxi_dy_vec[:,j,:]            = dxi_j_dy


        # fill up the last column of M matrix
        M_mat[:,:,self.n_pop -1] =  1

        RHS = (dxi_dy_vec/xi.unsqueeze(2)) * (self.sig_ay(y) * sigq_pred).unsqueeze(2) 

        # Input the frictions
        for j in range(0,self.n_pop-1):
            M_mat[:,j,j] -=k/eta[:,j]
            RHS[:,j,0]  -= k/eta[:,j]

        # Next, solve the sharp ratio/ risk allocation
        n_vec = torch.bmm(torch.inverse(M_mat),RHS)

        theta_vec   = n_vec[:,0:self.n_pop-1,:]   # The portfolio share
        rp          = -n_vec[:,self.n_pop-1,:]    # The risk-premium
        sig_eta_geo = -theta_vec * sigq_pred.unsqueeze(2)
        sig_eta_ari = sig_eta_geo * eta.unsqueeze(2)

        sig_q_ari   = torch.sum(sig_eta_ari * dq_deta,dim = 1) + torch.sum(dq_dy * self.sig_ay(y).unsqueeze(2),dim = 2)
        sig_q_geo   = sig_q_ari/q_pred
        b_div_a     = theta_vec.squeeze(dim=2)
        sharp_ratio = rp / sig_q_geo       # (rq-rf)/sig^q

        Ito_sigq    = (sig_q_geo - sigq_pred)

        dxi_rand_deta = self.get_derivs_1order(xi[:,index_sym].unsqueeze(1),eta)
        dxi_last_deta = self.get_derivs_1order(xi[:,index_last].unsqueeze(1),eta)
        dxi_rand_dy   = self.get_derivs_1order(xi[:,index_sym].unsqueeze(1),y)
        dxi_last_dy   = self.get_derivs_1order(xi[:,index_last].unsqueeze(1),y)
        d2xi_rand_detadetai = self.get_derivs_1order(dxi_rand_deta[:,0].unsqueeze(1),eta)
        d2xi_last_detadetai = self.get_derivs_1order(dxi_last_deta[:,0].unsqueeze(1),eta)
        d2xi_rand_detady = self.get_derivs_1order(dxi_rand_dy,eta)
        d2xi_last_detady = self.get_derivs_1order(dxi_last_dy,eta)
        d2xi_rand_dy2 = self.get_derivs_1order(dxi_rand_dy,y)
        d2xi_last_dy2 = self.get_derivs_1order(dxi_last_dy,y)
        
        d2q_detadetai   = self.get_derivs_1order(dq_deta[:,0].unsqueeze(1),eta)
        d2q_detady      = self.get_derivs_1order(dq_dy,eta)
        d2q_dy2         = self.get_derivs_1order(dq_dy,y)

        H_mat_rand[:,0,0:self.n_pop -1] = d2xi_rand_detadetai
        H_mat_rand[:,self.n_pop-1,0:self.n_pop -1] = d2xi_rand_detady
        H_mat_rand[:,0:self.n_pop -1,self.n_pop-1] = d2xi_rand_detady
        H_mat_rand[:,self.n_pop -1,self.n_pop-1] = d2xi_rand_dy2.squeeze()

        H_mat_last[:,0,0:self.n_pop -1] = d2xi_last_detadetai
        H_mat_last[:,self.n_pop-1,0:self.n_pop -1] = d2xi_last_detady
        H_mat_last[:,0:self.n_pop -1,self.n_pop-1] = d2xi_last_detady
        H_mat_last[:,self.n_pop -1,self.n_pop-1] = d2xi_last_dy2.squeeze()

        H_mat_price[:,0,0:self.n_pop -1] = d2q_detadetai
        H_mat_price[:,self.n_pop-1,0:self.n_pop -1] = d2q_detady
        H_mat_price[:,0:self.n_pop -1,self.n_pop-1] = d2q_detady
        H_mat_price[:,self.n_pop -1,self.n_pop-1] = d2q_dy2.squeeze()
        for j in range(1,self.n_pop-1):
            d2xi_rand_detadetaj = self.get_derivs_1order(dxi_rand_deta[:,j],eta)
            d2xi_last_detadetaj = self.get_derivs_1order(dxi_last_deta[:,j],eta)
            d2q_detadetaj   = self.get_derivs_1order(dq_deta[:,j].unsqueeze(1),eta)
            H_mat_rand[:,j,0:self.n_pop -1] = d2xi_rand_detadetaj
            H_mat_last[:,j,0:self.n_pop -1] = d2xi_last_detadetaj
            H_mat_price[:,j,0:self.n_pop -1] = d2q_detadetaj
        
        drift_eta_geo   = (y-iota_pred)/q_pred + b_div_a * (sig_q_geo ** 2) - c_pred[:,0:self.n_pop-1]/(eta * q_pred) + sharp_ratio * sig_eta_ari.squeeze()/eta
        drift_eta_geo = drift_eta_geo + death*eta_last/eta
        drift_eta_ari   = drift_eta_geo * eta
        # drift_stats_ari = torch.cat((drift_eta_ari,self.mu_ay(y)),axis =1).unsqueeze(2)
        sig_stats_ari   = torch.cat((sig_eta_ari.squeeze(),self.sig_ay(y)),axis =1).unsqueeze(2)

        # drift_stats_ari_t   = drift_stats_ari.transpose(1,2)
        sig_stats_ari_t     = sig_stats_ari.transpose(1,2)
        drift_price_ari     =   torch.sum(dq_deta.squeeze()* drift_eta_ari.squeeze(),1).unsqueeze(1)\
                               +0.5 * torch.bmm(torch.bmm(sig_stats_ari_t,H_mat_price),sig_stats_ari).squeeze(dim=2)\
                               +dq_dy.squeeze(dim=2) * self.mu_ay(y)
        drift_price_geo     = drift_price_ari/q_pred
        
        rq  = (y-iota_pred)/q_pred + drift_price_geo + drift_K_geo
        rf  = rq - sharp_ratio * sig_q_geo
        

        drift_xi_rand_ari   =   torch.sum(dxi_rand_deta * drift_eta_ari.squeeze(),1).unsqueeze(1)\
                               +0.5 * torch.bmm(torch.bmm(sig_stats_ari_t,H_mat_rand),sig_stats_ari).squeeze(dim=2)\
                               +dxi_rand_dy * self.mu_ay(y)
        
        drift_xi_last_ari   =   torch.sum(dxi_last_deta * drift_eta_ari.squeeze(),1).unsqueeze(1)\
                               +0.5 * torch.bmm(torch.bmm(sig_stats_ari_t,H_mat_last),sig_stats_ari).squeeze(dim=2)\
                               +dxi_last_dy * self.mu_ay(y)
        drift_xi_rand_geo   = drift_xi_rand_ari/xi[:,index_sym].unsqueeze(1)
        drift_xi_last_geo   = drift_xi_last_ari/xi[:,index_last].unsqueeze(1)
        Euler_eq_rand = (-self.rho_h + rf + drift_xi_rand_geo - self.gam*drift_K_geo) #/self.rho_h
        Euler_eq_last = (-self.rho_e + rf + drift_xi_last_geo - self.gam*drift_K_geo) #/self.rho_e
        if ies==1.0: Euler_eq_last = fe_Ve + rf + drift_xi_last_geo - self.gam*drift_K_geo
        else: Euler_eq_last = (-self.rho_e*(1-self.gam)/(1-1/ies) + ge_Ve + rf + drift_xi_last_geo - self.gam*drift_K_geo) #/self.rho_e
       
        eta.requires_grad_(False)
        y.requires_grad_(False)
        return Ito_sigq,Euler_eq_rand,Euler_eq_last
