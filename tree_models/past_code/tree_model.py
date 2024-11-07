"""
Author: Goutham G. 
"""
from pyDOE import lhs
import torch
import numpy as np 
import torch.optim as optim
import matplotlib.pyplot as plt
import json
import os
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from matplotlib.ticker import FormatStrFormatter
import torch.nn as nn
import torch.nn.functional as F
np.random.seed(42)
torch.manual_seed(42)
#os.environ["PYTHONHASHSEED"] = "42"
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False
#torch.set_num_threads(1)

# Import required user-defined libraries
from para import *

plot_directory = "./output/plots/"
if not os.path.exists(plot_directory):
    os.makedirs(plot_directory)
    
class Net1(torch.nn.Module):
    def __init__(self, nn_width, nn_num_layers,positive=False,sigmoid=False):
        super(Net1, self).__init__()
        # Initialize the first layer
        layers = [torch.nn.Linear(params['n_trees']-1, params['nn_width']), torch.nn.Tanh()]
        # Add the rest of the layers
            
        for i in range(1, nn_num_layers):
            layers.append(torch.nn.Linear(nn_width, nn_width))
            layers.append(torch.nn.Tanh())
        # Final output layer        
        layers.append(torch.nn.Linear(nn_width, 1))
        
        # Define functions
        self.positive   = positive  # Define positive function
        self.sigmoid    = sigmoid   # Define sigmoid function
        self.net        = torch.nn.Sequential(*layers) # Construct neural net
        
    def forward(self, Z):
        ''' Define the forward pass of the neural network
                - sigmoid is used when parameterizing c/a ratio
                - positive is used for ∂V/∂a > 0 and q > 0
        '''
        output = self.net(Z)
        if self.sigmoid: output = torch.nn.functional.sigmoid(output)
        elif self.positive: output = torch.nn.functional.softplus(output)
        return output

class Training_Sampler():
    def __init__(self, params):
        self.params = params

    def sample(self,N):
        ''' Construct share by renormalization
            - No active sampling at the moment.
        '''
        # Construct a by latin hypercube sampling. If there are
        # N trees, then construct N-1 shares 
        #Z       = 0.05+ (0.95-0.05)*np.random.rand(N)
        Z      = np.random.rand(N)
        
        return Z.reshape((N, 1))
    

class Training_pde(Environments):

    def __init__(self, params):
        super().__init__(params)
    
    def get_derivs_1order(self,y_pred,x):
        """ Returns the first order derivatives,
            Automatic differentiation used
        """
        dy_dx = torch.autograd.grad(y_pred, x,
                            create_graph=True,
                            grad_outputs=torch.ones_like(y_pred))[0]
        return dy_dx ## Return 'automatic' gradient.
    
    def batch_jacobian(func, x, create_graph=False):
        ''' Compute the jacobian of a function for a batch of inputs
        '''
        def _func_sum(x):
            return func(x).sum(dim=0)
        return torch.autograd.functional.jacobian(_func_sum, x, create_graph=create_graph).permute(1,0,2)
    
    def loss_fun_Net1(self,kappa1_nn,kappa2_nn,Z,return_=True):
        ''' Loss function for the neural network
        Remark: two trees is a one state variable problem. 
            - kappa1: kappa(Z)
            - kappa2: kappa(Z)
            - Z: share
        '''
        z   = Z.clone()
        z.requires_grad_(True)
        
        kappa1 = kappa1_nn(z)
        kappa2 = kappa2_nn(z)
        q1 = z / kappa1
        q2 = (1-z)/(kappa2 )

        # Automatic differentiation for q1 and q2
        q1_z        = self.get_derivs_1order(q1,z)
        q2_z        = self.get_derivs_1order(q2,z)
        q1_zz       = self.get_derivs_1order(q1_z,z)
        q2_zz       = self.get_derivs_1order(q2_z,z)

        kappa1_z    = self.get_derivs_1order(kappa1,z)
        kappa2_z    = self.get_derivs_1order(kappa2,z)
        kappa1_zz   = self.get_derivs_1order(kappa1_z,z)
        kappa2_zz   = self.get_derivs_1order(kappa2_z,z)

        
        # Compute dynamics of z
        mu_z_geo  = self.params['mu_y1'] - (z*self.params['mu_y1'] + (1-z)*self.params['mu_y2']) +\
                    (z*self.params['sig_y1'] + (1-z)*self.params['sig_y2']) * ((z*self.params['sig_y1'] + (1-z)*self.params['sig_y2'])  - self.params['sig_y1'])
        
        sig_z_geo =  self.params['sig_y1'] - (z*self.params['sig_y1'] + (1-z)*self.params['sig_y2']) 
        
        mu_z_ari    = mu_z_geo * z 
        sig_z_ari   = sig_z_geo * z

        # Compute dynamics of 1-z (required for computation of kappa_2 dynamics)
        mu_1minusz_ari  = -mu_z_ari
        sig_1minusz_ari = -sig_z_ari
        mu_1minusz_geo  = mu_1minusz_ari/(1-z)
        sig_1minusz_geo = sig_1minusz_ari/(1-z)

        mu_q1       = (q1_z * mu_z_ari + 0.5 * q1_zz * sig_z_ari**2)/q1
        mu_q2       = (q2_z * mu_z_ari + 0.5 * q2_zz * sig_z_ari**2)/q2
        sig_q1      = (q1_z * sig_z_ari)/q1
        sig_q2      = (q2_z * sig_z_ari)/q2

        r           = self.params['rho'] + self.params['gamma'] * (z*self.params['mu_y1']  + (1-z)*self.params['mu_y2']) -\
                        self.params['gamma'] * (1+self.params['gamma'])/2*(z**2 *self.params['sig_y1']**2  + (1-z)**2 *self.params['sig_y2']**2)
        zeta_1      = self.params['gamma'] * z * self.params['sig_y1']
        zeta_2      = self.params['gamma'] * (1-z) * self.params['sig_y2']
        
        mu_kappa1   = mu_z_geo - mu_q1 + sig_q1 *(sig_q1 - sig_z_geo) 
        mu_kappa2   = mu_1minusz_geo - mu_q2 + sig_q2 *(sig_q2 - sig_1minusz_geo)
        sig_kappa1  = sig_z_geo - sig_q1
        sig_kappa2  = sig_1minusz_geo - sig_q2

        hjb_kappa1          =  kappa1_z * mu_z_ari + 0.5 * kappa1_zz * sig_z_ari**2 - mu_kappa1 * kappa1
        hjb_kappa2          =  kappa2_z * mu_z_ari + 0.5 * kappa2_zz * sig_z_ari**2 - mu_kappa2 * kappa2
        consistency_kappa1  =  kappa1_z * sig_z_ari - sig_kappa1 * kappa1
        consistency_kappa2  =  kappa2_z * sig_z_ari - sig_kappa2 * kappa2
        l2_regularization   = torch.mean(kappa1**2 + kappa2**2)

        # Store variables
        self.kappa1, self.kappa2, self.q1, self.q2, self.zeta1, self.zeta2, self.r = kappa1, kappa2, q1, q2, zeta_1, zeta_2, r
        self.mu_z_geo, self.sig_z_geo, self.mu_z_ari, self.sig_z_ari = mu_z_geo, sig_z_geo, mu_z_ari, sig_z_ari
        self.mu_q1, self.mu_q2, self.sig_q1, self.sig_q2 = mu_q1, mu_q2, sig_q1, sig_q2
        self.mu_kappa1, self.mu_kappa2, self.sig_kappa1, self.sig_kappa2 = mu_kappa1, mu_kappa2, sig_kappa1, sig_kappa2
        

        if return_: return hjb_kappa1, hjb_kappa2, consistency_kappa1, consistency_kappa2, l2_regularization
    
if __name__ == '__main__':
    TP = Training_pde(params)
    TS = Training_Sampler(params)

    # Initialize neural networks
    kappa1_nn = Net1(params['nn_width'], params['nn_num_layers'],positive=True,sigmoid=False).to(device)
    kappa2_nn = Net1(params['nn_width'], params['nn_num_layers'],positive=True,sigmoid=False).to(device)
    kappa2_nn.load_state_dict(kappa1_nn.state_dict())
    # They should start with the same weights

    # Initialize neural nets to store last best model
    Best_model_kappa1   = Net1(params['nn_width'],params['nn_num_layers'],positive=True,sigmoid=False).to(device) 
    Best_model_kappa2   = Net1(params['nn_width'],params['nn_num_layers'],positive=True,sigmoid=False).to(device)
    Best_model_kappa2.load_state_dict(Best_model_kappa1.state_dict())

    para_nn = list(kappa1_nn.parameters()) + list(kappa2_nn.parameters())

    # Set hyperparameters
    optimizer       = optim.Adam(para_nn, lr=params['lr'])
    epochs          = 1001
    min_loss        = float('Inf')
    loss_data_count = 0
    for epoch in range(1,epochs):
        loss_data_count +=1
        Z = TS.sample(params['batch_size'])
        z_tensor = torch.tensor(Z, dtype=torch.float32, requires_grad=True).to(device)
        lhjb_kappa1_, lhjb_kappa2_, lconsistency_kappa1_, lconsistency_kappa2_,lreg_ = TP.loss_fun_Net1(kappa1_nn,kappa2_nn,z_tensor)

        lhjb_kappa1                 = torch.sum(torch.abs(lhjb_kappa1_))
        lhjb_kappa2                 = torch.sum(torch.abs(lhjb_kappa2_))
        lconsistency_kappa1         = torch.sum(torch.abs(lconsistency_kappa1_))
        lconsistency_kappa2         = torch.sum(torch.abs(lconsistency_kappa2_))
        lreg                        = torch.sum(torch.abs(lreg_))

        loss = lhjb_kappa1 + lhjb_kappa2 + lconsistency_kappa1 + lconsistency_kappa2 # + lreg
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(para_nn, 1)
  
        optimizer.step()
        loss_val = loss.item()
        if (loss_val < min_loss):
            min_loss = loss_val
            Best_model_kappa1.load_state_dict(kappa1_nn.state_dict())
            Best_model_kappa2.load_state_dict(kappa2_nn.state_dict())

            print( (epoch, lhjb_kappa1.item(), lhjb_kappa2.item(), lconsistency_kappa1.item(), lconsistency_kappa2.item()))

        if epoch % 1000 == 0:
            print('Epoch: %d, Loss: %.3f' % (epoch, loss_val))
            # Load last best model as the final neural network model
            kappa1_nn.load_state_dict(Best_model_kappa1.state_dict())
            kappa2_nn.load_state_dict(Best_model_kappa2.state_dict())
            # Save data
            Z                = np.linspace(0,1,params['batch_size']).reshape(-1,1)
            z_tensor         = torch.tensor(Z, dtype=torch.float32, requires_grad=True).to(device)
            
            TP  = Training_pde(params)
            TP.loss_fun_Net1(kappa1_nn,kappa2_nn,z_tensor,False)
            kappa1, kappa2, q1, q2, zeta1, zeta2, r = TP.kappa1, TP.kappa2, TP.q1, TP.q2, TP.zeta1, TP.zeta2, TP.r
            mu_z_geo, sig_z_geo, mu_z_ari, sig_z_ari = TP.mu_z_geo, TP.sig_z_geo, TP.mu_z_ari, TP.sig_z_ari
            mu_q1, mu_q2, sig_q1, sig_q2 = TP.mu_q1, TP.mu_q2, TP.sig_q1, TP.sig_q2
            mu_kappa1, mu_kappa2, sig_kappa1, sig_kappa2 = TP.mu_kappa1, TP.mu_kappa2, TP.sig_kappa1, TP.sig_kappa2
            
 
            fig, ax = plt.subplots(3,2,figsize=(16,9), num=1)
            ax[0,0].plot(Z,kappa1.detach().cpu().numpy(),label=r'$\kappa_1$')
            ax[0,0].plot(Z,kappa2.detach().cpu().numpy(),label=r'$\kappa_2$')
            ax[0,0].legend()
            
            
            ax[0,1].plot(Z,q1.detach().cpu().numpy(),label=r'$q_1$')
            ax[0,1].plot(Z,q2.detach().cpu().numpy(),label=r'$q_2$')
            ax[0,1].legend()
            

            ax[1,0].plot(Z,zeta1.detach().cpu().numpy(),label=r'$\zeta_1$')
            ax[1,0].plot(Z,zeta2.detach().cpu().numpy(),label=r'$\zeta_2$')
            ax[1,0].legend()
            

            ax[1,1].plot(Z,r.detach().cpu().numpy(),label=r'$r$')
            ax[1,1].set_title('r')

            ax[2,0].plot(Z,mu_z_geo.detach().cpu().numpy(),label=r'$\mu^z$')
            ax[2,0].plot(Z,sig_z_geo.detach().cpu().numpy(),label=r'$\sigma^z$')
            ax[2,0].legend()
            ax[2,0].set_xlabel('z')

            ax[2,1].plot(Z,mu_z_ari.detach().cpu().numpy(),label=r'$\mu_z$')
            ax[2,1].plot(Z,sig_z_ari.detach().cpu().numpy(),label=r'$\sigma_z$')
            ax[2,1].legend()
            ax[2,1].set_xlabel('z')

            
            name = 'equilibrium.jpg'
            plt.savefig(os.path.join(plot_directory, name),bbox_inches='tight',dpi=300)
            plt.show()
            plt.close('all')

    print('Training complete')


