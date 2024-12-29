import logging
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from TiDHy.utils.utils import requires_grad
from tqdm.auto import tqdm

def soft_thresholding(r, lmda):
    """Non-negative proxial gradient for L1 regularization."""
    with torch.no_grad():
        rtn = F.relu(torch.abs(r) - lmda) * torch.sign(r)
    return rtn.data

def soft_max(r):
    with torch.no_grad():
        rtn = F.softmax(r,dim=-1)
    return rtn.data

def heavyside(r,value):
    with torch.no_grad():
        rtn = torch.heaviside(r,values=value)
    return rtn.data

def poissonLoss(predicted, observed):
    """Custom loss function for Poisson model."""
    return (predicted-observed*torch.log(predicted))

def selu(r):
    with torch.no_grad():
        rtn = F.selu(r)
    return rtn.data

def selu(r):
    with torch.no_grad():
        rtn = F.silu(r)
    return rtn.data

class TiDHy(nn.Module):
    def __init__(self, params, device, show_progress=True,show_inf_progress=False):
        super(TiDHy, self).__init__()
        self.__dict__.update(params)
        
        # spatial: p(I | r)
        ##### Potential to use Binary Cross Entropy Loss  #####
        if params.loss_type == 'BCE':
            self.spatial_decoder = nn.Sequential(nn.Linear(params.r_dim, params.input_dim, bias=True),
                                                nn.Sigmoid())
            for p in self.spatial_decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_normal_(p)
                    
        ##### Non Linear Decoder #####
        if params.nonlin_decoder:
            self.spatial_decoder = nn.Sequential(nn.Linear(params.r_dim, params.hyper_hid_dim, bias=True),
                                                nn.LayerNorm(params.hyper_hid_dim),
                                                nn.ELU(),
                                                nn.Linear(params.hyper_hid_dim, params.hyper_hid_dim),
                                                nn.Linear(params.hyper_hid_dim, params.input_dim),)
        else:
            self.spatial_decoder = nn.Sequential(nn.Linear(params.r_dim, params.input_dim, bias=True),)
        #### Initialize Weights #####
        nn.init.xavier_normal_(self.spatial_decoder[0].weight)
        
        # temporal: p(r_t | r_(t-1), r2)
        if params.low_rank_temp:
            self.temporal = nn.Parameter(torch.randn((params.mix_dim, params.r_dim,2), requires_grad=True))
        else:
            self.temporal = nn.Parameter(torch.randn(params.mix_dim, params.r_dim*params.r_dim, requires_grad=True))
        nn.init.orthogonal_(self.temporal)
        if params.dyn_bias:
            self.temporal_bias = nn.Parameter(torch.randn(1, params.r_dim, requires_grad=True))
            nn.init.xavier_normal_(self.temporal_bias)
            
        ##### Initialize Hypernetwork #####
        self.hypernet = nn.Sequential(
            nn.Linear(params.r2_dim, params.hyper_hid_dim),
            nn.LayerNorm(params.hyper_hid_dim),
            nn.ELU(),
            nn.Linear(params.hyper_hid_dim, params.hyper_hid_dim),
            nn.Linear(params.hyper_hid_dim, params.mix_dim+params.r_dim,bias=False) if params.dyn_bias else nn.Linear(params.hyper_hid_dim, params.mix_dim,bias=True), # weight and +1/2x for bias
            nn.ReLU(), 
        )
        for p in self.hypernet.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
        
        if params.use_r2_decoder:
            self.R2_Decoder = nn.Sequential(
                    nn.Linear(params.r2_dim + params.r_dim, params.r2_decoder_hid_dim),
                    nn.LayerNorm(params.r2_decoder_hid_dim),
                    nn.ELU(),
                    nn.Linear(params.r2_decoder_hid_dim, params.r2_decoder_hid_dim),
                    nn.Linear(params.r2_decoder_hid_dim, params.r2_dim), 
                )
            for p in self.R2_Decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_normal_(p)
        
        self.loss_weights = None 
        self.loss_weights_inf = None

        ##### Define Loss #####
        if params.loss_type == 'Poisson':
            self.spat_loss = poissonLoss
        elif params.loss_type == 'BCE':
            self.spat_loss = nn.BCELoss(reduction='none')
        else:
            self.spat_loss = nn.MSELoss(reduction='none')

        # hyperparams
        self.device = device
        self.r_dim = params.r_dim
        self.r2_dim = params.r2_dim
        self.mix_dim = params.mix_dim
        self.dyn_bias = params.dyn_bias
        self.low_rank_temp = params.low_rank_temp
        self.iters = 0
        ##### Learning Rates #####
        self.lr_r = params.lr_r
        self.lr_r2 = params.lr_r2
        self.lr_weights = params.lr_weights
        self.lr_weights_inf = params.lr_weights_inf
        self.temp_weight = params.temp_weight
        ##### Regularization params #####
        self.lmda_r = params.lmda_r
        self.lmda_r2 = params.lmda_r2
        self.weight_decay = params.weight_decay
        self.cos_eta = params.cos_eta
        self.L1_alpha = params.L1_alpha 
        self.L1_inf_w = params.L1_inf_w
        self.L1_inf_r2 = params.L1_inf_r2
        self.L1_inf_r = params.L1_inf_r
        self.L1_alpha_inf = params.L1_alpha_inf
        self.L1_alpha_r2 = params.L1_alpha_r2
        self.grad_alpha = params.grad_alpha
        self.grad_alpha_inf = params.grad_alpha_inf
        self.clip_grad = params.clip_grad
        self.grad_norm_inf = params.grad_norm_inf

        self.max_iter = params.max_iter
        self.tol = params.tol
        self.r2_state = None
        self.converge_dyn = False
        self.normalize_spatial = params.normalize_spatial
        self.normalize_temporal = params.normalize_temporal
        self.stateful = params.stateful
        self.show_progress = show_progress
        self.show_inf_progress = show_inf_progress
        self.learning_rate_gamma = params.learning_rate_gamma
        self.batch_converge = params.batch_converge
        self.spat_weight = params.spat_weight

    def forward(self, X):
        """Forward pass"""
        batch_size = X.size(0)
        T = X.size(1)
        r, r2 = self.init_code_(batch_size)
        r_first = r.detach().clone()
        spatial_loss_rhat = self.spat_loss(self.spatial_decoder(r), X[:, 0]).view(batch_size, -1).sum(1).mean(0)
        spatial_loss_rbar = torch.zeros_like(spatial_loss_rhat)
        temp_loss = 0
        r2_losses = 0
        if (self.show_progress):
            log = logging.getLogger(__name__)
            t_range = tqdm(range(1, T),leave=False,dynamic_ncols=True)
        else:
            t_range = range(1, T)
        for t in t_range:
            r_p = r.detach().clone()
            r2_p = r2.detach().clone()
            r, r2, r2_loss = self.inf(X[:, t], r_p, r2.detach().clone())
            # learning
            x_hat = self.spatial_decoder(r)
            r_bar, _, _ = self.temporal_prediction(r_p, r2) 
            x_bar = self.spatial_decoder(r_bar) 
            # loss
            if self.use_r2_decoder:
                r2_hat = self.R2_Decoder(torch.cat([r_p,r2_p],dim=-1))
                r2_losses += torch.pow(r2 - r2_hat, 2).view(batch_size, -1).sum(1).mean(0)
            spatial_loss_rhat += self.spat_loss(x_hat, X[:, t]).view(batch_size, -1).sum(1).mean(0) ##### Spatial Loss due to x_hat #####
            spatial_loss_rbar += self.spat_loss(x_bar, X[:, t]).view(batch_size, -1).sum(1).mean(0) ###### Spatial Loss due to x_bar #####
            temp_loss += torch.pow(r - r_bar,2).view(batch_size, -1).sum(1).mean(0)  
            
            if self.show_progress:
                logging.info('tstep:{}, '.format(t) + self.log_msg)
        if (self.training==True) & (self.stateful==True):
            self.r2_state = r2.detach().clone()
            self.r_state = r.detach().clone()
        ##### Clear memory #####
        torch.cuda.empty_cache()
        return spatial_loss_rhat, spatial_loss_rbar, self.temp_weight * temp_loss, r2_losses, r_first, r2.detach().clone()

    def inf(self, x, r_p, r2):
        """Inference step: p(r_t, r2_t | x_t, r_t-1, r2_t-1)"""

        batch_size = x.size(0)
        r, _ = self.init_code_(batch_size)
        r2.requires_grad = True
        # fit r
        optim_r = torch.optim.SGD([r], self.lr_r, nesterov=True, momentum=0.9)
        optim_r2 = torch.optim.SGD([r2], self.lr_r2, nesterov=True, momentum=0.9)
        optimizers = [optim_r, optim_r2]
        milestones_r = [block for i, block in enumerate(range(self.max_iter//10,self.max_iter,self.max_iter//10))]
        milestones_r2 = [block for i, block in enumerate(range(self.max_iter//5,self.max_iter,self.max_iter//10))]
        scheduler = [MultiStepLR(optim_r, milestones=milestones_r, gamma=.5)] 
        converged = False
        i = 0
        r2_loss= []
        requires_grad(self.parameters(), False)
        while (converged==False) & (i < self.max_iter):
            old_r = r.detach().clone()
            old_r2 = r2.detach().clone()
            # prediction
            x_bar = self.spatial_decoder(r)
            r_bar, V_t, w = self.temporal_prediction(r_p, r2)
            # prediction error
            if self.use_r2_decoder:
                r2_bar = self.R2_Decoder(torch.cat([old_r,old_r2],dim=-1))
                r2_loss = torch.pow(r2 - r2_bar, 2).view(batch_size, -1).sum(1).mean(0)
            spatial_loss = self.spat_loss(x_bar,x).view(batch_size, -1).sum(1).mean(0)
            temporal_loss =  torch.pow(r - r_bar, 2).view(batch_size, -1).sum(1).mean(0)
            # update latent activity
            ##### Grad Norm if Needed #####
            if (self.grad_norm_inf):
                losses = torch.stack([spatial_loss, temporal_loss, temporal_loss])
                weights, losses, optimizers, r, r2 = self.grad_norm_inf_step(r, r2, losses, optimizers, i)
            else:
                inf_W_sparcity  = self.L1_inf_w*torch.norm(w,p=1)
                inf_r2_sparcity = self.L1_inf_r2*torch.norm(r2,p=1)
                inf_r_sparcity  = self.L1_inf_r*torch.norm(r,p=1)
                loss = spatial_loss + self.temp_weight * (temporal_loss) + inf_W_sparcity + inf_r2_sparcity + inf_r_sparcity
                if self.use_r2_decoder:
                    loss += r2_loss
                else:
                    r2_loss.append(temporal_loss.item())
                loss.backward()
                for optim in optimizers: optim.step()
                for optim in optimizers: optim.zero_grad()
            for sched in scheduler: sched.step()

            ##### shrinkage #####
            r2.data = soft_thresholding(r2, self.lmda_r2)
            r.data = soft_thresholding(r, self.lmda_r)

            # convergence
            with torch.no_grad():
                if self.batch_converge:
                    r2_converge = torch.linalg.norm(r2 - old_r2) / (torch.linalg.norm(old_r2) + 1e-16)
                    r_converge = torch.linalg.norm(r - old_r) / (torch.linalg.norm(old_r) + 1e-16)
                else:
                    r2_converge = torch.linalg.norm(r2 - old_r2,dim=-1) / (torch.linalg.norm(old_r2,dim=-1) + 1e-16)
                    r_converge = torch.linalg.norm(r - old_r,dim=-1) / (torch.linalg.norm(old_r,dim=-1) + 1e-16)
            converged = torch.all(r_converge < self.tol) and torch.all(r2_converge < self.tol)
            ##### Show Inference Progress #####
            if self.show_inf_progress:
                self.log_msg = 'inf_it:{}, r2_conv:{:.03f}, r_conv:{:.03f}, spat_loss:{:.02f}, temp_loss:{:.02f} '.format(i,torch.sum(r2_converge > self.tol), torch.sum(r_converge > self.tol),spatial_loss.item(),temporal_loss.item(),temporal_loss.item())
                if self.grad_norm_inf:
                    self.log_msg += 'gnorm_w:{}, '.format(list(weights.detach().cpu().numpy()))
                logging.info(self.log_msg)
            i += 1
        ##### Show Progress for time t #####
        if self.show_progress:
            self.log_msg = 'inf_it:{}, '.format(i)
            if self.grad_norm_inf:
                self.log_msg += 'gnorm_w:{}, '.format(list(weights.detach().cpu().numpy()))
        self.converge_warning_(i, 'r/r2 did not converge: r2_conv:{}, r_conv:{}, '.format(torch.sum(r2_converge>self.tol).item(),torch.sum(r_converge>self.tol).item()))
        requires_grad(self.parameters(), True)
        return r.detach().clone(), r2.detach().clone(), r2_loss

    def inf_first_step(self, x):
        """First step inference: p(r_1 | x_1) (no temporal prior)"""

        batch_size = x.size(0)
        r, _ = self.init_code_(batch_size)
        optim = torch.optim.AdamW([r], self.lr_r)
        converged = False
        i = 0
        requires_grad(self.parameters(), False)
        while not converged and i < self.max_iter:
            old_r = r.detach().clone()
            # prediction
            x_bar = self.spatial_decoder(r)
            # prediction error
            loss = self.spat_loss(x_bar,x).view(batch_size, -1).sum(1).mean(0)
            # update neural activity
            loss.backward()
            optim.step()
            optim.zero_grad()
            # convergence
            with torch.no_grad():
                # print(torch.linalg.norm(r - old_r) / (torch.linalg.norm(old_r) + 1e-16))
                r_converge = torch.linalg.norm(r - old_r,dim=-1) / (torch.linalg.norm(old_r,dim=-1) + 1e-16)
                converged = torch.all(r_converge < self.tol)
            i += 1
        self.converge_warning_(i, "first step r did not converge")
        requires_grad(self.parameters(), True)
        return r.detach().clone()

    def temporal_prediction(self, r, r2):
        ##### Batch Size #####
        batch_size = r.size(0)
        
        wb = self.hypernet(r2)
        ##### Split if bias was added #####
        w = wb[:, :self.mix_dim]
        b = wb[:, self.mix_dim:]
        ##### Low rank temporal prediction #####
        if self.low_rank_temp:
            Vk = torch.bmm(self.temporal,self.temporal.permute(0,2,1)).reshape(self.mix_dim,-1)
            V_t = torch.matmul(w, Vk).reshape(batch_size, self.r_dim, self.r_dim)
        else:
            V_t = torch.matmul(w, self.temporal).reshape(batch_size, self.r_dim, self.r_dim)
        
        ###### Handle Bias if set #####
        if self.dyn_bias:
            r_hat = (torch.bmm(V_t, r.unsqueeze(2)) + (b*self.temporal_bias).unsqueeze(2)).squeeze(dim=-1) # (batch_size, r_dim)
        else:
            r_hat = (torch.bmm(V_t, r.unsqueeze(2))).squeeze(dim=-1) # (batch_size, r_dim)
        return r_hat, V_t, w

    def init_code_(self, batch_size):
        r = torch.zeros((batch_size, self.r_dim), requires_grad=True, device=self.device)
        r2 = torch.zeros((batch_size, self.r2_dim),requires_grad=True, device=self.device)
        return r, r2

    def normalize(self):
        with torch.no_grad():
            if self.normalize_spatial:
                self.spatial_decoder[0].weight.data = F.normalize(self.spatial_decoder[0].weight.data, dim=0)
            if (self.normalize_temporal):
                self.temporal.data = F.normalize(self.temporal.data, dim=-1)

    def converge_warning_(self, i, msg):
        if i >= self.max_iter:
            logging.info(msg)
            
    def grad_norm_inf_step(self,r,r2,loss,optimizer,iters):
        if iters == 0:
            # init weights
            weights = torch.ones_like(loss)
            weights = torch.nn.Parameter(weights)
            T = weights.detach().sum() # sum of weights
            # set optimizer for weights
            optimizer.append(torch.optim.AdamW([weights], lr=self.lr_weights_inf))
            # set L(0)
            if torch.any(loss==0):
                loss[loss==0] = 1e-6
            self.l0_inf = loss.detach()
        else:
            weights = self.loss_weights_inf
            T = weights.detach().sum() # sum of weights
            weights = torch.nn.Parameter(weights)
            optimizer[-1] = torch.optim.AdamW([weights], lr=self.lr_weights_inf)
        # compute the weighted loss
        weighted_loss = weights @ loss
        # clear gradients of network
        for opt in optimizer[:-1]: opt.zero_grad()
        # backward pass for weigthted task loss
        weighted_loss.backward(retain_graph=True)
        # compute the L2 norm of the gradients for each task
        gw = []
        model_latents = [[r,r2], r, r2]
        for i in range(len(loss)):
            if i == 0:
                dl_r,dl_r2 = torch.autograd.grad(weights[i]*loss[i], model_latents[i], retain_graph=True, create_graph=True)
                dl_norm = torch.norm(dl_r)+torch.norm(dl_r2)
                    # torch.tensor([torch.norm(dl[n]) for n in range(len(model_latents[i]))],requires_grad=True)).to(self.device)
                gw.append(dl_norm)
            else:
                dl = torch.autograd.grad(weights[i]*loss[i], model_latents[i], retain_graph=True, create_graph=True)[0]
                gw.append(torch.norm(dl))
        gw = torch.stack(gw)
        # compute loss ratio per task
        loss_ratio = loss.detach() / self.l0_inf
        # compute the relative inverse training rate per task
        rt = loss_ratio / loss_ratio.mean()
        # compute the average gradient norm
        gw_avg = gw.mean().detach()
        # compute the GradNorm loss
        constant = (gw_avg * rt ** self.grad_alpha_inf).detach()
        gradnorm_loss = torch.abs(gw - constant).sum()
        # backward pass for GradNorm
        optimizer[-1].zero_grad()
        gradnorm_loss.backward()

        # update model weights and loss weights
        for opt in optimizer: opt.step()
        ##### renormalize weights #####
        # if torch.any(weights < 0):
        #     weights = F.relu(weights,threshold=0) + 1e-6
        weights = torch.exp(weights)
        weights = (weights / weights.sum() * T).detach()
        self.loss_weights_inf = weights
        # update iters
        return weights.detach(), loss, optimizer, r, r2

    def evaluate_record(self, data_batch):
        """Forward pass for evaluation"""
        batch_size = data_batch.size(0)
        T = data_batch.size(1)
        input_dim = data_batch.size(2)
        # saving values
        I_bar = torch.zeros((batch_size, T, input_dim))                # Input prediction from Dynamics
        I_hat = torch.zeros((batch_size, T, input_dim))                # Input prediction from Inference
        I = torch.zeros((batch_size, T, input_dim))                    # True input
        R_bar = torch.zeros((batch_size, T, self.r_dim))               # Latent prediction from dynamics 
        R_hat = torch.zeros((batch_size, T, self.r_dim))               # Latent prediction from Inference
        R2_hat = torch.zeros((batch_size, T, self.r2_dim))             # Higher order latent prediction from Inference
        W = torch.zeros((batch_size, T, self.mix_dim))                 # Mixture weights
        temp_loss = torch.zeros((batch_size, T, self.r_dim))           # Temporal dynamics loss
        spatial_loss_rhat = torch.zeros((batch_size, T, self.input_dim))    # Reconstruction Loss
        spatial_loss_rbar = torch.zeros((batch_size, T, self.input_dim))    # Reconstruction Loss
        if self.dyn_bias:
            b = torch.zeros((batch_size, T, self.r_dim))             # Bias
        Ut = torch.zeros((batch_size, T, self.r_dim, self.r_dim)) # Temporal prediction matrices
        # initialize embedding
        r, r2 = self.init_code_(batch_size)
        R_bar[:, 0] = r.detach().clone().cpu()
        R2_hat[:, 0] = r2.detach().clone().cpu()
        I_bar[:, 0] = self.spatial_decoder(r).detach().clone().cpu()
        spatial_loss_rbar[:,0] = self.spat_loss(self.spatial_decoder(r), data_batch[:, 0])
        # p(r_1 | I_1)
        r = self.inf_first_step(data_batch[:, 0])
        R_hat[:, 0] = r.detach().clone().cpu()
        I_hat[:, 0] = self.spatial_decoder(r).detach().clone().cpu()
        I[:, 0] = data_batch[:, 0]
        r_first = r.detach().clone()
        spatial_loss_rhat[:,0] = self.spat_loss(self.spatial_decoder(r), data_batch[:, 0])

        for t in tqdm(range(1, T), leave=False):
            r2_p = r2.detach().clone()
            r_p = r.detach().clone()

            # hypernet prediction
            r_bar,V_t,w = self.temporal_prediction(r_p, r2)
            R_bar[:, t] = r_bar.detach().clone().cpu()
            x_bar = self.spatial_decoder(r_bar)
            I_bar[:, t] = x_bar.detach().clone().cpu()

            # inference
            r, r2, _ = self.inf(data_batch[:, t], r_p, r2.detach().clone())
            R_hat[:, t] = r.detach().clone().cpu()
            x_hat = self.spatial_decoder(r)
            I_hat[:, t] = x_hat.detach().clone().cpu()
            I[:, t] = data_batch[:, t]
            R2_hat[:, t]= r2.detach().clone().cpu()
            wb = self.hypernet(r2)
            W[:, t] = wb[:,:self.mix_dim].reshape(batch_size, -1).detach().clone().cpu()
            if self.dyn_bias:
                b[:, t] = wb[:,self.mix_dim:].reshape(batch_size, -1).detach().clone().cpu()
            Ut[:, t] = V_t.detach().clone().cpu()

            # loss
            spatial_loss_rhat[:,t,:] = self.spat_loss(x_hat, data_batch[:, t])
            spatial_loss_rbar[:,t,:] = self.spat_loss(x_bar, data_batch[:, t])
            temp_loss[:,t,:] = torch.pow(r - r_bar,2)

        # result dict
        result_dict = {}
        result_dict['I_bar'] = I_bar
        result_dict['I_hat'] = I_hat
        result_dict['I'] = I
        result_dict['R_bar'] = R_bar
        result_dict['R_hat'] = R_hat
        result_dict['R2_hat'] = R2_hat
        result_dict['W'] = W
        result_dict['Ut'] = Ut
        result_dict['temp_loss'] = temp_loss
        result_dict['spatial_loss_rhat'] = spatial_loss_rhat
        result_dict['spatial_loss_rbar'] = spatial_loss_rbar
        if self.dyn_bias:
            result_dict['b'] = b
            
        spatial_loss_rhat_avg = spatial_loss_rhat.reshape(batch_size,-1).sum(1).mean(0)
        spatial_loss_rbar_avg = spatial_loss_rbar.reshape(batch_size,-1).sum(1).mean(0)
        return spatial_loss_rhat_avg, spatial_loss_rbar_avg, self.temp_weight * temp_loss.reshape(batch_size,-1).sum(1).mean(0), result_dict
    
