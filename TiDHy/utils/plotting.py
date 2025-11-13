import numpy as np
import matplotlib.pyplot as plt
# import torch
# import torch.nn.functional as F
import matplotlib.gridspec as gridspec
import seaborn as sns

# from TiDHy.utils.utils import angle_between, add_colorbar

##### Plotting settings ######
import matplotlib as mpl
mpl.rcParams.update({'font.size':          10,
                     'axes.linewidth':     2,
                     'xtick.major.size':   5,
                     'ytick.major.size':   5,
                     'xtick.major.width':  2,
                     'ytick.major.width':  2,
                     'axes.spines.right':  False,
                     'axes.spines.top':    False,
                     'pdf.fonttype':       42,
                     'xtick.labelsize':    10,
                     'ytick.labelsize':    10,
                     'figure.facecolor':   'white',
                     'pdf.use14corefonts': True,
                     'font.family':        'Arial',
                    })


def plot_dynamics_2d(dynamics_matrix,
                     bias_vector,
                     mins=(-40,-40),
                     maxs=(40,40),
                     npts=20,
                     axis=None,
                     **kwargs):
    """Utility to visualize the dynamics for a 2 dimensional dynamical system.

    Args
    ----

        dynamics_matrix: 2x2 numpy array. "A" matrix for the system.
        bias_vector: "b" vector for the system. Has size (2,).
        mins: Tuple of minimums for the quiver plot.
        maxs: Tuple of maximums for the quiver plot.
        npts: Number of arrows to show.
        axis: Axis to use for plotting. Defaults to None, and returns a new axis.
        kwargs: keyword args passed to plt.quiver.

    Returns
    -------

        q: quiver object returned by pyplot
    """
    assert dynamics_matrix.shape == (2, 2), "Must pass a 2 x 2 dynamics matrix to visualize."
    assert len(bias_vector) == 2, "Bias vector must have length 2."

    x_grid, y_grid = np.meshgrid(np.linspace(mins[0], maxs[0], npts), np.linspace(mins[1], maxs[1], npts))
    xy_grid = np.column_stack((x_grid.ravel(), y_grid.ravel(), np.zeros((npts**2,0))))
    dx = xy_grid.dot(dynamics_matrix.T) + bias_vector - xy_grid

    if axis is not None:
        q = axis.quiver(x_grid, y_grid, dx[:, 0], dx[:, 1], **kwargs)
    else:
        q = plt.quiver(x_grid, y_grid, dx[:, 0], dx[:, 1], **kwargs)
    return q

# Helper functions for plotting results
def plot_trajectory(z, x, ax=None, ls="-"):
    color_names = ["windows blue", "red", "amber", "faded green"]
    zcps = np.concatenate(([0], np.where(np.diff(z))[0] + 1, [z.size]))
    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.gca()
    for start, stop in zip(zcps[:-1], zcps[1:]):
        ax.plot(x[start:stop + 1, 0],
                x[start:stop + 1, 1],
                lw=1, ls=ls,
                color=color_names[z[start] % len(color_names)],
                alpha=.75)
    return ax

def plot_observations(z, y, ax=None, ls="-", lw=1,):
    colors = plt.get_cmap('turbo', len(np.unique(z)))
    zcps = np.concatenate(([0], np.where(np.diff(z))[0] + 1, [z.size]))
    if ax is None:
        fig = plt.figure(figsize=(8, 4))
        ax = fig.gca()
    T, N = y.shape
    t = np.arange(T)
    for n in range(N):
        for start, stop in zip(zcps[:-1], zcps[1:]):
            ax.plot(t[start:stop + 1], y[start:stop + 1, n],
                    lw=lw, ls=ls,
                    color=colors(z[start]),
                    alpha=1.0)
    return ax

def loss_plots(nfig,result_dict,cfg,save_figs=False):
    loss_tr = result_dict['loss_tr']
    loss_te = result_dict['loss_te']
    fig,axs = plt.subplots(1,2,figsize=(10,5))
    ax = axs[0]
    ax.plot(loss_tr,'k',label='train')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax = axs[1]
    ax.plot(loss_te,'k',label='test')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Test Loss')
    plt.tight_layout()
    if save_figs:
        fig.savefig(cfg.paths.fig_dir/'{}_LossTrace.png'.format(nfig),dpi=300)
        nfig = nfig+1
    return nfig, fig, axs

def plot_inputs(nfig,result_dict,cfg,figsize=None,save_figs=False,t=0,dt=None,fontsize=10):
    if figsize==None:
        figsize = (10,8)
    if dt == None: 
        dt = 3*cfg.train['sequence_length']
    ##### Plot Emission Predictions #####
    from copy import deepcopy
    I_shuff  = deepcopy(result_dict['I'].reshape(-1,result_dict['I'].shape[-1]))
    spacing= .5
    I = result_dict['I'].reshape(-1,result_dict['I'].shape[-1])
    Ihat = result_dict['I_hat'].reshape(-1,result_dict['I_hat'].shape[-1])
    fig,axs = plt.subplots(1,2,figsize=figsize,sharey=True, width_ratios=[3, 1], gridspec_kw={'wspace':.15})
    ax = axs[0]
    # colormap = plt.get_cmap('turbo', len(np.unique(full_state_z)))
    hlines_I,hlines_Ihat = [],[]
    for n in range(I.shape[-1]):
        mean_centered_I = I[t:t+dt,n] - np.mean(I[t:t+dt,n],axis=0)
        ax.plot(mean_centered_I + n/spacing,color='k', lw=1,zorder=1)
        hlines_I.append(np.mean(mean_centered_I + n/spacing,axis=0))
        mean_centered_Ihat = Ihat[t:t+dt,n] - np.mean(Ihat[t:t+dt,n],axis=0)
        ax.plot(mean_centered_Ihat + n/spacing,ls='--',color='r', lw=1,zorder=2)
        hlines_Ihat.append(np.mean(mean_centered_Ihat + n/spacing,axis=0))
    ax.set_yticks(hlines_I)
    ax.set_yticklabels(np.arange(1,len(hlines_I)+1))
    ax.set_xlabel('Timesteps',fontsize=fontsize)
    ax.set_ylabel('Observation #',fontsize=fontsize)
    ax.set_title('Learned Observations',fontsize=fontsize)

    ax = axs[1]
    input_CC = np.array([(np.corrcoef(I[:,celln],Ihat[:,celln])[0, 1]) for celln in range(I.shape[1])])
    # Input_Errors = np.mean((I-Ihat)**2,axis=0)
    heights = deepcopy(input_CC)
    # I_shuff = shuffle_along_axis(I_shuff,axis=1)
    # Input_Errors_shuff = np.mean((I_shuff-I)**2,axis=0)
    xs = np.arange(I.shape[-1])
    ax.barh(y=hlines_I,width=heights,color='k',)
    # ax.errorbar(heights,hlines_I,xerr=np.std((I-Ihat),axis=0)/np.sqrt((I-Ihat).shape[0]),ls='none',color='tab:gray',capsize=3)
    # ax.axvline(x=np.mean(Input_Errors_shuff),c='k',ls='--',lw=2,label='shuffle error')
    ax.set_xticks(np.arange(0,1.2,.2))
    ax.set_xlim([0,1.05])
    ax.set_xlabel('CC',fontsize=fontsize)
    ax.set_title('CC',fontsize=fontsize)
    # ax.set_xticklabels(np.arange(heights.shape[-1])+1)
    fig.tight_layout()
    if save_figs:
        fig.savefig(cfg.paths.fig_dir/'{}_observations.png'.format(nfig),dpi=300)
        nfig = nfig+1
    return nfig, fig, axs

def plot_spatial_decoder(model,nfig,cfg,figsize=None,save_figs=False,fontsize=10):
    ##### Plotting Spatial Decoder #####
    if figsize==None:
        figsize=(8,4)
    fig, axs = plt.subplots(1, 1, figsize=figsize)
    ax = axs
    spatial_decoder = model.spatial_decoder[0].weight.data.detach().cpu().numpy().T
    im = ax.imshow(spatial_decoder,cmap='bwr')
    cbar = add_colorbar(im)
    ax.set_title('Spatial Decoder',fontsize=fontsize)
    ax.set_xlabel('Observations',fontsize=fontsize)
    ax.set_ylabel('Latent Variables',fontsize=fontsize)
    plt.tight_layout()

    if save_figs:
        fig.savefig(cfg.paths.fig_dir/'{}_spatial_decoder.png'.format(nfig))
        nfig = nfig+1

    ##### Plotting Clustered Spatial Decoder #####
    if cfg.model['input_dim'] > 1:
        clustermap = sns.clustermap(spatial_decoder,figsize=(8,8),cmap='bwr')
        clustermap.tick_params(axis='both', which='major', labelsize=10, labelbottom = True, bottom=False, top = False, labeltop=False)
        clustermap.ax_heatmap.set_xlabel('Observations',fontsize=fontsize)
        clustermap.ax_heatmap.set_ylabel('Latent Variables',fontsize=fontsize)
        if save_figs:
            fig = clustermap.fig
            fig.savefig(cfg.paths.fig_dir/'{}_clustered_spatial_decoder.png'.format(nfig),dpi=300)
            nfig = nfig+1
    return nfig, fig, axs

def plot_temporal_matrices(model,nfig,cfg,figsize=None,save_figs=False,fontsize=10):
    ##### Plot dynamic matrices #####
    if cfg.model.low_rank_temp:
        V = torch.bmm(model.temporal.unsqueeze(-1),model.temporal.unsqueeze(1)).data.cpu().detach()
    else:
        V = model.temporal.data.cpu().detach().reshape(model.mix_dim,model.r_dim,model.r_dim)
    # Calcualte cosine similarity between dynamic matrices
    with torch.no_grad():
        cos_reg = torch.sum(torch.abs(torch.tril(F.normalize(V.reshape(model.mix_dim, -1)) @ F.normalize(V.reshape(model.mix_dim, -1)).t(),diagonal=-1)))
    print('Cos sim:',cos_reg)
    V = V.numpy()
    cmax = np.max(np.abs(V))
    # col = np.ceil(V.shape[0]/2).astype(int)
    nrow = np.ceil(V.shape[0]/5).astype(int)
    if figsize==None:
        figsize=1.5
    fig, axs = plt.subplots(nrow,5,figsize=(figsize*5,figsize*nrow),layout='constrained')
    axs = axs.flatten()
    for n in range(V.shape[0]):
        ax = axs[n]
        im = ax.imshow(V[n],aspect='auto',cmap='bwr',vmin=-cmax,vmax=cmax)
        ax.set_title('$V_{{{}}}$'.format(n),fontsize=fontsize)
        ax.set_box_aspect(1)
    cbar = fig.colorbar(im, ax=axs[4::5],aspect=50)
    # plt.tight_layout()

    if save_figs:
        fig.savefig(cfg.paths.fig_dir/'{}_dynamic_matricies.png'.format(nfig),dpi=300)
        nfig = nfig+1
    return nfig, fig, axs

def plot_latent_LDS(nfig,result_dict,cfg,figsize=None,save_figs=False,t=0,dt=None,fontsize=10):
    ssm_params = cfg.dataset.ssm_params
    if dt == None: 
        dt = 3*cfg.train.sequence_length
    if 'states_x_val' in result_dict.keys():
        states_x_test = result_dict['states_x_val']/np.max(np.abs(result_dict['states_x_val']))
    elif 'states_x_test' in result_dict.keys():
        states_x_test = result_dict['states_x_test']/np.max(np.abs(result_dict['states_x_test']))
    if figsize == None:
        figsize=(15,2)
        
    ##### Calculated correlation ######
    R_bar = result_dict['R_bar'].reshape(-1,result_dict['R_bar'].shape[-1])
    # R_bar = result_dict['R_bar'].reshape(-1,result_dict['R_bar'].shape[-1])
    ccs = np.zeros((R_bar.shape[-1],states_x_test.shape[-1]))
    for n in range(R_bar.shape[-1]):
        for m in range(states_x_test.shape[-1]):
            ccs[n,m] = np.corrcoef(R_bar[:,n],states_x_test[:,m])[0,1]
    Rbar_plot = np.zeros_like(states_x_test)
    # Rbar_plot = np.zeros_like(states_x_test)
    x_plot = np.zeros_like(R_bar)
    norm = True
    ind = np.zeros((states_x_test.shape[-1],),dtype=int)
    for n in range(states_x_test.shape[-1]):
        ind[n] = np.nanargmax(np.abs(ccs[:,n]))
        flip_Rbar = -1 if ccs[ind[n],n,] < 0 else 1
        Rbar_plot[:,n] = flip_Rbar*R_bar[:,ind[n]]
        # Rbar_plot[:,n] = flip_Rbar*R_bar[:,ind[n]]
        if norm:
            Rbar_plot[:,n] = Rbar_plot[:,n]/np.max(np.abs(Rbar_plot[:,n]))
            # Rbar_plot[:,n] = Rbar_plot[:,n]/np.max(np.abs(Rbar_plot[:,n]))
    # R_acor = np.stack([autocorr(R_bar[:,n],norm=norm) for n in range(R_bar.shape[-1])],axis=1)
    # x_acor = np.stack([autocorr(states_x_test[:,n],norm=norm) for n in range(states_x_test.shape[-1])],axis=1)
    
    print(ind,np.max(np.abs(np.round(ccs,decimals=2)),axis=0))

    figs=[]; axes=[]
    ##### Plotting latent states #####
    spacing= .5
    max_ccs = np.max(np.abs(np.round(ccs,decimals=2)),axis=0)
    fig,axs = plt.subplots(1,1,figsize=(5,5),sharey=True,gridspec_kw={'wspace':.15})
    ax = axs
    # colormap = plt.get_cmap('turbo', len(np.unique(full_state_z)))
    hlines_x,hlines_Rbar = [],[]
    for n in range(states_x_test.shape[-1]):
        mean_centered_x = states_x_test[t:t+dt,n] - np.mean(states_x_test[t:t+dt,n],axis=0)
        ax.plot(mean_centered_x + n/spacing,color='k', lw=1,zorder=1)
        hlines_x.append(np.mean(mean_centered_x + n/spacing,axis=0))
        mean_centered_Rbar = Rbar_plot[t:t+dt,n] - np.mean(Rbar_plot[t:t+dt,n],axis=0)
        ax.plot(mean_centered_Rbar + n/spacing,ls='--',color='r', lw=1,zorder=2,label='$\hat{{R}}_{{{}}}$={:.02}'.format(ind[n],max_ccs[n]))
        hlines_Rbar.append(np.mean(mean_centered_Rbar + n/spacing,axis=0))
    ax.set_yticks(hlines_x)
    ax.set_yticklabels(np.arange(1,len(hlines_x)+1))
    ax.set_xlabel('Timesteps',fontsize=fontsize)
    ax.set_ylabel('$\hat{{r}}_t$',fontsize=fontsize)
    ax.set_title('Learned Latent States',fontsize=fontsize)
    ax.set_xlabel('Time')
    for n in range(len(hlines_x)):
        ax.text(x=dt+15,y=hlines_x[n],s='$r_{{{}}}$={:.02}'.format(ind[n],max_ccs[n]),fontsize=10)
    # ax.legend()
    plt.tight_layout()
    
    figs.append(fig)
    axes.append(axs)
    if save_figs:
        fig.savefig(cfg.paths.fig_dir/'{}_Rbar_vs_x.png'.format(nfig),dpi=300)
        nfig = nfig+1

    ##### Plotting CCA Correlations #####
    from sklearn.cross_decomposition import CCA
    n_comps=states_x_test.shape[-1]
    cca = CCA(n_components=n_comps,max_iter=1000)
    X_c,Y_c = cca.fit_transform(states_x_test,R_bar)
    cca_coefficient = np.corrcoef(X_c.T, Y_c.T).diagonal(offset=n_comps)
    x_w = cca.x_weights_
    y_w = cca.y_weights_
    cca_angles = [np.rad2deg(angle_between(X_c[:,n],Y_c[:,n])) for n in range(n_comps)]
    cca_angles_x = [np.rad2deg(angle_between(X_c[:,n],states_x_test[:,n])) for n in range(n_comps)]
    cca_angles_r = [np.rad2deg(angle_between(Y_c[:,n],R_bar[:,n])) for n in range(n_comps)]
    # X_c2,Y_c2 = cca.fit_transform(R_bar,states_x_test)
    # cca_coefficient2 = np.corrcoef(X_c2.T, Y_c2.T).diagonal(offset=n_comps)
    # x_w2 = cca.x_weights_
    # y_w2 = cca.y_weights_
    # cca_angles2 = [angle_between(X_c2[:,n],Y_c2[:,n]) for n in range(n_comps)]

    for n in range(n_comps):
        print('Rbar: comp {}, cc: {:.03}, ang: {:.03}'.format(n,cca_coefficient[n],cca_angles[n]))
        # print('Rbar: comp {}, cc: {:.03}, ang: {:.03}'.format(n,cca_coefficient2[n],cca_angles2[n]))

    spacing= .5
    fig,axs = plt.subplots(1,1,figsize=(5,5))
    ax = axs
    # colormap = plt.get_cmap('turbo', len(np.unique(full_state_z)))
    hlines_x,hlines_Rbar = [],[]
    for n in range(X_c.shape[-1]):
        mean_centered_x = X_c[t:t+dt,n] - np.mean(X_c[t:t+dt,n],axis=0)
        mean_centered_x=mean_centered_x/(np.max(np.abs(mean_centered_x)))
        ax.plot(mean_centered_x + n/spacing,color='k', lw=1,zorder=1)
        hlines_x.append(np.mean(mean_centered_x + n/spacing,axis=0))
        mean_centered_Rbar = Y_c[t:t+dt,n] - np.mean(Y_c[t:t+dt,n],axis=0)
        mean_centered_Rbar=mean_centered_Rbar/(np.max(np.abs(mean_centered_Rbar)))
        ax.plot(mean_centered_Rbar + n/spacing,ls='--',color='r', lw=1,zorder=2,label='$\hat{{R}}_{{{}}}$={:.02}'.format(ind[n],max_ccs[n]))
        hlines_Rbar.append(np.mean(mean_centered_Rbar + n/spacing,axis=0))
    ax.set_yticks(hlines_x)
    ax.set_yticklabels(np.arange(1,len(hlines_x)+1))
    ax.set_xlabel('Timesteps',fontsize=fontsize)
    ax.set_ylabel('$\hat{{r}}_t$',fontsize=fontsize)
    ax.set_title('CCA States',fontsize=fontsize)
    ax.set_xlabel('Time')
    for n in range(len(hlines_x)):
        ax.text(x=dt+15,y=hlines_x[n],s='r={:.03}'.format(cca_coefficient[n]) +'\n' + r'$\theta={:.3} \degree$'.format(cca_angles[n]),fontsize=10)
    # ax.legend()
    plt.tight_layout()
    if save_figs:
        fig.savefig(cfg.paths.fig_dir/'{}_CCA_components.png'.format(nfig),dpi=300)
        nfig = nfig+1
    figs.append(fig)
    axes.append(axs)
    
    ##### CCA Per System #####
    spacing = .5
    n_comps=cfg.dataset.ssm_params['latent_dim']
    fig,axs = plt.subplots(1,1,figsize=(5,5),layout='constrained')
    ax = axs
    count=0
    hlines_x,hlines_Rbar = [],[]
    for p in range(ssm_params['Nlds']):
        states_x_cca = states_x_test[:,(ssm_params['latent_dim']*(p)):(p+1)*ssm_params['latent_dim']]
        cca = CCA(n_components=ssm_params['latent_dim'])
        X_c,Y_c = cca.fit_transform(states_x_cca,R_bar)
        cca_coefficient = np.corrcoef(X_c.T, Y_c.T).diagonal(offset=n_comps)
        x_w = cca.x_weights_
        y_w = cca.y_weights_
        cca_angles = [np.rad2deg(angle_between(X_c[:,n],Y_c[:,n])) for n in range(n_comps)]
        cca_angles_x = [np.rad2deg(angle_between(X_c[:,n],states_x_cca[:,n])) for n in range(n_comps)]
        cca_angles_r = [np.rad2deg(angle_between(Y_c[:,n],R_bar[:,n])) for n in range(n_comps)]
        for n in range(n_comps):
            print('comp {}, cc: {:.03}, ang: {:.03}, ang_x:{:.03}, ang_r:{:.03}'.format(n,cca_coefficient[n],cca_angles[n],cca_angles_x[n],cca_angles_r[n]))

        for i in range(X_c.shape[-1]):
            mean_centered_x = X_c[t:t+dt,i] - np.mean(X_c[t:t+dt,i],axis=0)
            mean_centered_x=mean_centered_x/(np.max(np.abs(mean_centered_x)))
            ax.plot(mean_centered_x + count/spacing,color='k', lw=1.5,zorder=1)
            hlines_x.append(np.mean(mean_centered_x + count/spacing,axis=0))
            mean_centered_Rbar = Y_c[t:t+dt,i] - np.mean(Y_c[t:t+dt,i],axis=0)
            mean_centered_Rbar=mean_centered_Rbar/(np.max(np.abs(mean_centered_Rbar)))
            ax.plot(mean_centered_Rbar + count/spacing,ls='--',color='r', lw=1,zorder=2,label='$\hat{{R}}_{{{}}}$={:.02}'.format(ind[n],max_ccs[n]),alpha=.75)
            hlines_Rbar.append(np.mean(mean_centered_Rbar + count/spacing,axis=0))
            ax.text(x=dt+15,y=hlines_x[count],s='r={:.03}'.format(cca_coefficient[i]) +'\n' + r'$\theta={:.3} \degree$'.format(cca_angles[i]),fontsize=10)
            count += 1
        ax.set_yticks(hlines_x)
        ax.set_yticklabels(np.arange(1,len(hlines_x)+1))
        ax.set_xlabel('Timesteps',fontsize=fontsize)
        ax.set_ylabel('$\hat{{R}}_t$',fontsize=fontsize)
        ax.set_title('CCA per system',fontsize=fontsize)
        ax.set_xlabel('Time')
    plt.tight_layout()
    if save_figs:
        fig.savefig(cfg.paths.fig_dir/'{}_CCA_per_sys.png'.format(nfig),dpi=300)
        nfig = nfig+1
    figs.append(fig)

    return nfig, figs, axes

def plot_SLDS_plots(nfig,result_dict,data_dict,cfg,figsize=None,save_figs=False,t=0,dt=None,fontsize=10):    
    ssm_params = cfg.dataset.ssm_params
    if dt == None: 
        dt = 3*cfg.train.sequence_length
    if 'states_x_val' in result_dict.keys():
        states_x_test = result_dict['states_x_val']/np.max(np.abs(result_dict['states_x_val']))
        states_z_test = data_dict['states_z_val']
    elif 'states_x_test' in result_dict.keys():
        states_x_test = result_dict['states_x_test']/np.max(np.abs(result_dict['states_x_test']))
        states_z_test = data_dict['states_z_test']
    R_bar = result_dict['R_bar'].reshape(-1,result_dict['R_bar'].shape[-1])
    R_hat = result_dict['R_hat'].reshape(-1,result_dict['R_hat'].shape[-1])
    
    figs=[]; axes=[]
    ##### Plottign Eigenvalues #####
    As = np.stack([v for v in data_dict['As'].values()])
    fig,axs = plt.subplots(3,2,figsize=(6,8))
    for p in range(ssm_params.Nlds):
        for k in range(ssm_params.n_disc_states):
            rhat2 = R_hat[np.where(states_z_test[p]==k)[0],:]
            Uhat_0 = np.linalg.inv(rhat2[:-1].T@rhat2[:-1])@rhat2[:-1].T@rhat2[1:]
            rbar2 = R_bar[np.where(states_z_test[p]==k)[0],:]
            Ubar_0 = np.linalg.inv(rbar2[:-1].T@rbar2[:-1])@rbar2[:-1].T@rbar2[1:]
            evals_Uhat0 = np.linalg.eigvals(Uhat_0)
            evals_Ubar0 = np.linalg.eigvals(Ubar_0)
            evals_A = np.linalg.eigvals(As[p,k])

            ax = axs[p,k]
            ax.scatter(np.real(evals_Uhat0.reshape(-1)),np.imag(evals_Uhat0.reshape(-1)),10,c='g',zorder=10,label='Uhat')
            ax.scatter(np.real(evals_Ubar0.reshape(-1)),np.imag(evals_Ubar0.reshape(-1)),10,c='r',zorder=10,label='Ubar')
            ax.scatter(np.real(evals_A.reshape(-1)),np.imag(evals_A.reshape(-1)),25,c='k',zorder=5,label='data')
            ax.axis('square')
            ax.set_ylim([-1.1,1.1])
            ax.set_xlim([-1.1,1.1])
            # Move left y-axis and bottom x-axis to centre, passing through (0,0)
            ax.spines['left'].set_position('center')
            ax.spines['bottom'].set_position('center')

            # Eliminate upper and right axes
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')

            # Show ticks in the left and lower axes only
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            ax.set_xticks([-1,1])
            ax.set_yticks([-1,1])
            circ = plt.Circle((0, 0), radius=1, edgecolor='k', facecolor='None')
            ax.add_patch(circ)
            # ax.set_title('System {}'.format(n+1))
            
    axs[0,0].set_title('Eval Dynamics 1',fontsize=fontsize)
    axs[0,1].set_title('Eval Dynamics 2',fontsize=fontsize)
    plt.tight_layout()
    if save_figs:
        fig.savefig(cfg.paths.fig_dir/'{}_evals_per_sys.png'.format(nfig),dpi=300)
        nfig = nfig+1
    figs.append(fig)
    axes.append(axs)
    return nfig, figs, axes

def plot_latent_states(nfig,result_dict,cfg,figsize=None,save_figs=False,reduce_plots=False,t=0,dt=None):
    if dt == None: 
        dt = 3*cfg.train.sequence_length
    R_bar = result_dict['R_bar'].reshape(-1,result_dict['R_bar'].shape[-1])

    if figsize==None:
        figsize=(10,2)
    fig, axs = plt.subplots(1,1,figsize=figsize,sharey=True)
    clrs = plt.get_cmap('cool',R_bar.shape[-1]+1)
    for n in range(R_bar.shape[-1]):
        ax = axs
        if n == 0:
            ax.set_title("Predicted latent states",y=1.1)
        elif n == R_bar.shape[-1]:
            ax.set_xlabel("Time")
        # ax.plot(R_bar[t:t+dt,n]/np.max(np.abs(R_bar[:,n]),keepdims=True),c=clrs(n),zorder=10)
        ax.plot(R_bar[t:t+dt,n],c=clrs(n),label='$\hat{{R}}_{{{}}}$'.format(n))

    ax.set_xlabel('Time')
    ax.set_ylabel('Latent State')
    # ax.legend(labelcolor='linecolor',markerscale=0, handlelength=0, handletextpad=-1.5,bbox_to_anchor=(1, 1.15), loc='upper right', borderaxespad=0.,fontsize=12,ncol=R_bar.shape[-1],frameon=False)
    # plt.tight_layout()
    # ax.set_ylim([-1.1,1.1])
    if save_figs:
        fig.savefig(cfg.paths.fig_dir/'{}_Rbar_stacked.png'.format(nfig),dpi=300)
        nfig = nfig+1
    return nfig, fig, axs

def plot_Ws(nfig,result_dict,cfg,figsize=None,save_figs=False,t=0,dt=None,fontsize=10):
    if dt == None: 
        dt = 3*cfg.train.sequence_length
    W = result_dict['W'].reshape(-1,result_dict['W'].shape[-1])
    if 'b' in result_dict.keys():
        b = result_dict['b'].reshape(-1,result_dict['b'].shape[-1])
    ##### Plotting W #####
    clrs = plt.get_cmap('turbo',W.shape[-1]+1)
    if figsize==None:
        figsize=(8,4)
    if 'b' in result_dict.keys():
        fig, axs = plt.subplots(2, 1, figsize=figsize, sharey=False)
        ax = axs[0]
    else: 
        fig, axs = plt.subplots(1, 1, figsize=figsize, sharey=False)
        ax = axs

    for n in range(W.shape[-1]):
        ax.plot(W[t:t+dt,n], c=clrs(n),label='$W_{{{}}}$'.format(n+1))

    ax.set_title("Ws stacked",fontsize=fontsize)
    ax.set_xlabel("Time",fontsize=fontsize)
    ax.set_ylabel('W',fontsize=fontsize)

    for xline in np.arange(0,dt,cfg.train.sequence_length):
        ax.axvline(xline,c='k',ls='--',alpha=1)
    if 'b' in result_dict.keys():
        ax =axs[1]
        ax.plot(b[t:t+dt])
        for xline in np.arange(0,dt,cfg.train.sequence_length):
            ax.axvline(xline,c='k',ls='--',alpha=1)
        ax.set_title("Bias stacked",fontsize=fontsize)
        ax.set_ylabel('b',fontsize=fontsize)
        ax.set_xlabel("Time",fontsize=fontsize)

    plt.tight_layout()
    if save_figs:
        fig.savefig(cfg.paths.fig_dir/'{}_Ws.png'.format(nfig),dpi=300)
        nfig = nfig+1
    return nfig, fig, axs

def plot_R2_hat(nfig,result_dict,cfg,figsize=None,save_figs=False,t=0,dt=None,fontsize=10):
    if dt == None: 
        dt = 3*cfg.train.sequence_length
    R2_hat = result_dict['R2_hat'].reshape(-1,result_dict['R2_hat'].shape[-1])
    ##### Plotting R2_hat #####
    clrs = plt.get_cmap('turbo',R2_hat.shape[-1]+1)
    if figsize==None:
        figsize=(8,2)
    fig, axs = plt.subplots(1, 1, figsize=figsize, sharey=False)
    ax = axs
    for n in range(R2_hat.shape[-1]):
        ax.plot(R2_hat[t:t+dt,n], c=clrs(n))
    ax.set_title("R2_hat stacked",fontsize=fontsize)
    ax.set_xlabel("Time",fontsize=fontsize)
    ax.set_ylabel('R2_hat',fontsize=fontsize)

    for xline in np.arange(0,dt,cfg.train.sequence_length):
        ax.axvline(xline,c='k',ls='--',alpha=1)
    plt.tight_layout()
    if save_figs:
        fig.savefig(cfg.paths.fig_dir/'{}_R2hat.png'.format(nfig),dpi=300)
        nfig = nfig+1
    return nfig, fig, axs

def plot_true_traj(nfig,result_dict,cfg,figsize=(8, 9),save_figs=False,t=0,dt=None,spacing=1,fontsize=10):
    if dt == None:
        dt = 10*cfg.train.sequence_length

    fig = plt.figure(figsize=figsize,layout='compressed')
    gs = gridspec.GridSpec(nrows=5, ncols=3,hspace=.5) 
    gs0 = gridspec.GridSpecFromSubplotSpec(1, cfg.dataset['ssm_params']['Nlds'], subplot_spec=gs[0,:], wspace=.4,hspace=.5)
    gs1 = gridspec.GridSpecFromSubplotSpec(1, cfg.dataset['ssm_params']['Nlds'], subplot_spec=gs[1,:], wspace=.2,hspace=.5)
    gs2 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[2:,:], wspace=0,hspace=.10)

    if 'states_z_test' in result_dict.keys():
        import itertools
        lst = list(itertools.product([0, 1], repeat=3))
        full_state_z = np.zeros(cfg.dataset.ssm_params['time_bins'],dtype=int)
        for n in range(len(lst)):
            full_state_z[np.apply_along_axis(lambda x: np.all(x == lst[n]),0,result_dict['states_z_test'])] = n


    I = result_dict['I'].reshape(-1,result_dict['I'].shape[-1])
    if cfg.delay.delay_embed:
        I = I[:,:cfg.model.orig_input_size]
    l = [(n,n+1) for n in np.arange(0,cfg.dataset['ssm_params']['latent_dim']*cfg.dataset['ssm_params']['Nlds'],2)]
    count = 1
    for k in range(cfg.dataset['ssm_params']['Nlds']):
        if len(result_dict['As'][k].shape) == 3:
            As = result_dict['As'][k][0]
            bs = result_dict['bs'][k][0]
        else:
            As = result_dict['As'][k]
            bs = result_dict['bs'][k]
        ax = plt.subplot(gs0[0,k])
        plot_x = result_dict['states_x_test'][:,l[k]]
        q = plot_dynamics_2d(As, 
                            bias_vector=bs,
                            mins=[-1,-1],#plot_x.min(axis=0),
                            maxs=[1,1],#plot_x.max(axis=0),
                            axis=ax)
        # ax.set_ylim(-1,1)
        # ax.set_xlim(-1,1)
        ax.plot(plot_x[t:t+dt,0], plot_x[t:t+dt,1], '-k', lw=1)
        ax.set_xlabel("$x_{}$".format(count),fontsize=fontsize)
        count +=1
        ax.set_ylabel("$x_{}$".format(count),fontsize=fontsize)
        ax.set_xticks([-1,0,1])
        ax.set_xticklabels([-1,0,1])
        ax.set_yticks([-1,0,1])
        ax.set_yticklabels([-1,0,1])
        # ax.set_title(f'Latent States {k+1}' '\n' r'$\tau=${}'.format(cfg.dataset.ssm_params['timescales'][k]),fontsize=fontsize)
        ax.set_title(f'Latent States {k+1}',fontsize=fontsize)
        ax.set_aspect('equal', 'box')
        
        ax = plt.subplot(gs1[0,k])
        ax.acorr(result_dict['states_x_test'][:,l[k][0]],maxlags=100,usevlines=True,linestyle='-',color='k',lw=2)
        ax.set_xlabel("Lag", fontsize=fontsize)
        ax.set_yticks([0,1])
        ax.set_ylim(0,1)
        if k == 0:
            ax.set_ylabel("Autocorrelation",fontsize=fontsize)
        elif k == cfg.dataset.ssm_params['Nlds']-1:
            ax.set_yticks([])

    axs = np.array([fig.add_subplot(gs2[0]),fig.add_subplot(gs2[1])])
    ax = axs[0]
    # colormap = plt.get_cmap('turbo', len(np.unique(full_state_z)))
    spacing = .5
    hlines = []
    for n in range(result_dict['states_x_test'].shape[-1]):
        mean_centered = result_dict['states_x_test'][t:t+dt,n] - np.mean(result_dict['states_x_test'][t:t+dt,n],axis=0)
        mean_centered = mean_centered/np.max(np.abs(mean_centered))
        ax.plot(mean_centered + n/spacing,color='k', lw=1)
        hlines.append(np.mean(mean_centered + n/spacing,axis=0))
    ax.set_yticks(hlines)
    ax.set_yticklabels(np.arange(1,len(hlines)+1))
    ax.set_xticks([])
    ax.set_ylabel('Latent States',fontsize=fontsize)
    # im = ax.imshow(full_state_z[None,t:t+dt],cmap=colormap,alpha=0.5,extent=[0,dt,-1/spacing,n/spacing+1/spacing],aspect='auto')

    ax = axs[1]
    hlines = []
    for n in range(I.shape[-1]):
        mean_centered = I[t:t+dt,n] - np.mean(I[t:t+dt,n],axis=0)
        mean_centered = mean_centered/np.max(np.abs(mean_centered))
        ax.plot(mean_centered + n/spacing,color='k', lw=1)
        hlines.append(np.mean(mean_centered + n/spacing,axis=0))
    ax.set_yticks(hlines)
    ax.set_yticklabels(np.arange(1,len(hlines)+1))
    ax.set_xlabel('Timesteps',fontsize=fontsize)
    ax.set_ylabel("Observations",fontsize=fontsize)
    # im = ax.imshow(full_state_z[None,t:t+dt],cmap=colormap,alpha=0.5,extent=[0,dt,-1/spacing,n/spacing+1/spacing],aspect='auto')
    # cbar = fig.colorbar(im,ax=axs.flatten(),aspect=30)
    # cbar.set_ticks(np.arange(len(np.unique(full_state_z))))
    # cbar.set_ticklabels(lst)
        
    # plt.tight_layout()

    if save_figs:
        fig.savefig(cfg.paths.fig_dir/'{}_True_traj.png'.format(nfig),dpi=300)
        nfig = nfig+1

    return nfig, fig

# TODO: make sure cfg and inputs are correct
def plot_figures(model,result_dict,cfg,save_figs=True,t=0,dt=None,reduce_plots=False):
    
    ##### Plotting ######
    nfig=0
    if dt == None:
        dt = 10*cfg.train['sequence_length']

    print('Plotting Results...')
    
    ##### Plotting Loss #####
    if 'loss_tr' in result_dict.keys():
        nfig, fig, axs = loss_plots(nfig,result_dict,cfg,save_figs=save_figs)
    if 'As' in result_dict.keys():
        nfig,fig     = plot_true_traj(nfig,result_dict,cfg,save_figs=save_figs,t=t,dt=dt)
    nfig,fig,axs = plot_inputs(nfig,result_dict,cfg,save_figs=save_figs,t=t,dt=dt)
    nfig,fig,axs = plot_temporal_matrices(model,nfig,cfg,figsize=None,save_figs=save_figs,fontsize=10)
    nfig,fig,axs = plot_spatial_decoder(model,nfig,cfg,save_figs=save_figs)
    nfig,fig,axs = plot_latent_LDS(nfig,result_dict,cfg,save_figs=save_figs,t=t,dt=dt)
    nfig,fig,axs = plot_Ws(nfig,result_dict,cfg,save_figs=save_figs,t=t,dt=dt)
    nfig,fig,axs = plot_R2_hat(nfig,result_dict,cfg,save_figs=save_figs,t=t,dt=dt)

def plot_figures_real(model,result_dict,cfg,save_figs=True,t=0,dt=None,reduce_plots=False):
    
    ##### Plotting ######
    nfig=0
    if dt == None:
        dt = 10*cfg.train['sequence_length']

    print('Plotting Results...')
    
    ##### Plotting Loss #####
    if 'loss_tr' in result_dict.keys():
        nfig, fig, axs = loss_plots(nfig,result_dict,cfg,save_figs=save_figs)
    nfig,fig,axs = plot_inputs(nfig,result_dict,cfg,save_figs=save_figs,t=t,dt=dt)
    nfig,fig,axs = plot_temporal_matrices(model,nfig,cfg,figsize=None,save_figs=save_figs,fontsize=10)
    nfig,fig,axs = plot_spatial_decoder(model,nfig,cfg,save_figs=save_figs)
    nfig,fig,axs = plot_latent_states(nfig,result_dict,cfg,save_figs=save_figs,t=t,dt=dt)
    nfig,fig,axs = plot_Ws(nfig,result_dict,cfg,save_figs=save_figs,t=t,dt=dt)
    nfig,fig,axs = plot_R2_hat(nfig,result_dict,cfg,save_figs=save_figs,t=t,dt=dt)



def plot_r2_loss(r2):
    fig = plt.figure()
    plt.plot(r2)
    return fig

def plot_hist(trace,lim0,lim1,hbins,ax,label,clr='k',alpha=.5):
    count,edges = np.histogram(trace,bins=np.arange(lim0,lim1,hbins))
    edges_mid = np.array([(edges[i]+edges[i+1])/2 for i in range(len(edges)-1)])
    ax.bar(edges_mid, count/len(trace),color=clr,width=hbins, alpha=alpha,zorder=1,label=label) 
    return ax


def plot_dynamic_matrix(V,figsize=None,fontsize=10):
    cmax = np.max(np.abs(V))
    # col = np.ceil(V.shape[0]/2).astype(int)
    nrow = np.ceil(V.shape[0]/5).astype(int)
    if figsize==None:
        figsize=1.5
    fig, axs = plt.subplots(nrow,5,figsize=(figsize*5,figsize*nrow),layout='constrained')
    axs = axs.flatten()
    for n in range(V.shape[0]):
        ax = axs[n]
        im = ax.imshow(V[n],aspect='auto',cmap='bwr',vmin=-cmax,vmax=cmax)
        ax.set_title('$V_{{{}}}$'.format(n),fontsize=fontsize)
        ax.set_box_aspect(1)
    cbar = fig.colorbar(im, ax=axs[4::5],aspect=50)
    return fig, axs


def plot_dynamics_2d(dynamics_matrix,
                     bias_vector,
                     mins=(-40,-40),
                     maxs=(40,40),
                     npts=20,
                     axis=None,
                     **kwargs):
    """Utility to visualize the dynamics for a 2 dimensional dynamical system.

    Args
    ----

        dynamics_matrix: 2x2 numpy array. "A" matrix for the system.
        bias_vector: "b" vector for the system. Has size (2,).
        mins: Tuple of minimums for the quiver plot.
        maxs: Tuple of maximums for the quiver plot.
        npts: Number of arrows to show.
        axis: Axis to use for plotting. Defaults to None, and returns a new axis.
        kwargs: keyword args passed to plt.quiver.

    Returns
    -------

        q: quiver object returned by pyplot
    """
    assert dynamics_matrix.shape == (2, 2), "Must pass a 2 x 2 dynamics matrix to visualize."
    assert len(bias_vector) == 2, "Bias vector must have length 2."

    x_grid, y_grid = np.meshgrid(np.linspace(mins[0], maxs[0], npts), np.linspace(mins[1], maxs[1], npts))
    xy_grid = np.column_stack((x_grid.ravel(), y_grid.ravel(), np.zeros((npts**2,0))))
    dx = xy_grid.dot(dynamics_matrix.T) + bias_vector - xy_grid

    if axis is not None:
        q = axis.quiver(x_grid, y_grid, dx[:, 0], dx[:, 1], **kwargs)
    else:
        q = plt.quiver(x_grid, y_grid, dx[:, 0], dx[:, 1], **kwargs)
    return q

# Helper functions for plotting results
def plot_trajectory(z, x, ax=None, ls="-"):
    zcps = np.concatenate(([0], np.where(np.diff(z))[0] + 1, [z.size]))
    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.gca()
    for start, stop in zip(zcps[:-1], zcps[1:]):
        ax.plot(x[start:stop + 1, 0],
                x[start:stop + 1, 1],
                lw=1, ls=ls,
                # color=colors[z[start] % len(colors)],
                alpha=.75)
    return ax

def plot_observations(z, y, ax=None, ls="-", lw=1):

    zcps = np.concatenate(([0], np.where(np.diff(z))[0] + 1, [z.size]))
    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.gca()
    T, N = y.shape
    t = np.arange(T)
    for n in range(N):
        for start, stop in zip(zcps[:-1], zcps[1:]):
            ax.plot(t[start:stop + 1], y[start:stop + 1, n],
                    lw=lw, ls=ls,
                    # color=colors[z[start] % len(colors)],
                    alpha=1.0)
    return ax


def plot_spatial_rf(U, size=(10,4)):
    ##### Plotting Spatial Decoder #####
    fig, axs = plt.subplots(1, 1, figsize=size)
    ax = axs 
    im = ax.imshow(U,cmap='bwr')
    cbar = add_colorbar(im)
    ax.set_title('Spatial Decoder')
    ax.set_xlabel('Observations')
    ax.set_ylabel('Latent Variables')
    plt.tight_layout()
    return fig, axs

