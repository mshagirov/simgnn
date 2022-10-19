import time
import copy
import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
from os import path
from inspect import signature

# Default loss functions
mse_loss = torch.nn.MSELoss(reduction='mean')
l1_loss = torch.nn.L1Loss(reduction='mean')


def np_loss_wrt_time(pred, tgt, loss_type='mse'):
    '''Calculate loss averaged over axis=0 (time) for `np.ndarray`'''
    assert loss_type in ('mse', 'l1')
    if loss_type == 'mse':
        ls = (pred - tgt)**2
    elif loss_type == 'l1':
        ls = np.abs(pred-tgt)
    if pred.ndim > 2:
        # for var-s with >1 dim-s, e.g. velocity, position, etc.
        ls = ls.sum(axis=-1)
    return ls.mean(axis=0)


def train_model(model,
                data_loaders,
                optimizer,
                num_epochs=5,
                scheduler=None,
                device=torch.device('cpu'),
                model_states=['train', 'val', 'hara'],
                loss_func=l1_loss,
                use_force_loss={'train': [True, True], 'val': [True, True], 'hara': [False, False]},
                ignore_short_edges=False,
                edge_len_threshold=10**-4,
                return_best=False):
    '''
    Arg-s:
    - model : torch.nn.Module, should accept pt-geometric graph and return tuple
    (node_output, edge_output, cell_output), i.e. `X_vel, E_tens, C_pres = model(data)`
    - data_loaders : dict of datasets with `model_states` as keys.
    - optimizer, num_epochs, scheduler: optimizer (e.g. SGD), number of total epochs, and
                                        scheduler.
    - device : torch.device, should be same as the model's device.
    - model_states : list of dataset names, first two elem-s are always ['train', 'val']
    - loss_func : loss function for training w/ reduction='mean'. Same loss function is
                  used for all var types (i.e. node, edge, cell).
                  Total batch losses are weighted by #nodes in each batch.
    - ignore_short_edges : loss and gradients are not computed for edges with lengths
                           shorter than the `edge_len_threshold`.
    - use_force_loss: dict of states (from `model_states`) w/ lists of Booleans ([Tension, Pressure])
                      for deciding whether to compute loss for a variable. Order of the Booleans in the
                      list are as follows: `use_force_loss[state] = [tension, pressure]`.
                      This arg. is useful when the dataset does not contain targets for certain variables
                      (e.g. tensions and pressures in Hara movies) or when some variables are not needed
                      (e.g. model only computes velocities). Note that `model` should still output a tuple.
                      In order to ignore a variable, `model` could output `None` as an output and set corresponding
                      `use_force_loss` entries for *all* states as `False`.
    '''
    assert num_epochs > 0
    print('Training param-s:', end=' ')

    # init loss tracking
    loss_categories = ['tot', 'y', 'T', 'P']

    train_log = {state+'_loss_'+k: [] for state in model_states for k in loss_categories}
    running_losses = {k: 0.0 for k in train_log}
    n_samples = {k: 0.0 for k in train_log}

    # logs
    train_log['total_epochs'] = num_epochs
    train_log['loss_func'] = loss_func.__repr__()
    train_log['optimizer'] = optimizer.__class__.__name__
    train_log['optimizer_params'] = optimizer.__repr__()
    train_log['batch_size'] = data_loaders['train'].batch_size
    train_log['scheduler'] = 'none' if scheduler is None else scheduler.__class__.__name__
    train_log['scheduler_params'] = 'none' if scheduler is None else scheduler.state_dict().__repr__()
    train_log['return_best'] = return_best
    train_log['model'] = model.__repr__()

    print(f"#epochs={num_epochs}, metric={train_log['loss_func']}, " +
          f"batch_size={train_log['batch_size']}, " +
          f"optim={train_log['optimizer']}, sch-r={train_log['scheduler']}, " +
          f"return_best={train_log['return_best']}", end="\n---\n")

    time_start = time.time()
    best_wts = None
    best_loss = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}: ', end='')
        for state in model_states:
            if state == 'train':
                model.train()  # training mode
            else:
                model.eval()  # evaluation mode

            for k in loss_categories:
                running_losses[f'{state}_loss_{k}'] = 0.0  # reset loss accumulators
                n_samples[f'{state}_loss_{k}'] = 0

            for data in data_loaders[state]:
                data = data.to(device)
                optimizer.zero_grad()  # zero grad accumulator

                with torch.set_grad_enabled(state == 'train'):
                    X_vel, E_tens, C_pres = model(data)  # outputs:(#nodes/#edges/#cells, #dims)

                    vel_loss = loss_func(X_vel, data.y) if data.y is not None else 0.0

                    # ignore NaN targets
                    tens_compute_loss = use_force_loss[state][0] and (not torch.any(data.edge_tensions.isnan()).item())
                    if tens_compute_loss and ignore_short_edges:
                        edge_mask = data.edge_length > edge_len_threshold
                        tens_loss = loss_func(E_tens[edge_mask], data.edge_tensions[edge_mask])
                    elif tens_compute_loss:
                        tens_loss = loss_func(E_tens, data.edge_tensions)
                    else:
                        tens_loss = 0.0
                    pres_loss = loss_func(C_pres, data.cell_pressures) if use_force_loss[state][1] else 0.0

                    # total loss
                    loss = vel_loss + tens_loss + pres_loss

                    if state == 'train':
                        loss.backward()
                        optimizer.step()
                # accumulate losses
                running_losses[f'{state}_loss_y'] += vel_loss.item()*data.x.size(0) if data.y is not None else 0.0
                n_samples[f'{state}_loss_y'] += data.x.size(0)

                # total loss values are valid only for datasets w/ all variable
                # running_losses[f'{state}_loss_tot'] += loss.item()*data.x.size(0)
                # n_samples[f'{state}_loss_tot'] += data.x.size(0)

                if tens_compute_loss:
                    running_losses[f'{state}_loss_T'] += tens_loss.item()*data.edge_tensions.size(0)
                    n_samples[f'{state}_loss_T'] += data.edge_tensions.size(0)

                if use_force_loss[state][1]:
                    running_losses[f'{state}_loss_P'] += pres_loss.item()*data.cell_pressures.size(0)
                    n_samples[f'{state}_loss_P'] += data.cell_pressures.size(0)

            loss_sum_categories = 0
            for k in loss_categories:
                # log losses for dataset==state
                if (n_samples[f'{state}_loss_{k}'] < 1) or (k == 'tot'):
                    # print(f"{state}_loss_{k}=NA",end='; ')
                    continue
                train_log[f'{state}_loss_{k}'].append(
                 running_losses[f'{state}_loss_{k}']/n_samples[f'{state}_loss_{k}'])

                # accumulate total loss
                loss_sum_categories += running_losses[f'{state}_loss_{k}']/n_samples[f'{state}_loss_{k}']
                # print all losses for 'train' and only total loss for others.

            train_log[f'{state}_loss_tot'].append(loss_sum_categories)

            # print losses
            for k in loss_categories:
                if (state != 'train' and k != 'tot') or (n_samples[f'{state}_loss_{k}'] < 1 and k != 'tot'):
                    continue
                print(f"{state}_loss_{k}={train_log[f'{state}_loss_{k}'][-1]:8.4g}", end='; ')
            print('|', end='')

            if state == 'val' and epoch == 0:
                best_loss = train_log['val_loss_tot'][-1]
                if return_best:
                    best_wts = copy.deepcopy(model.state_dict())  # best weights

            if state == 'val' and train_log['val_loss_tot'][-1] < best_loss:
                best_loss = train_log['val_loss_tot'][-1]
                if return_best:
                    best_wts = copy.deepcopy(model.state_dict())  # best weights
            
             # apply LR schedule
            if (state == 'val') and (scheduler is not None):
                if 'metrics' in signature(scheduler.step).parameters:
                    scheduler.step(train_log['val_loss_tot'][-1])
                else:
                    scheduler.step()

        print(f'{time.time() - time_start:.0f}s')

    time_elapsed = time.time() - time_start
    print(f'Total elapsed time : {time_elapsed//60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val loss: {best_loss:4f} (return best:{return_best})')

    if return_best:
        model.load_state_dict(best_wts)  # load best model weights

    return model, train_log


def write_log(fpath, train_log):
    '''
    Write training log dict as a pickle file.
    Arg-s:
    - fpath: location to write the file, e.g. 'train_log.pkl'.
    - train_log: a python dict w/ training logs.
    '''
    with open(fpath, 'wb') as f:
        pickle.dump(train_log, f)


def load_log(fpath):
    '''Load training log dict from a pickle file ('*.pkl').'''
    with open(fpath, 'rb') as f:
        train_log = pickle.load(f)
    return train_log


def plot_losses(train_log, loaders, dataset_legend, figsize=[15, 8],
                legend_prefix='', legend_suffix='', return_axs=False,
                **plot_kwargs):
    '''Plot training losses for the logged datasets in `train_log`'''
    if figsize is not None:
        plt.figure(figsize=figsize)
    training_epochs = np.arange(train_log['total_epochs'])
    if return_axs:
        axs = []

    for data_name in loaders:
        # plot losses for each dataset
        ax = plt.plot(training_epochs, train_log[f'{data_name}_loss_tot'], lw=3,
                      label=legend_prefix + f'{dataset_legend[data_name]}' + legend_suffix,
                      **plot_kwargs)
        if return_axs:
            axs.append(ax)
    plt.legend()
    if return_axs:
        return axs


@torch.no_grad()
def predict_sample(model, input_data, device=torch.device('cpu')):
    '''
    Processes a single sample w/ enabled torch.no_grad().

    Use model.training, model.eval(), and model.train() to set and reset the training mode before
    using `predict_sample()`.

    - model :  `torch.nn` model.
    - input_data : a pt-geometric graph data compatible w/ the `model`.
    - device : target device for the input_data (must be same device as the model).
    '''
    return model(input_data.to(device))


def predict_dataset_tension(model, input_dataset, device=torch.device('cpu')):
    '''
    A simple "For loop" through the dataset.

    - input_dataset : pt-geometric dataset compatible with the `model`
    - device : device for the graph data, must be same device as the `model`.

    Returns (tens_pred, tens_tgt), if the data *doesn't contain* `is_rosette` labels
        tens_pred, tens_tgt : predicted and target tensions; lists of np.arrays, where each list elem-t
                             represents a single graph.

    Returns (tens_pred, tens_tgt, is_rosette), if the data contains rosette labels:
        tens_pred, tens_tgt, is_rosette: predicted and target tensions, and `np.bool` array with rosette
                                        labels (used for Hara's ablation data)
    '''
    tens_tgt = []
    tens_pred = []
    contains_rosette = False

    if 'is_rosette' in input_dataset[0]:
        # contains rosette labels
        is_rosette = np.zeros((len(input_dataset),), dtype=np.bool_)
        contains_rosette = True

    is_train_mode = model.training
    if is_train_mode:
        model.eval()

    for k, d_k in enumerate(input_dataset):
        _, Tp_k, _ = predict_sample(model, d_k, device=device)

        # targets
        if d_k.edge_tensions != None:
            tens_tgt.append(d_k.edge_tensions.cpu().numpy().reshape(1, -1))
        else:
            tens_tgt.append(np.full((1,Tp_k.size(0)), np.nan))

        # predictions
        tens_pred.append(Tp_k.to('cpu').numpy().reshape(1, -1))

        if contains_rosette:
            is_rosette[k] = d_k.is_rosette

    if is_train_mode:
        model.train()
    if contains_rosette:
        return tens_pred, tens_tgt, is_rosette

    return tens_pred, tens_tgt


def predict_dataset(model, input_dataset, device=torch.device('cpu'), concat=False):
    '''
    A simple "For loop" through the dataset.

    Arg-s:
        - input_dataset : pt-geometric dataset compatible with the `model`
        - device : device for the graph data, must be same device as the `model`.
        - concat : concartenates "velocity" and "tensions" as frames if `concat=True`.

    Returns: dict w/ keys 'predictions' and 'targets', which have values as follows,
        - predictions: dict of predictions, w/ keys ['tension', 'velocity'].
        - targets: dict of target/ground truth values w/ ['tension', 'velocity']. If the
                   dataset containes 'is_rosette' property then `targets` has three keys
                   ['tension', 'velocity', 'is_rosette'].
    '''
    results_ = {}
    results_['predictions'] = {'tension': [], 'velocity': []}
    results_['targets'] = {'tension': [], 'velocity': []}

    contains_rosette = False

    if 'is_rosette' in input_dataset[0]:
        # contains rosette labels
        results_['targets']['is_rosette'] = np.zeros((len(input_dataset),), dtype=np.bool_)
        contains_rosette = True

    is_train_mode = model.training  # remember the current model state
    if is_train_mode:
        model.eval()

    for k, d_k in enumerate(input_dataset):
        Vp_k, Tp_k, _ = predict_sample(model, d_k, device=device)

        # targets
        if d_k.y != None:
            results_['targets']['velocity'].append(d_k.y.cpu().numpy().reshape(1, -1, 2))

        if d_k.edge_tensions != None:
            results_['targets']['tension'].append(d_k.edge_tensions.cpu().numpy().reshape(1, -1))

        # predictions
        if Vp_k != None:
            results_['predictions']['velocity'].append(Vp_k.cpu().numpy().reshape(1, -1, 2))
        
        if Tp_k != None:
            results_['predictions']['tension'].append(Tp_k.cpu().numpy().reshape(1, -1))

        if contains_rosette:
            results_['targets']['is_rosette'][k] = d_k.is_rosette

    if concat:
        if len(results_['targets']['velocity']) > 0:
            results_['targets']['velocity'] = np.concatenate(results_['targets']['velocity'], axis=0)
        
        if len(results_['predictions']['velocity']) > 0:
            results_['predictions']['velocity'] = np.concatenate(results_['predictions']['velocity'], axis=0)
        
        if len(results_['targets']['tension']) > 0:
            results_['targets']['tension'] = np.concatenate(results_['targets']['tension'], axis=0)
        
        if len(results_['predictions']['tension']) > 0:
            results_['predictions']['tension'] = np.concatenate(results_['predictions']['tension'], axis=0)

    if is_train_mode:
        model.train()

    return results_


def predict_abln_tension(model, abln_dataset, device=torch.device('cpu')):
    '''
    Predicts Hara ablation edge tensions.

    Returns: tens_pred, recoils, is_rosette
    '''
    Tp, Tt, is_ros_ = predict_dataset_tension(model, abln_dataset, device=device)
    for k, (Tp_k, Tt_k) in enumerate(zip(Tp, Tt)):
        Tp[k] = Tp_k[~np.isnan(Tt_k)]
        Tt[k] = Tt_k[~np.isnan(Tt_k)]
    return np.concatenate(Tp), np.concatenate(Tt), is_ros_


# predict using for pt-geometric batches
@torch.no_grad()
def predict(model, input_data, loss_func=l1_loss,
            use_force_loss=True,
            return_losses=True,
            device=torch.device('cpu')):
    '''
    Arg-s: model, input_data, loss_func, device
    Returns: outputs, losses
    - outputs : model outputs, e.g. (X_vel, E_tens, C_pres)
    - losses : losses for model outputs, e.g. (vel_loss, tens_loss) {optional : return_losses=True}
    '''
    input_data = input_data.to(device)
    model.eval()  # evaluation mode

    X_vel, E_tens, C_pres = model(input_data)  # shapes: (#nodes/#edges/#cells, #dims)

    if not return_losses:
        return (X_vel, E_tens, C_pres), None

    vel_loss = loss_func(X_vel, input_data.y) if input_data.y is not None else 0.0

    # tens_mask = torch.logical_not(input_data.edge_tensions.isnan()) if use_force_loss[0] else None
    tens_loss = loss_func(E_tens, input_data.edge_tensions) if use_force_loss else 0.0

    pres_loss = loss_func(C_pres, input_data.cell_pressures) if (C_pres is not None) else 0.0

    return (X_vel, E_tens, C_pres), (vel_loss, tens_loss, pres_loss)


def predict_batch(model, data_loaders,
                  loss_func=l1_loss,
                  use_force_loss={'train': True, 'val': True, 'hara': False},
                  return_losses=True,
                  device=torch.device('cpu')):
    '''
    Returns:
    - outputs : tuple (X_vel, E_tens)
    - targets : tuple (X_vel_targets, E_tens_targets)
    - running_losses : dict of losses for input dataset loaders. Returns if return_losses is "True".
    '''
    # init loss tracking
    if return_losses:
        loss_categories = ['tot', 'y', 'T', 'P']
        loss_names = [f"{name}_loss_{k}" for name in data_loaders for k in loss_categories]
        running_losses = {k: 0.0 for k in loss_names}
        n_samples = {k: 0.0 for k in loss_names}
        loss_log = {k: 0.0 for k in loss_names}

    # predcitions
    X_vel_datasets = {name: [] for name in data_loaders}
    E_tens_datasets = {name: [] for name in data_loaders}
    C_pres_datasets = {name: [] for name in data_loaders}
    # true answers
    X_vel_targets = {name: [] for name in data_loaders}
    E_tens_targets = {name: [] for name in data_loaders}
    C_pres_targets = {name: [] for name in data_loaders}

    for name in data_loaders:
        for data in data_loaders[name]:
            outputs, losses = predict(model, data, loss_func=loss_func,
                                      use_force_loss=use_force_loss[name],
                                      return_losses=return_losses, device=device)
            # losses==None if return_losses is True
            
            X_vel_datasets[name].append(outputs[0].cpu())
            if data.y is not None:
                X_vel_targets[name].append(data.y.cpu())

            if (outputs[1] is not None) and (data.edge_tensions is not None):
                E_tens_datasets[name].append(outputs[1].cpu())
                E_tens_targets[name].append(data.edge_tensions.cpu())

            if (outputs[2] is not None) and (data.cell_pressures is not None):
                C_pres_datasets[name].append(outputs[2].cpu())
                C_pres_targets[name].append(data.cell_pressures.cpu())

            # loss tracking
            if return_losses:
                vel_loss, tens_loss, pres_loss = losses
                # accumulate losses
                # assumes Loss(reduction='sum')
                running_losses[f'{name}_loss_y'] += vel_loss.item()/2 if data.y is not None else 0.0
                n_samples[f'{name}_loss_y'] += data.x.size(0)

                # total loss values weighted by #nodes in each graph batch
                # running_losses[f'{name}_loss_tot'] += tot_loss.item()*data.x.size(0)
                # n_samples[f'{name}_loss_tot'] += data.x.size(0)

                if use_force_loss[name]:
                    running_losses[f'{name}_loss_T'] += tens_loss.item()
                    n_samples[f'{name}_loss_T'] += data.edge_tensions.size(0)

                if (outputs[2] is not None) and (data.cell_pressures is not None):
                    running_losses[f'{name}_loss_P'] += pres_loss.item()*data.cell_pressures.size(0)
                    n_samples[f'{name}_loss_P'] += data.cell_pressures.size(0)
        # compute mean losses
        if return_losses:
            loss_sum_categories = 0
            for k in loss_categories:
                if k == 'tot':
                    continue
                elif n_samples[f'{name}_loss_{k}'] < 1:
                    running_losses[f'{name}_loss_{k}'] = None
                    continue
                loss_log[f'{name}_loss_{k}'] = running_losses[f'{name}_loss_{k}']/n_samples[f'{name}_loss_{k}']
                # accumulate total loss
                loss_sum_categories += loss_log[f'{name}_loss_{k}']

            loss_log[f'{name}_loss_tot'] = loss_sum_categories

    if return_losses:
        return (X_vel_datasets, E_tens_datasets, C_pres_datasets), \
               (X_vel_targets, E_tens_targets, C_pres_targets), loss_log

    return (X_vel_datasets, E_tens_datasets, C_pres_datasets), (X_vel_targets, E_tens_targets, C_pres_targets)


def plot_velocity_predictions(vel_pred, vel_tgt, dataset_legend, var_name = r'$\Delta{}v$',
                              xlabel='True', ylabel='Predicted', plot_kw={}, axis_lims=None, n_points=1000, rng_seed=None,
                              subplots_kw={}, line45_kw={}, show_figs=True,
                              save_path=None, save_type='png', save_kw={}):
    '''Concatenate all batches and plot scatter plot target vs predicted velocity values.
    
    - plot_kw : plot kwargs, defaults: {'marker':'o', 'ms':7, 'mfc':'tomato', 'alpha':.25}
    - axis_lims : axis limits [min_x, max_x, min_y, max_y]. Default: [minY, maxY, minY, maxY].
    - n_points, rng_seed : number of random example points to plot, and corresponding numpy RNG seed (default: 42). 
    - subplots_kw: kwargs for plt.subplots; defaults: {'nrows':1, 'ncols':2, 'figsize':[15, 7], 'sharex':True, 'sharey':True}.
    - line45_kw : kwargs for 45-degree line; defaults: {'ls':'--', 'color':'b', 'lw':3, 'alpha':.25}
    
    '''
    if n_points!=None:
        rng = np.random.default_rng(rng_seed if rng_seed!=None else 42)
    # plot kwargs
    plt_plot_kw = {'ls':'', 'marker':'o', 'ms':3, 'mfc':'tomato', 'mec':'maroon', 'alpha':.25}
    for k in plot_kw:
       plt_plot_kw[k] = plot_kw[k]

    # subplots kwargs
    plt_subplots_kw={'nrows':1, 'ncols':2, 'figsize':[15, 7], 'sharex':True, 'sharey':True}
    for k in subplots_kw:
        plt_subplots_kw[k] = subplots_kw[k]

    # 45-degree line kwargs
    plt_line45_kw  = {'ls':'--', 'color':'b', 'lw':2, 'alpha':.5}
    for k in line45_kw:
        plt_line45_kw[k] = line45_kw[k]  # overwrite defaults

    data_names = [e for e in vel_tgt if len(vel_tgt[e]) > 1]
    for data_name in data_names:
        minY, maxY = torch.cat(vel_tgt[data_name], dim=0).min(), torch.cat(vel_tgt[data_name], dim=0).max()
        fig, axs = plt.subplots(**plt_subplots_kw)
        for k, ax in enumerate(axs):
            y_true = torch.cat(vel_tgt[data_name], dim=0)[:, k]
            y_pred = torch.cat(vel_pred[data_name], dim=0)[:, k]
            if n_points!=None:
                N_avail = min([n_points, y_true.size(0)])
                v_ids = rng.choice(y_true.size(0), size=(N_avail,), replace=False)
                y_true = y_true[v_ids]
                y_pred = y_pred[v_ids]
                
            ax.plot(y_true, y_pred, **plt_plot_kw)
            
            if axis_lims != None:
                ax.plot(axis_lims[:2], axis_lims[2:], **plt_line45_kw)
                ax.axis(axis_lims)
            else:
                ax.plot([minY, maxY], [minY, maxY], **plt_line45_kw)
                ax.axis([minY, maxY, minY, maxY])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f'{var_name}$_{k}$')
        plt.suptitle(f'{dataset_legend[data_name]}')
        if save_path!=None:
            plt.savefig(save_path+data_name+'.'+save_type, **save_kw)
        if show_figs:
            plt.show()


def plot_tension_prediction(t_pred, t_tgt, dataset_legend, var_name = r'$T$',
                            xlabel='True', ylabel='Predicted', plot_kw={}, axis_lims=None, n_points=1000, rng_seed=None,
                            figure_kw={}, line45_kw={}, show_figs=True,
                            save_path=None, save_type='png', save_kw={}):
    '''Concatenate all batches and plot scatter plot target vs predicted tension values'''
    if n_points!=None:
        rng = np.random.default_rng(rng_seed if rng_seed!=None else 42)
    # plot kwargs
    plt_plot_kw = {'ls':'', 'marker':'o', 'ms':3, 'mfc':'b', 'mec':'teal', 'alpha':.25}
    for k in plot_kw:
       plt_plot_kw[k] = plot_kw[k]
    # subplots kwargs
    plt_figure_kw={'figsize':[7, 7]}
    for k in figure_kw:
        plt_figure_kw[k] = figure_kw[k]
    # 45-degree line kwargs
    plt_line45_kw  = {'ls':'--', 'color':'orange', 'lw':2, 'alpha':.5}
    for k in line45_kw:
        plt_line45_kw[k] = line45_kw[k]  # overwrite defaults
    
    data_names = [e for e in t_tgt if len(t_tgt[e]) > 1]
    
    for data_name in data_names:
        fig = plt.figure(**plt_figure_kw)
        
        t_tgt_i = torch.cat(t_tgt[data_name], dim=0)
        t_mask = torch.logical_not(t_tgt_i.isnan())
        T_true  = t_tgt_i[t_mask]        
        T_pred = torch.cat(t_pred[data_name], dim=0)[t_mask]
        
        minY, maxY = T_true.min(), T_true.max()
        
        if n_points!=None:
            N_avail = min([n_points, T_true.size(0)])
            v_ids = rng.choice(T_true.size(0), size=(N_avail,), replace=False)
            T_true = T_true[v_ids]
            T_pred = T_pred[v_ids]

        plt.plot(T_true, T_pred, **plt_plot_kw)
        
        if axis_lims != None:
                plt.plot([minY, maxY], [minY, maxY], **plt_line45_kw)
                plt.axis(axis_lims)
        else:
            plt.plot([minY, maxY], [minY, maxY], **plt_line45_kw)
            plt.axis([minY, maxY, minY, maxY])

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if dataset_legend!=None:
            plt.title(f'{dataset_legend[data_name]}')
        if save_path!=None:
            plt.savefig(save_path+data_name+'.'+save_type, **save_kw)
        if show_figs:
            plt.show()
