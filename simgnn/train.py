import time
import copy
import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Default loss functions
mse_loss = torch.nn.MSELoss(reduction='mean')
l1_loss = torch.nn.L1Loss(reduction='mean')


def train_model(model,
                data_loaders,
                optimizer,
                num_epochs=5,
                scheduler=None,
                device=torch.device('cpu'),
                model_states=['train', 'val', 'hara'],
                loss_func=l1_loss,
                use_force_loss={'train': [True, True], 'val': [True, True], 'hara': [False, False]},
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
                    tens_mask = torch.logical_not(data.edge_tensions.isnan()) if use_force_loss[state][0] else None
                    tens_loss = loss_func(E_tens[tens_mask],
                                          data.edge_tensions[tens_mask]) if use_force_loss[state][0] else 0.0
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
                running_losses[f'{state}_loss_tot'] += loss.item()*data.x.size(0)
                n_samples[f'{state}_loss_tot'] += data.x.size(0)

                if use_force_loss[state][0]:
                    running_losses[f'{state}_loss_T'] += tens_loss.item()*data.edge_tensions.size(0)
                    n_samples[f'{state}_loss_T'] += data.edge_tensions.size(0)

                if use_force_loss[state][1]:
                    running_losses[f'{state}_loss_P'] += pres_loss.item()*data.cell_pressures.size(0)
                    n_samples[f'{state}_loss_P'] += data.cell_pressures.size(0)

            for k in loss_categories:
                # log losses for dataset==state
                if n_samples[f'{state}_loss_{k}'] < 1:
                    # print(f"{state}_loss_{k}=NA",end='; ')
                    continue
                train_log[f'{state}_loss_{k}'].append(
                 running_losses[f'{state}_loss_{k}']/n_samples[f'{state}_loss_{k}'])
                # print all losses for 'train' and only total loss for others.
                if state != 'train' and k != 'tot':
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

        print(f'{time.time() - time_start:.0f}s')

        # apply LR schedule
        if scheduler is not None:
            scheduler.step()

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


def plot_losses(train_log, loaders, dataset_legend, figsize=[15, 8]):
    '''Plot training losses for the logged datasets in `train_log`'''
    if figsize is not None:
        plt.figure(figsize=figsize)
    training_epochs = np.arange(train_log['total_epochs'])
    for data_name in loaders:
        # plot losses for each dataset
        plt.plot(training_epochs, train_log[f'{data_name}_loss_tot'], lw=3,
                 label=f'{dataset_legend[data_name]}')
    plt.legend()


def predict(model, input_data, loss_func=l1_loss,
            use_force_loss=[True, True],
            return_losses=True,
            device=torch.device('cpu')):
    '''
    Arg-s: model, input_data, loss_func, device

    Returns: outputs, losses
    - outputs : tuple (X_vel, E_tens, C_pres)
    - losses : tuple (vel_loss, tens_loss, pres_loss, loss) {optional : return_losses=True}
    '''
    input_data = input_data.to(device)
    model.eval()  # evaluation mode
    with torch.set_grad_enabled(False):
        X_vel, E_tens, C_pres = model(input_data)  # shapes: (#nodes/#edges/#cells, #dims)

        if not return_losses:
            return (X_vel, E_tens, C_pres), None

        vel_loss = loss_func(X_vel, input_data.y) if input_data.y is not None else 0.0

        tens_mask = torch.logical_not(input_data.edge_tensions.isnan()) if use_force_loss[0] else None
        tens_loss = loss_func(E_tens[tens_mask], input_data.edge_tensions[tens_mask]) if use_force_loss[0] else 0.0

        pres_loss = loss_func(C_pres, input_data.cell_pressures) if use_force_loss[1] else 0.0
        loss = vel_loss + tens_loss + pres_loss  # total loss

    return (X_vel, E_tens, C_pres), (vel_loss, tens_loss, pres_loss, loss)


def predict_batch(model, data_loaders,
                  loss_func=l1_loss,
                  use_force_loss={'train': [True, True], 'val': [True, True], 'hara': [False, False]},
                  return_losses=True,
                  device=torch.device('cpu')):
    '''
    Returns:
    - outputs : tuple (X_vel, E_tens, C_pres)
    - targets : tuple (X_vel_targets, E_tens_targets, C_pres_targets)
    - running_losses : dict of losses for input dataset loaders. Returns if return_losses is "True".
    '''
    # init loss tracking
    if return_losses:
        loss_categories = ['tot', 'y', 'T', 'P']
        loss_names = [f"{name}_loss_{k}" for name in data_loaders for k in loss_categories]
        running_losses = {k: 0.0 for k in loss_names}
        n_samples = {k: 0.0 for k in loss_names}

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
                (vel_loss, tens_loss, pres_loss, tot_loss) = losses
                # accumulate losses
                running_losses[f'{name}_loss_y'] += vel_loss.item()*data.x.size(0) if data.y is not None else 0.0
                n_samples[f'{name}_loss_y'] += data.x.size(0)

                # total loss values weighted by #nodes in each graph batch
                running_losses[f'{name}_loss_tot'] += tot_loss.item()*data.x.size(0)
                n_samples[f'{name}_loss_tot'] += data.x.size(0)

                if use_force_loss[name][0]:
                    running_losses[f'{name}_loss_T'] += tens_loss.item()*data.edge_tensions.size(0)
                    n_samples[f'{name}_loss_T'] += data.edge_tensions.size(0)

                if use_force_loss[name][1]:
                    running_losses[f'{name}_loss_P'] += pres_loss.item()*data.cell_pressures.size(0)
                    n_samples[f'{name}_loss_P'] += data.cell_pressures.size(0)
        # compute mean losses
        if return_losses:
            for k in loss_categories:
                if n_samples[f'{name}_loss_{k}'] < 1:
                    running_losses[f'{name}_loss_{k}'] = None
                    continue
                running_losses[f'{name}_loss_{k}'] = running_losses[f'{name}_loss_{k}']/n_samples[f'{name}_loss_{k}']

    if return_losses:
        return (X_vel_datasets, E_tens_datasets, C_pres_datasets), (X_vel_targets, E_tens_targets, C_pres_targets), running_losses

    return (X_vel_datasets, E_tens_datasets, C_pres_datasets), (X_vel_targets, E_tens_targets, C_pres_targets)


def plot_velocity_predictions(vel_pred, vel_tgt, dataset_legend,
                              figsize=[15, 7]):
    '''Concatenate all batches and plot scatter plot target vs predicted velocity values'''
    var_name = '$\Delta{}x$'
    data_names = [e for e in vel_tgt if len(vel_tgt[e]) > 1]
    for data_name in data_names:
        minY, maxY = torch.cat(vel_tgt[data_name], dim=0).min(), torch.cat(vel_tgt[data_name], dim=0).max()
        fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=figsize)
        for k, ax in enumerate(axs):
            ax.plot([minY, maxY], [minY, maxY], '--', color='b', lw=3, alpha=.5)
            ax.plot(torch.cat(vel_tgt[data_name], dim=0)[:, k],
                    torch.cat(vel_pred[data_name], dim=0)[:, k], 'o', ms=10, mfc='tomato', alpha=.25)
            ax.set_xlabel('True')
            ax.set_ylabel('Predicted')
            ax.set_title(f'{var_name}$_{k}$')
        plt.suptitle(f'{dataset_legend[data_name]}')
        plt.show()


def plot_tension_prediction(t_pred, t_tgt, dataset_legend,
                            nrows=1, ncols=3, figsize=[23, 7]):
    '''Concatenate all batches and plot scatter plot target vs predicted tension values'''
    data_names = [e for e in t_tgt if len(t_tgt[e]) > 1]
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axs = axs.ravel()
    for data_name, ax in zip(data_names, axs):
        t_tgt_i = torch.cat(t_tgt[data_name], dim=0)
        t_mask = torch.logical_not(t_tgt_i.isnan())

        minY, maxY = t_tgt_i[t_mask].min(), t_tgt_i[t_mask].max()

        ax.plot([minY, maxY], [minY, maxY], '--', color='orange', lw=3, alpha=.8)
        ax.plot(t_tgt_i[t_mask],
                torch.cat(t_pred[data_name], dim=0)[t_mask], 'o',
                ms=10, c='c', mfc='teal', alpha=.2)

        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')
        ax.set_title(f'{dataset_legend[data_name]}')
    plt.suptitle('Tension')
    plt.show()
