import time
import copy
import torch
import pickle

# Default loss functions
mse_loss = torch.nn.MSELoss(reduction='mean')
l1_loss = torch.nn.L1Loss(reduction='mean')

def train_model(model,
                data_loaders,
                optimizer,
                num_epochs = 5,
                scheduler = None,
                device = torch.device('cpu'),
                model_states = ['train', 'val', 'hara'],
                loss_func = l1_loss,
                use_force_loss = {'train':[True,True], 'val':[True,True], 'hara':[False,False]},
                return_best = False):
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
    assert num_epochs>0
    print(f'Training param-s:', end=' ')

    # init loss tracking
    loss_categories = ['tot','y', 'T', 'P']

    train_log = {state+'_loss_'+k: [] for state in model_states for k in loss_categories}
    running_losses = {k:0.0 for k in train_log}
    n_samples = {k:0.0 for k in train_log}

    # logs
    train_log['total_epochs'] = num_epochs
    train_log['loss_func'] = loss_func.__repr__()
    train_log['optimizer'] = optimizer.__repr__()
    train_log['scheduler'] = 'none' if scheduler is None else scheduler.__class__.__name__
    train_log['scheduler_params'] = 'none' if scheduler is None else scheduler.state_dict().__repr__()
    train_log['return_best'] = return_best
    train_log['model'] = model.__repr__()

    print(f"#epochs={num_epochs}, metric={train_log['loss_func']}, "+
          f"optim={train_log['optimizer']}, sch-r={train_log['scheduler']}, "+
          f"return_best={train_log['return_best']}",end="\n---\n")

    time_start = time.time()
    best_wts = None
    best_loss = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}: ', end='')
        for state in model_states:
            if state == 'train':
                model.train() # training mode
            else:
                model.eval() # evaluation mode

            for k in loss_categories:
                running_losses[f'{state}_loss_{k}']=0.0 # reset loss accumulators
                n_samples[f'{state}_loss_{k}']=0

            for data in data_loaders[state]:
                data = data.to(device)
                optimizer.zero_grad() # zero grad accumulator

                with torch.set_grad_enabled(state=='train'):
                    X_vel, E_tens, C_pres = model(data) # outputs:(batch, #dims)
                    vel_loss    = loss_func(X_vel, data.y)

                    tens_loss   = loss_func(E_tens, data.edge_tensions) if use_force_loss[state][0] else 0.0

                    pres_loss   = loss_func(C_pres, data.cell_pressures) if use_force_loss[state][1] else 0.0

                    loss = vel_loss + tens_loss + pres_loss # total loss

                    if state=='train':
                        loss.backward()
                        optimizer.step()
                # accumulate losses
                running_losses[f'{state}_loss_y'] += vel_loss.item()*data.x.size(0)
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
                if n_samples[f'{state}_loss_{k}']<1:
                    #print(f"{state}_loss_{k}=NA",end='; ')
                    continue
                train_log[f'{state}_loss_{k}'].append( running_losses[f'{state}_loss_{k}']/n_samples[f'{state}_loss_{k}'])
                print(f"{state}_loss_{k}={train_log[f'{state}_loss_{k}'][-1]:.4f}",end='; ')
            print('|',end='')

            if state == 'val' and epoch==0 :
                best_loss = train_log['val_loss_tot'][-1]
                if return_best:
                    best_wts = copy.deepcopy(model.state_dict()) # best weights

            if state == 'val' and train_log['val_loss_tot'][-1] < best_loss:
                best_loss = train_log['val_loss_tot'][-1]
                if return_best:
                    best_wts = copy.deepcopy(model.state_dict()) # best weights

        print(f'{time.time() - time_start:.0f}s')

        # apply LR schedule
        if scheduler!=None:
            scheduler.step()

    time_elapsed = time.time() - time_start
    print(f'Total elapsed time : {time_elapsed//60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val loss: {best_loss:4f} (return best:{return_best})')

    if return_best:
        model.load_state_dict(best_wts) # load best model weights

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
