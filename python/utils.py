import numpy as np
import torch
import math
from torch.utils.data.dataset import Dataset
from datetime import datetime
import matplotlib.pyplot as plt
import scipy.io


class CellDatasetTorch(Dataset):
    def __init__(self, path, n_labels, data_type='exp', data_field=None, step=1, seed=1234,
                 snr=None, shift=None, data_frac=1):
        """
        Input:
        :param path (str)          : path to csv file
        :param n_labels (int)      : number of samples in one observation
        :param data_type (str)     : indentifier which data to use
        :param data_field (str)    : which field in mat-file to use
        :param step (int)          : step size while reading in data (subsampling)
        :param seed (int)          : seed for random number (for reproducibility)
        :param snr (float)         : signal to noise ratio in samples
        :param shift (float)       : max percentage of shift
        :param data_frac (float)   : fraction of data to use
        """

        mat_file = scipy.io.loadmat(path)
        self.rng = np.random.default_rng(seed)

        # assign parameters
        self.n_labels = n_labels
        self.snr = snr
        self.shift = shift

        if data_type == 'sim':
            # use simulated data
            self.data = mat_file['data_final']

            # choose amount of samples according to data_frac
            samples_per_label = int(len(self.data) / self.n_labels)
            samples_per_label_new = int(data_frac * samples_per_label)

            if data_frac < 1:
                # choose randomly an equal amount of samples per label
                data_choice = self.rng.choice(samples_per_label, samples_per_label_new, replace=False)
                data_choice = np.tile(data_choice, self.n_labels)
                data_choice += np.repeat(np.arange(0, len(self.data), samples_per_label), samples_per_label_new)
                self.data = self.data[data_choice]

            self.samples_per_label = samples_per_label_new

            self.labels = np.repeat(np.arange(1, self.n_labels + 1), self.samples_per_label)

        elif data_type == 'exp':
            # use experimental data
            if data_field is None:
                data_4um, data_8um = mat_file['data_4um'], mat_file['data_8um']
                data_4um8um = mat_file['data_4um8um']
                labels_4um = np.array(mat_file['labels_4um'], dtype=int).flatten()
                labels_8um = np.array(mat_file['labels_8um'], dtype=int).flatten()
                labels_4um8um = np.array(mat_file['labels_4um8um'], dtype=int).flatten()
                self.data = np.concatenate((data_4um, data_8um, data_4um8um), axis=0)
                self.labels = np.concatenate((labels_4um, labels_8um, labels_4um8um), axis=0)
            else:
                self.data = mat_file[data_field]
                self.labels = mat_file['labels']

        else:
            raise ValueError('Select correct dataset!')

        # subsample data according to step
        self.data = self.data[:, ::step]

        # assign remaining dataset sizes
        self.n_samples = len(self.data)
        self.n_features = self.data.shape[1]

        # calculate power of signals
        self.p_signals = np.mean(self.data**2)

    def __getitem__(self, index):
        label = torch.tensor(self.labels[index] - 1, dtype=int)
        signal_as_np = self.data[index]

        # shift according to max. defined fraction of total samples
        if self.shift is not None:
            left_shift = True if (self.rng.integers(2) == 0) else False
            shift = self.rng.integers(int(self.shift * self.n_features))
            if left_shift:
                signal_as_np = np.concatenate((signal_as_np[shift:], signal_as_np[:shift]))
            else:
                signal_as_np = np.concatenate((signal_as_np[-shift:], signal_as_np[:-shift]))

        # add noise according to snr
        if self.snr is not None:
            p_noise = self.p_signals / self.snr
            signal_as_np = signal_as_np + self.rng.normal(0, np.sqrt(p_noise), len(signal_as_np))

        # normalize
        signal_as_np = signal_as_np / np.max(np.abs(signal_as_np))

        # reshape sample into numpy array with one channel
        signal_as_np = np.expand_dims(signal_as_np, 0)

        # convert to tensor
        sample_as_tensor = torch.from_numpy(signal_as_np).float()

        return sample_as_tensor, label

    def __len__(self):
        return self.n_samples


class CellDatasetTorchPred(Dataset):
    def __init__(self, path, data_field, step=1, seed=1234):
        """
        Input:
        :param path (string)       : path to csv file
        :param data_field (string) : field in mat-file which contains data
        :param step (int)          : step size while reading in data (subsampling)
        :param seed (int)          : seed for random number (for reproducibility)
        """

        # load data from mat-file and corresponding field
        self.data = scipy.io.loadmat(path)[data_field]

        # set custom seed for reproducibility
        self.rng = np.random.default_rng(seed)

        # subsample data according to step
        self.data = self.data[:, ::step]

        # assign remaining dataset sizes
        self.n_samples = len(self.data)
        self.n_features = self.data.shape[1]

    def __getitem__(self, index):

        # get signal with index
        signal_as_np = self.data[index]

        # normalize
        signal_as_np = signal_as_np / np.max(np.abs(signal_as_np))

        # reshape sample into numpy array with one channel
        signal_as_np = np.expand_dims(signal_as_np, 0)

        # convert to tensor
        sample_as_tensor = torch.from_numpy(signal_as_np).float()

        return sample_as_tensor

    def __len__(self):
        return self.n_samples


class ParameterDatasetTorch(Dataset):
    def __init__(self, path_prm, labeled=1, data_spec='p', label_spec='p_vd'):
        """
        Input:
        :param path_prm (string)   : path to mat file
        :param labeled (int)       : is the data labeled
        :param data_spec (str)     : which field from the mat-file to choose for data
        :param label_spec (str)    : which field of the mat-file to choose for label
        """
        self.labeled = labeled

        # load data and set seed
        mat_file = scipy.io.loadmat(path_prm)

        if self.labeled == 1:
            self.data = mat_file[data_spec]
            self.label = mat_file[label_spec][:, 1]
        elif self.labeled == 0:
            self.data = mat_file[data_spec]
        else:
            raise ValueError('Either choose train or evaluation data.')

        # extract desired parameters
        mu_diff = self.data[:, 1] - self.data[:, 0]
        self.data = np.concatenate((mu_diff[..., np.newaxis], self.data[:, 2:]), axis=1)

        # assign dataset sizes
        self.n_samples = len(self.data)
        self.n_features = self.data.shape[1]

    def __getitem__(self, index):
        prm = self.data[index]

        # convert to tensor
        prm = torch.from_numpy(prm).float()

        if self.labeled == 1:
            return prm, self.label[index]
        else:
            return prm, []

    def __len__(self):
        return len(self.data)


def load_model(model, optimizer, args):

    if args.load_checkpoint == 1:
        # load model from checkpoint
        checkpoint = torch.load(args.models_dir + 'checkpoint-%s.pt' % args.model)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        args.epoch_begin = checkpoint['epoch'] + 1
        args.best_epoch = checkpoint['best_epoch']
        args.best_test_acc = checkpoint['best_test_acc'] if 'best_test_acc' in checkpoint else []
        args.best_test_loss = checkpoint['best_test_loss']

    elif args.load_checkpoint == 2:
        # load model from specific data
        model.load_state_dict(torch.load(args.models_dir + args.model_file))
        args.epoch_begin = 1
        args.best_epoch = 0
        args.best_test_acc = 0
        args.best_test_loss = math.inf

    else:
        # set up new model
        args.epoch_begin = 1
        args.best_epoch = 0
        args.best_test_acc = 0
        args.best_test_loss = math.inf

    return model, optimizer, args


def train_and_test_cl(model, train_loader, test_loader, optim, criterion, scheduler, args):

    for epoch in range(args.epoch_begin, args.epoch_begin + args.n_epochs):

        # use functions for classification
        train_cl(model, epoch, train_loader, criterion, optim, scheduler, args)
        test_loss_mean, _, test_acc, _ = test_cl(model, epoch, test_loader, criterion, args)
        scheduler.step(test_loss_mean)

        # save best state
        if test_acc > args.best_test_acc:
            args.best_epoch = epoch
            args.best_test_acc = test_acc
            args.best_test_loss = test_loss_mean
            torch.save(model.state_dict(), args.models_dir + 'best-%s.pt' % args.model)

        # save checkpoint
        torch.save({'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optim.state_dict(),
                    'best_epoch': args.best_epoch,
                    'best_test_loss': args.best_test_loss,
                    'best_test_acc': args.best_test_acc},
                   args.models_dir + 'checkpoint-%s.pt' % args.model)

    return args


def train_and_test_reg(model, train_loader, test_loader, optim, criterion, scheduler, args):

    for epoch in range(args.epoch_begin, args.epoch_begin + args.n_epochs):

        # use functions for regression
        train_reg(model, epoch, train_loader, criterion, optim, scheduler, args)
        test_loss_mean, _ = test_reg(model, epoch, test_loader, criterion, args)
        scheduler.step(test_loss_mean)

        # save best state
        if test_loss_mean < args.best_test_loss:
            args.best_epoch = epoch
            args.best_test_loss = test_loss_mean
            torch.save(model.state_dict(), args.models_dir + 'best-%s.pt' % args.model)

        # save checkpoint
        torch.save({'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optim.state_dict(),
                    'best_epoch': args.best_epoch,
                    'best_test_loss': args.best_test_loss},
                   args.models_dir + 'checkpoint-%s.pt' % args.model)

    return args


def train_cl(model, epoch, train_loader, criterion, optimizer, scheduler, args):
    model.train()
    loss_list = []
    acc_list = []

    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(args.device), labels.to(args.device)
        optimizer.zero_grad()

        output, cam = model(data)
        loss = criterion(output, labels)
        loss.backward()

        optimizer.step()
        # scheduler.step()
        loss_list.append(loss.item())

        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max probability
        acc = pred.eq(labels.view_as(pred)).float().mean()
        acc_list.append(acc.item())

        if (batch_idx % args.log_freq == 0) and (batch_idx > 0):
            msg = 'Train epoch {} [{}/{} ({:.0f}%)] \t avg. loss: {:.3f} ± {:.3f}, \t accuracy: {:.2f}%'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                np.mean(loss_list), np.std(loss_list), 100*np.mean(acc_list))
            print(msg, file=open(args.results_dir + args.results_file, "a"))

            if args.verbose > 0:
                loginfo(msg)

            loss_list.clear()
            acc_list.clear()


def train_reg(model, epoch, train_loader, criterion, optimizer, scheduler, args):
    model.train()
    loss_list = []

    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(args.device), labels.to(args.device)
        optimizer.zero_grad()

        output = torch.squeeze(model(data))
        loss = criterion(output.double(), labels)
        loss.backward()

        optimizer.step()
        # scheduler.step()
        loss_list.append(loss.item())

        if (batch_idx % args.log_freq == 0) and (batch_idx > 0):
            msg = 'Train epoch {} [{}/{} ({:.0f}%)] \t mse: {:.4f} ± {:.4f}, \t rmse: {:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                np.mean(loss_list), np.std(loss_list), np.sqrt(np.mean(loss_list)))

            if args.verbose > 0:
                loginfo(msg)

            loss_list.clear()


def test_cl(model, epoch, test_loader, criterion, args):
    model.eval()
    loss_list = []
    pred_list = []
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(test_loader):
            data, labels = data.to(args.device), labels.to(args.device)
            output, cam = model(data)

            loss_list.append(criterion(output, labels).item())  # append batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max probability
            correct += pred.eq(labels.view_as(pred)).sum().item()
            pred_list.extend((pred + 1).tolist())

    # mean and std of losses and accuracy
    loss_mean, loss_std = np.mean(loss_list), np.std(loss_list)
    acc = correct / len(test_loader.dataset)

    msg = 'Test set' + (epoch is not None) * ' after {} epochs'.format(epoch) + ':\tloss: {:.3f} ± {:.3f},\t\
        accuracy: {}/{} ({:.2f}%)\n'.format(loss_mean, loss_std, correct, len(test_loader.dataset), 100. * acc)
    if epoch is None:
        print(msg, file=open(args.results_dir + args.results_file, 'a'))

    # display info message
    if args.verbose > 0:
        loginfo(msg)

    return loss_mean, loss_std, acc, pred_list


def test_reg(model, epoch, test_loader, criterion, args):
    model.eval()
    loss_list = []

    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(args.device), labels.to(args.device)
            output = torch.squeeze(model(data))

            loss_list.append(criterion(output, labels).item())  # append batch loss

    # mean and standard deviation of losses
    loss_mean = np.mean(loss_list)
    loss_std = np.std(loss_list)

    msg = 'Test set' + (epoch is not None) * ' after {} epochs'.format(epoch) + ':\t mse: {:.4f} ± {:.4f},\t' \
        'rmse: {:.4f}\n'.format(loss_mean, loss_std, np.sqrt(loss_mean))
    print(msg, file=open(args.results_dir + args.results_file, 'a'))

    if args.verbose > 0:
        loginfo(msg)

    return loss_mean, loss_std


def evaluate_reg(model, loader, args):
    model.eval()
    prm_list = []
    labels_list = []

    # run all data through model and save outputs in list
    with torch.no_grad():
        for data, labels in loader:
            data = data.to(args.device)

            output = torch.squeeze(model(data))

            prm_list.extend(output)
            labels_list.extend(labels)

    return np.array(prm_list), np.array(labels_list)


def loginfo(msg):
    now = datetime.now()
    display_now = str(now).split(' ')[1][:-3]
    print('[' + display_now + ']' + ' ' + msg)
