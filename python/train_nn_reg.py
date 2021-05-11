import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import argparse
from utils import train_and_test_reg, ParameterDatasetTorch, evaluate_reg, load_model, loginfo

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./", type=str, help="Path where to save output.")
    parser.add_argument("--data_path", default="./data/", type=str, help="Path to files.")
    parser.add_argument("--data_filename", default="exp_prm_train.mat", type=str, help="Parameter filename.")
    parser.add_argument("--data_spec", default='p', type=str, help="Which field from mat-file is data?")
    parser.add_argument("--label_spec", default='p_vd', type=str, help="Which field from mat-file is label?")
    parser.add_argument("--labeled", default=1, type=int, help="Is the data labeled?.")
    parser.add_argument("--results_dir", default="results/", type=str, help="Path where to save results.")
    parser.add_argument("--results_file", default='results_reg.txt', type=str, help="Results filename.")
    parser.add_argument("--train", default=0, type=int, help="Train the model.")
    parser.add_argument("--test", default=0, type=int, help="Test the model on test dataset.")
    parser.add_argument("--eval", default=1, type=int, help="Evaluate the model with histograms.")
    parser.add_argument("--workers", default=2, type=int, help="Number of workers to use.")
    parser.add_argument("--seed", default=1234, type=int, help="Seed for reproducibility.")
    parser.add_argument("--n_epochs", default=300, type=int, help="Number of epochs.")
    parser.add_argument("--hidden_dim", default=10, type=int, help="Number of neurons in hidden layers.")
    parser.add_argument("--lr", default=1e-2, type=float, help="Learning rate.")
    parser.add_argument("--gamma", default=0.9999, type=float, help="Decay for Exponential LR scheduler.")
    parser.add_argument("--factor", default=0.5, type=float, help="Plateau scaling factor for LR scheduler.")
    parser.add_argument("--activation", default='relu', type=str, help="Activation function.")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size to use during training.")
    parser.add_argument("--train_frac", default=0.7, type=float, help="Fraction of training dataset.")
    parser.add_argument("--test_frac", default=0.3, type=float, help="Fraction of test dataset.")
    parser.add_argument("--log_freq", default=10, type=int, help="Display frequency")
    parser.add_argument("--verbose", default=1, type=int, help="Which output level to display.")
    parser.add_argument("--model", default="mlp", type=str, help="Which model to use.")
    parser.add_argument("--models_dir", default="models/", type=str, help="Path where to save models.")
    parser.add_argument("--load_checkpoint", default=2, type=int, help="Resume training from a checkpoint.")
    parser.add_argument("--model_file", default='mlp_10_exp.pt', type=str)

    args = parser.parse_args()
    print(args, file=open(args.results_dir + args.results_file, "a"))
    if args.verbose > 0:
        print(args)

    torch.manual_seed(args.seed)

    kwargs = {'num_workers': args.workers, 'pin_memory': False}

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define dataset and split it
    dataset = ParameterDatasetTorch(args.data_path + args.data_filename, args.labeled, args.data_spec, args.label_spec)
    if args.train:
        length_train = int(args.train_frac * len(dataset))
        dataset_train = Subset(dataset, np.arange(length_train))
        dataset_test = Subset(dataset, np.concatenate((np.arange(length_train, 320), np.arange(321, len(dataset)))))
        train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.test or args.eval:
        train_loader = []
        test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
    else:
        raise ValueError('Choose to train, test or evaluate model.')

    # define model
    model = nn.Sequential(
        nn.Linear(dataset.n_features, args.hidden_dim, bias=False),
        nn.BatchNorm1d(args.hidden_dim),
        nn.ReLU(),
        nn.Linear(args.hidden_dim, args.hidden_dim, bias=False),
        nn.BatchNorm1d(args.hidden_dim),
        nn.ReLU(),
        nn.Linear(args.hidden_dim, args.hidden_dim, bias=False),
        nn.BatchNorm1d(args.hidden_dim),
        nn.ReLU(),
        nn.Linear(args.hidden_dim, 1, bias=True)
    )

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.factor)
    model.to(args.device)

    criterion = nn.MSELoss()
    criterion.to(args.device)

    model, optimizer, args = load_model(model, optimizer, args)

    if args.train:

        # train for number of epochs
        args = train_and_test_reg(model, train_loader, test_loader, optimizer, criterion, scheduler, args)

        msg = 'Best model @ epoch {}'.format(args.best_epoch)
        print(msg, file=open(args.results_dir + args.results_file, "a"))
        if args.verbose > 0:
            loginfo(msg)

        # test once again at best epoch
        model.load_state_dict(torch.load(args.models_dir + 'best-%s.pt' % args.model))

    if (args.test or args.train) and args.labeled:

        # only test model on dataset
        output, labels = evaluate_reg(model, test_loader, args)

        # calculate statistics
        err = (output - labels) ** 2
        err_mu, err_std, err_med, err_max = np.mean(err), np.std(err), np.median(err), err.max()
        msg = '\nMean squared error on test set:\t {:.3f} ± {:.3f}\n'.format(err_mu, err_std)

        print('{:.4f},{:.4f},{:.4f},{:.4f}'.format(err_mu, err_std, err_med, err_max),
              file=open(args.results_dir + args.results_file[:-4] + args.label_spec[1:3] + '.csv', "a"))
        if args.verbose > 0:
            loginfo(msg)

    if args.eval:

        # run through model
        output, labels = evaluate_reg(model, test_loader, args)

        # save output
        np.savetxt(args.results_dir + 'prm_pred' + args.data_spec[2:] + '.csv', output, delimiter=',', fmt='%f')

        # plot histogram
        fig, ax = plt.subplots(figsize=(10, 8))
        bins = np.arange(1, 20, 0.1)
        if args.labeled:
            # ax.hist([output, labels], bins=bins, density=True, label=['pred.', 'true'])
            ax.hist(labels, bins=bins, density=True)
        else:
            ax.hist(output, bins=bins, density=True, label=['pred.'])
        ax.legend()
        ax.set_xlabel('Cell diameter')
        ax.set_ylabel('Relative frequency')
        ax.set_ylim([0.0, 0.6])
        fig.savefig(args.results_dir + 'histogram_' + args.data_spec + '_mlp.pdf', format='pdf')

    print('Finished.')
