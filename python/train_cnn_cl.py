import torch
import argparse
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from utils import train_and_test_cl, test_cl, CellDatasetTorch, CellDatasetTorchPred, load_model, loginfo
from model import FCN

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", default="models/", type=str, help="Path where to save models.")
    parser.add_argument("--results_dir", default="results/", type=str, help="Path where to save results.")
    parser.add_argument("--results_file", default='results_cl.txt', type=str, help="Results filename.")
    parser.add_argument("--train", default=1, type=int, help="Train the model.")
    parser.add_argument("--test", default=0, type=int, help="Test the model.")
    parser.add_argument("--pred", default=0, type=int, help="Predict labels for unlabelled data.")
    parser.add_argument("--data_path", default="./data/", type=str, help="Path to data-file.")
    # parser.add_argument("--data_filename", default="sim_data_eval_4um8um_shift-50.mat", type=str, help="Input data.")
    parser.add_argument("--data_filename", default="exp_data_4um8um.mat", type=str, help="Input data.")
    parser.add_argument("--data_type", default="exp", type=str, help="Which data to use.")
    parser.add_argument("--data_field", default=None, type=str, help="Which field in the mat-file to use.")
    parser.add_argument("--n_labels", default=4, type=int, help="Number of different labels/classes.")
    parser.add_argument("--data_frac", default=1, type=float, help="Fraction of data to use.")
    parser.add_argument("--workers", default=4, type=int, help="Number of workers to use.")
    parser.add_argument("--step", default=4, type=int, help="Step size for subsampling during data loading.")
    parser.add_argument("--snr", default=30., type=float, help="SNR in dB for noise addition.")
    parser.add_argument("--shift", default=0.025, type=float, help="Random max. percentage shift of loaded data.")
    parser.add_argument("--seed", default=1234, type=int, help="Seed for reproducibility.")
    parser.add_argument("--n_epochs", default=1, type=int, help="Number of epochs.")
    parser.add_argument("--lr", default=5e-4, type=float, help="Learning rate.")
    parser.add_argument("--gamma", default=0.9999, type=float, help="Decay for Exponential LR scheduler.")
    parser.add_argument("--factor", default=0.5, type=float, help="Plateau scaling factor for LR scheduler.")
    parser.add_argument("--channels", default=[1, 16, 32, 16], nargs='+', type=int, help="Channel dimensions.")
    parser.add_argument("--activation", default='relu', type=str, help="Activation function.")
    # parser.add_argument("--kernels", default=[8, 5, 3], nargs='+', type=int, help="Kernel sizes.")
    parser.add_argument("--kernels", default=[16, 10, 6], nargs='+', type=int, help="Kernel sizes.")
    # parser.add_argument("--kernels", default=[24, 15, 9], nargs='+', type=int, help="Kernel sizes.")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size to use during training.")
    parser.add_argument("--train_frac", default=0.7, type=float, help="Fraction of training dataset.")
    parser.add_argument("--test_frac", default=0.3, type=float, help="Fraction of test dataset.")
    parser.add_argument("--log_freq", default=100, type=int, help="Display frequency")
    parser.add_argument("--verbose", default=2, type=int, help="Which output level to display.")
    parser.add_argument("--model", default="fcn", type=str, help="Which model to use.")
    parser.add_argument("--load_checkpoint", default=2, type=int, help="Resume training from a checkpoint.")
    # parser.add_argument("--model_file", default='fcn_sim_eval_25dB.pt', type=str)
    parser.add_argument("--model_file", default='fcn_exp_eval.pt', type=str)

    args = parser.parse_args()
    if args.verbose > 0:
        print(args)
    print(args, file=open(args.results_dir + args.results_file, "a"))

    args.snr = 10 ** (args.snr / 10) if args.snr is not None else None

    torch.manual_seed(args.seed)

    kwargs = {'num_workers': args.workers, 'pin_memory': False}

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define dataset and split it
    if args.train or args.test:
        dataset = CellDatasetTorch(args.data_path + args.data_filename, args.n_labels, args.data_type, args.data_field,
                                   args.step, args.seed, args.snr, args.shift, args.data_frac)
    elif args.eval:
        dataset = CellDatasetTorchPred(args.path, args.data_field, args.step, args.seed)
    else:
        raise ValueError('Define dataset!')

    if args.train:
        length_train = int(args.train_frac * len(dataset))
        lengths = [length_train, len(dataset) - length_train]
        dataset_train, dataset_test = random_split(dataset, lengths)
        train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.test or args.pred:
        train_loader = []
        dataset_test = dataset
        test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
    else:
        raise ValueError('Choose to train or test model or predict labels.')

    # define model
    if args.model == "fcn":
        model = FCN(args.channels, args.kernels, args.activation, args.n_labels)
    elif args.model == "resnet":
        model = ...
    else:
        raise ValueError('Specify correct model.')

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.factor)
    model.to(args.device)

    criterion = nn.NLLLoss()
    criterion.to(args.device)

    model, optimizer, args = load_model(model, optimizer, args)

    if args.train:
        # train and test model for number of epochs
        args = train_and_test_cl(model, train_loader, test_loader, optimizer, criterion, scheduler, args)
        msg = 'Best model @ epoch {}'.format(args.best_epoch)
        print(msg, file=open(args.results_dir + args.results_file, "a"))
        if args.verbose > 0:
            loginfo(msg)

        # evaluate best model once again
        model.load_state_dict(torch.load(args.models_dir + 'best-%s.pt' % args.model))
        loss_mean, loss_std, acc, preds = test_cl(model, None, test_loader, criterion, args)

    elif args.test:
        # only test model on dataset
        loss_mean, loss_std, acc, preds = test_cl(model, None, test_loader, criterion, args)
        print('{:.4f},{:.4f},{:.4f}'.format(loss_mean, loss_std, acc),
              file=open(args.results_dir + args.results_file[:-4] + '.csv', "a"))

    elif args.pred:
        # evaluate model without labels
        pred_list = []
        for data in test_loader:
            output, _ = model(data)
            pred = output.argmax(dim=1) + 1
            pred_list.extend(pred.tolist())
        np.savetxt(args.results_dir + "pred_" + args.results_file, pred_list, delimiter=',', fmt='%f')
