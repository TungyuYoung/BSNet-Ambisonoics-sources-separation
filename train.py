import argparse
import multiprocessing
import os
import json
import numpy as np
import torch
import torch.optim as optim
from pathlib import Path
from dataset import Dataset
from network import BSNet, load_pretrain
from scipy.io import wavfile
import time


def train_epoch(model, device, optimizer, train_loader, epoch):
    model.train()
    losses = []
    interval_losses = 0
    mse_losses = 0
    si_sdr_losses = 0
    count = 0

    for batch_idx, (ambi_mixes, target_signals, target_direction, beamformer_audio) in enumerate(train_loader):
        ambi_mixes = ambi_mixes.clone().to(device)
        target_signals = target_signals.clone().to(device)
        optimizer.zero_grad()
        output_signal, mask_ = model(ambi_mixes, beamformer_audio)
        loss, mse_loss, si_sdr_loss = model.loss(output_signal, target_signals)
        interval_losses += loss.item()
        mse_losses += mse_loss.item()
        si_sdr_losses += np.sum(si_sdr_loss)

        count = count + 1

        loss.backward()

        optimizer.step()

        if batch_idx % 10 == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)] \t LOSS: {:.6f} \t MSE LOSS: {:.6f} \t SI-SDR-LOSS: {:.6f}".format(
                epoch, batch_idx * len(ambi_mixes), len(train_loader.dataset),
                100.0 * batch_idx / len(train_loader), interval_losses / count, mse_losses / count,
                si_sdr_losses / count))


            losses.append(interval_losses)

    return np.mean(losses)


def testt_epoch(model, device, test_loader, args, epoch):
    model.eval()
    test_loss = 0
    output_folder = os.path.join(args.checkpoints_dir, args.name, 'samples')

    with torch.no_grad():
        for batch_idx, (ambi_mixes, target_signals, target_direction, beamformer_audio) in enumerate(test_loader):
            ambi_mixes = ambi_mixes.clone().to(device)
            ambi_mixes_original = ambi_mixes
            target_signals = target_signals.clone().to(device)
            beamformer_audio = beamformer_audio.clone().to(device)

            output_signal, mask_ = model(ambi_mixes, beamformer_audio)

            if batch_idx == 0 and epoch % 10 == 0:
                for b in range(output_signal.shape[0]):
                    output_signal_np = output_signal.detach().cpu().numpy()
                    target_signals_np = target_signals.detach().cpu().numpy()
                    ambi_mixes_original_np = ambi_mixes_original.detach().cpu().numpy()

                    output_signal_np = output_signal_np * np.iinfo(np.int16).max
                    target_signals_np = target_signals_np * np.iinfo(np.int16).max
                    ambi_mixes_original_np = ambi_mixes_original_np * np.iinfo(np.int16).max

                    wavfile.write(os.path.join(output_folder, 'epoch_' + str(epoch) + '_batch_pos_' + str(b) +
                                               '_output_signal.wav'), args.sr,
                                  output_signal_np[b, ...].T.astype(np.int16))
                    wavfile.write(os.path.join(output_folder,
                                               'epoch_' + str(epoch) + '_batch_pos_' + str(
                                                   b) + '_label_source_signal.wav'),
                                  args.sr, (0.45 * target_signals_np[b, ...]).T.astype(np.int16))

                    wavfile.write(os.path.join(output_folder,
                                               'epoch_' + str(epoch) + '_batch_pos_' + str(b) + '_input_mixture.wav'),
                                  args.sr, ambi_mixes_original_np[b, ...].T.astype(np.int16))

            loss, mse_loss, si_sdr_losses = model.loss(output_signal, target_signals)
            test_loss += loss.item()

            if batch_idx % 20 == 0:
                print("Test Loss: {}".format(loss))

        time.sleep(5)

        test_loss /= len(test_loader)
        print("\nTest set: Average Loss: {:.4f}\n".format(test_loss))

        return test_loss


def train(args):
    torch.cuda.empty_cache()
    args.sr = 44100
    data_train = Dataset(args.train_dir, sr=args.sr, ambiorder=args.ambiorder, angular_window_deg=2.5,
                         ambimode=args.ambimode, dataset=args.dataset)

    data_test = Dataset(args.test_dir, sr=args.sr, ambiorder=args.ambiorder, angular_window_deg=2.5,
                        ambimode=args.ambimode, dataset=args.dataset)

    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    num_workers = min(multiprocessing.cpu_count(), args.n_workers)
    kwargs = {
        'num_workers': num_workers,
        'pin_memory': True
    }

    train_loader = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=args.batch_size, **kwargs)

    print("SETTING UP MODEL")

    if args.ambimode == 'implicit':
        model = BSNet()
    elif args.ambimode == 'mixed':
        model = BSNet()

    model.to(device)

    print("MODEL SET UP")

    if not os.path.exists(os.path.join(args.checkpoints_dir, args.name)):
        os.makedirs(os.path.join(args.checkpoints_dir, args.name))
    if not os.path.exists(os.path.join(args.checkpoints_dir, args.name, 'samples')):
        os.makedirs(os.path.join(args.checkpoints_dir, args.name, 'samples'))

    commandline_args_path = os.path.join(args.checkpoints_dir, args.name, 'commandline_args.txt')
    with open(commandline_args_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.decay_rate)

    if args.pretrain_path:
        print('LOADING PRETRAINED')
        state_dict = torch.load(args.pretrain_path)
        load_pretrain(model, state_dict)
        print('PRETRAINED LOADED')

    if args.start_epoch is not None:
        assert args.start_epoch > 0, "start_epoch must be greater than 0!"
        start_epoch = args.start_epoch
        checkpoint_path = Path(args.checkpoints_dir) / "{}.pt".format(start_epoch - 1)
        state_dict = torch.load(checkpoint_path)
    else:
        start_epoch = 0

    best_error = float("inf")
    train_losses = []
    test_losses = []

    loss_dict = {'train': [], 'test': []}
    print("GOING TO TRAINING LOOP")

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            train_loss = train_epoch(model, device, optimizer, train_loader, epoch)
            print("This epoch total loss: ", train_loss)
            torch.save(model.state_dict(), os.path.join(args.checkpoints_dir, args.name, "last.pt"))
            print("Done with training, start to testing!")
            test_loss = testt_epoch(model, device, test_loader, args, epoch)

            if test_loss < best_error:
                best_error = test_loss
                torch.save(model.state_dict(), os.path.join(args.checkpoints_dir, args.name, "best.pt"))

            scheduler.step()
            train_losses.append((epoch, train_loss))
            test_losses.append((epoch, test_loss))

            loss_dict['train'].append(train_loss)
            loss_dict['test'].append(test_loss)
            json_path = os.path.join(args.checkpoints_dir, args.name, 'loss.json')
            with open(json_path, 'w') as fp:
                json.dump(loss_dict, fp)

        return train_losses, test_losses

    except KeyboardInterrupt:
        import traceback
        traceback.print_exc()
    except Exception as _:
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("Using cuda0!")
    parser = argparse.ArgumentParser()
    # Data Params
    parser.add_argument('--decay_step', type=int, default=10, help='Learning rate decay steps.')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='Learning rate decay rate.')
    parser.add_argument('train_dir', type=str,
                        help="Path to the training dataset")
    parser.add_argument('test_dir', type=str,
                        help="Path to the testing dataset")
    parser.add_argument('--name', type=str, default="multimic_experiment",
                        help="Name of the experiment")
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints',
                        help="Path to the checkpoints")
    parser.add_argument('--batch_size', type=int, default=8,
                        help="Batch size")
    parser.add_argument('--ambiorder', type=int, default=4,
                        help="Ambisonics order")
    parser.add_argument('--ambimode', type=str, default='implicit',
                        help="Ambisonics mode. 'implicit': raw Ambisonics mixture as input. "
                             "'mixed': raw first order Ambisonics mixture and bf concatenated.")
    parser.add_argument('--dataset', type=str, default="musdb",
                        help="Dataset to train")

    # Training Params
    parser.add_argument('--epochs', type=int, default=350,
                        help="Number of epochs")
    parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
    parser.add_argument('--sr', type=int, default=44100, help="Sampling rate")
    parser.add_argument('--decay', type=float, default=0, help="Weight decay")
    parser.add_argument('--n_workers', type=int, default=16,
                        help="Number of parallel workers")
    parser.add_argument('--print_interval', type=int, default=20,
                        help="Logging interval")
    parser.add_argument('--start_epoch', type=int, default=None,
                        help="Start epoch")
    parser.add_argument('--pretrain_path', type=str,
                        help="Path to pretrained weights")
    parser.add_argument('--use_cuda', dest='use_cuda', action='store_true',
                        help="Whether to use cuda")


    class Args():
        def __init__(self):
            super().__init__()
            self.train_dir = '../musdb18/mini_dataset_ambi/train/'
            self.test_dir = '../musdb18/mini_dataset_ambi/test/'
            self.name = 'multimic_minidataset_ambi_loss_fremse_nonchubc'
            self.checkpoints_dir = './checkpoints_minidataset_fre_cl_'
            self.batch_size = 8
            self.ambiorder = 4
            self.ambimode = 'mixed'
            self.dataset = 'musdb'
            self.epochs = 350
            self.lr = 1e-3
            self.decay_step = 3000
            self.decay_rate = 0.1
            self.sr = 44100
            self.decay = 0
            self.n_workers = 0
            self.print_interval = 10
            self.start_epoch = None
            self.pretrain_path = None
            self.use_cuda = True


    args = Args()
    train(args)
