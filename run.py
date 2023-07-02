import argparse
import random
import numpy as np
import torch
import pathlib
import tqdm
from torch import nn, optim
import json
import glob
from scipy.spatial.distance import jensenshannon
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize

from my_utils import get_datadir, load_dataset
from dataset import TrajectoryDataset, make_padded_collate, RealFakeDataset, make_padded_collate_for_GANs
from models import SelfAttention, make_sample
from rollout import Rollout
from loss import GANLoss
from data_pre_processing import save_state_with_nan_padding

from pytorchtools import EarlyStopping

from logging import getLogger, config
from rollout import Rollout
from models import Discriminator
from loss import GANLoss


def compute_next_location_distribution(target, trajectories, n_locations):
    # compute the next location probability for each location
    count = Counter([trajectory[i+1] for trajectory in trajectories for i in range(len(trajectory)-1) if trajectory[i]==target])
    summation = sum(count.values())
    if summation == 0:
        return None
    distribution = []
    for i in range(n_locations):
        if i not in count:
            distribution.append(0)
        else:
            distribution.append(count[i]/summation)
    return distribution

def compute_location_t_distribution(trajs, t, n_locations):
    locatoin_ts = [v[t] for v in trajs if len(v) > t]
    counts = Counter(locatoin_ts)
    distribution = np.zeros(n_locations)
    for i in range(n_locations):
        if i in counts:
            distribution[i] = counts[i]
        else:
            distribution[i] = 0
    distribution = distribution / sum(distribution)
    return distribution

def train(generator, optimizer, loss_model, input, target):
    generator.train()

    output = generator(input)
    output_v = output.view(-1,output.shape[-1])
    loss = loss_model(output_v, target.reshape(-1))
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    return loss.item()


def evaluate():
 
    # count the number of appearance of top 10 locations in meta_traj
    counts = []
    for location in top_10_locations:
        count = 0
        for traj in test_traj:
            count += traj.count(location)
        counts.append(count)

    next_location_jss = []
    for i, location in enumerate(top_10_locations):
        next_location_distribution = compute_next_location_distribution(location, test_traj, dataset.n_locations)
        if next_location_distribution is not None:
            next_location_jss.append(jensenshannon(next_location_distribution, next_location_distributions[i])**2)
            # next_location_emds.append(earth_movers_distance_pmf(next_location_distribution, next_location_distributions[i], distance_matrix))
        else:
            next_location_jss.append(1)
            # next_location_emds.append(max_distance)
            next_location_distribution = np.zeros((n_bins+2)**2)

        # visualize the next location distribution
        values = np.array(next_location_distribution).reshape(n_bins+2, n_bins+2)
        plt.figure(figsize=(10, 10))
        ax = sns.heatmap(values, cmap="YlGnBu", vmin=0, vmax=values.max(), square=True, cbar_kws={"shrink": 0.8})
        plt.savefig(save_path / f"next_location_distribution_{i}.png")
    logger.info(f'the apparances of top 10 locations: {counts} | next_location_js:={next_location_jss}')

    ts = [0,1,2]
    for t in ts:
        # compute the distribution of location at time t
        real_location_t_distribution = compute_location_t_distribution(trajectories, t, dataset.n_locations)
        gene_location_t_distribution = compute_location_t_distribution(test_traj, t, dataset.n_locations)
        # compare pr(location_t)
        location_t_js = jensenshannon(real_location_t_distribution, gene_location_t_distribution)**2
        logger.info(f'js divergence of location_{t}_js:={location_t_js}')

        # visualize the distribution
        values = np.array(gene_location_t_distribution).reshape(n_bins+2, n_bins+2)
        plt.figure(figsize=(10, 10))
        ax = sns.heatmap(values, cmap="YlGnBu", vmin=0, vmax=values.max(), square=True, cbar_kws={"shrink": 0.8})
        plt.savefig(save_path / f"gene_location_{t}_distribution.png")


        # visualize the distribution
        values = np.array(real_location_t_distribution).reshape(n_bins+2, n_bins+2)
        plt.figure(figsize=(10, 10))
        ax = sns.heatmap(values, cmap="YlGnBu", vmin=0, vmax=values.max(), square=True, cbar_kws={"shrink": 0.8})
        plt.savefig(save_path / f"location_{t}_distribution.png")

if __name__ == "__main__":
    # set argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_number', type=int)
    parser.add_argument('--n_pre_training_epochs', type=int)
    parser.add_argument('--n_epochs', type=int)
    parser.add_argument('--n_generated', type=int)
    parser.add_argument('--n_generated_for_discriminator', type=int)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--data_name', type=str)
    parser.add_argument('--training_data_name', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--embed_dim', type=int)
    parser.add_argument('--save_name', type=str)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--patience', type=int)
    parser.add_argument('--pre_training', action='store_true')
    parser.add_argument('--add_residual', action='store_true')
    args = parser.parse_args()

    args.algorithm = "MoveSim"
    
    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms = True
    torch.backends.cudnn.deterministic = True
    
    # set directory
    data_dir = get_datadir()
    data_path = data_dir / args.dataset / args.data_name / args.training_data_name
    save_path = data_dir / "results" / args.dataset / args.data_name / args.training_data_name / args.save_name
    save_path.mkdir(exist_ok=True, parents=True)

    # set logger
    with open('./log_config.json', 'r') as f:
        log_conf = json.load(f)
    log_conf["handlers"]["fileHandler"]["filename"] = str(save_path / "log.log")
    config.dictConfig(log_conf)
    logger = getLogger(__name__)
    logger.info('log is saved to {}'.format(save_path / "log.log"))
    logger.info(f'used parameters {vars(args)}')



    # load dataset config    
    with open(data_path / "params.json", "r") as f:
        param = json.load(f)
    n_bins = param["n_bins"]
    lat_range = param["lat_range"]
    lon_range = param["lon_range"]

    training_data_path = data_path / "training_data.csv"
    trajectories = load_dataset(training_data_path, logger=logger)
    logger.info(f"len of trajectories: {len(trajectories)}")

    max_seq_len = max([len(trajectory) for trajectory in trajectories])
    logger.info(f"max seq len: {max_seq_len}")

    dataset = TrajectoryDataset(trajectories, n_bins)
    n_vocabs = len(dataset.vocab)

    if args.n_generated == 0:
        args.n_generated = len(dataset)
        logger.info("generating " + str(args.n_generated) + " samples")

    transition_matrix = torch.tensor(normalize(np.load(get_datadir() / args.dataset / args.data_name / args.training_data_name / 'transition_matrix.npy'))).cuda(args.cuda_number)
    distance_matrix = torch.tensor(normalize(np.load(get_datadir() / args.dataset / args.data_name / args.training_data_name / 'distance_matrix.npy'))).cuda(args.cuda_number)

    embed_size = args.embed_dim
    linear_dim = args.embed_dim
    inner_ff_size = embed_size*4
    n_heads = 1
    n_code = 1
    drop_out = 0.1
    generator = SelfAttention(n_code, n_heads, embed_size, inner_ff_size, linear_dim, dataset.n_locations, n_vocabs, n_vocabs, max_seq_len+1, transition_matrix, distance_matrix, 0.1, add_residual=args.add_residual).cuda(args.cuda_number)

    # find the top_k appearing locations in the dataset
    top_k = 50
    top_10_locations_count = Counter([location for trajectory in trajectories for location in trajectory]).most_common(top_k)
    top_10_locations = [location for location, _ in top_10_locations_count]
    logger.info(f"top {top_k} locations: " + str(top_10_locations_count))

    # compute the next location probability for each location
    next_location_distributions = []
    for i, location in enumerate(top_10_locations):
        next_location_distribution = compute_next_location_distribution(location, trajectories, dataset.n_locations)
        if next_location_distribution is None:
            next_location_distribution = np.zeros(dataset.n_locations)
        next_location_distributions.append(next_location_distribution)
        # visualize the next location distribution
        values = np.array(next_location_distribution).reshape(n_bins+2, n_bins+2)
        plt.figure(figsize=(10, 10))
        ax = sns.heatmap(values, cmap="YlGnBu", vmin=0, vmax=values.max(), square=True, cbar_kws={"shrink": 0.8})
        plt.savefig(save_path / f"real_next_location_distribution_{i}.png")

    collate_fn = make_padded_collate(dataset.IGNORE_IDX, dataset.START_IDX, dataset.END_IDX)
    data_loader = torch.utils.data.DataLoader(dataset, num_workers=0, shuffle=True, pin_memory=True, batch_size=args.batch_size, collate_fn=collate_fn)
    optimizer = optim.Adam(generator.parameters(), lr=args.lr, weight_decay=1e-4, betas=(.9,.999))

    logger.info(f"IGNORE_IDX: {dataset.IGNORE_IDX}")
    loss_model = nn.NLLLoss(ignore_index=dataset.IGNORE_IDX)
 
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=save_path / "checkpoint.pt", trace_func=logger.info)
    logger.info(f"early stopping patience: {args.patience}, save path: {save_path / 'checkpoint.pt'}")
    
    generator.train()

    logger.info("start pre_training")
    if args.pre_training:
            
        for epoch in tqdm.tqdm(range(args.n_pre_training_epochs)):

            loss = 0
            counter = 0

            for i, batch in enumerate(data_loader):
                # print(i)
                counter += 1
                if len(batch["input"]) == 0:
                    continue

                input = batch["input"].cuda(args.cuda_number, non_blocking=True)
                target = batch["target"].cuda(args.cuda_number, non_blocking=True)

                batch_size = input.shape[0]
                seq_len = input.shape[1]

                loss += train(generator, optimizer, loss_model, input, target)
                
            early_stopping(loss / counter, generator)
            logger.info(f'epoch: {early_stopping.epoch} | best loss: {early_stopping.best_score} | current loss: {loss / counter}')
            
            if (epoch == args.n_epochs-1) or early_stopping.early_stop:
                
                if early_stopping.early_stop:
                    logger.info("finish because of early stopping")
                    generator.load_state_dict(torch.load(early_stopping.path))
                else:
                    logger.info("finish because of epoch")
                    generator.load_state_dict(torch.load(early_stopping.path))
                    generator.eval()

                generator.eval()

                test_traj =  make_sample(args.batch_size, generator, args.n_generated, dataset, real_start=False, remove_end=True)

                evaluate()

                generated_data_path = save_path / f"gene.csv"
                save_state_with_nan_padding(generated_data_path, test_traj)
                logger.info(f"save generated data to {generated_data_path}")

                generator.train()

                break
    
    logger.info("start training")

    discriminator = Discriminator(traj_length=max_seq_len, n_vocabs=n_vocabs).cuda(args.cuda_number)
    rollout = Rollout(generator, 0.8, args.cuda_number, dataset.START_IDX, dataset.END_IDX)

    gen_gan_loss = GANLoss()
    gen_gan_optm = optim.Adam(generator.parameters(),lr=0.0001)

    dis_loss = nn.NLLLoss()
    dis_optm = optim.Adam(discriminator.parameters(),lr=0.0001)

    for epoch in range(args.n_epochs):

        # Train the generator for one step
        for it in range(1):
            gene_trajs =  torch.tensor(make_sample(args.batch_size, generator, args.batch_size, dataset, real_start=False, remove_end=False), dtype=torch.long)
            # input is a dataset that is made by adding start_index to the front of the trajectory
            starts = torch.ones((args.batch_size, 1)).type(torch.LongTensor) * dataset.START_IDX
            inputs = torch.cat([starts, gene_trajs], dim=1)[:, :-1].contiguous().cuda(args.cuda_number)
            targets = gene_trajs.contiguous().view((-1,)).cuda(args.cuda_number)

            # calculate the reward
            rewards = rollout.get_reward(gene_trajs, 16, discriminator, dataset.n_locations)
            rewards = torch.Tensor(rewards)
            rewards = torch.exp(rewards.cuda(args.cuda_number)).contiguous().view((-1,))
            prob = torch.exp(generator(inputs))
            gloss = gen_gan_loss(prob, targets, rewards, args.cuda_number)

            # remove ploss
            # if ploss_alpha != 0.:
            #     p_crit = period_loss(24)
            #     p_crit = p_crit.to(device)
            #     pl = p_crit(samples.float())
            #     gloss += ploss_alpha * pl
            # remove dloss
            # if dloss_alpha != 0.:
            #     d_crit = distance_loss(device=device,datasets=opt.data)
            #     d_crit = d_crit.to(device)
            #     dl = d_crit(samples.float())
            #     gloss += dloss_alpha * dl

            gen_gan_optm.zero_grad()
            gloss.backward()
            gen_gan_optm.step()
        
        rollout.update_params()
        for _ in range(4):
            gene_trajs =  make_sample(args.batch_size, generator, args.n_generated_for_discriminator, dataset, real_start=False, remove_end=False)
            dis_dataset = RealFakeDataset(trajectories, gene_trajs)
            dis_data_loader = torch.utils.data.DataLoader(dis_dataset, num_workers=0, shuffle=True, pin_memory=True, batch_size=args.batch_size, collate_fn=make_padded_collate_for_GANs(dataset.END_IDX))
            for _ in range(2):
                total_loss = 0.
                for data, target in dis_data_loader:
                    data = torch.LongTensor(data).cuda(args.cuda_number)
                    target = torch.LongTensor(target).contiguous().view(-1).cuda(args.cuda_number)
                    pred = discriminator(data)
                    dloss = dis_loss(pred, target)
                    total_loss += dloss.item()
                    dis_optm.zero_grad()
                    dloss.backward()
                    dis_optm.step()

        generator.eval()
        test_traj =  make_sample(args.batch_size, generator, args.n_generated, dataset, real_start=False, remove_end=True)
        evaluate()
        generated_data_path = save_path / f"gene.csv"
        save_state_with_nan_padding(generated_data_path, test_traj)
        logger.info(f"save generated data to {generated_data_path}")
        generator.train()

        logger.info('Epoch [%d] Generator Loss: %f, Discriminator Loss: %f' % (epoch, gloss.item(), dloss.item()))

    args.end_epoch = early_stopping.epoch
    args.end_loss = early_stopping.best_score

    # concat vars(args) and param
    param.update(vars(args))
    logger.info(f"save param to {save_path / 'params.json'}")
    logger.info(f"args: {param}")
    with open(save_path / "params.json", "w") as f:
        json.dump(param, f)
        