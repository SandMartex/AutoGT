import os
import time
import pytz
import json
import torch
import random
import datetime
import numpy as np
from data import get_dataset
import torch.nn.functional as F
from argparse import ArgumentParser
from model import get_model, save_model, load_model


def get_directory(args):
    directory = f'./checkpoints/{args.dataset_name}/{str(args.seed)}/{str(args.data_split)}/'
    return directory


def gen_layer(dic, params):
    layer = []
    hidden_in = random.choice(dic['hidden_in'])
    num_heads = random.choice(dic['num_heads'])
    att_size = random.choice(dic['att_size'])
    hidden_mid = random.choice(dic['hidden_mid'])
    ffn_size = random.choice(dic['ffn_size'])
    mask = random.choice(dic['mask'])
    layer.append([hidden_in, num_heads, att_size, hidden_mid, ffn_size, mask])
    cen = random.choice([True, False])
    eig = random.choice([True, False])
    svd = random.choice([True, False])
    layer.append([cen, eig, svd])
    spa = params[1][0][2][0]
    edg = params[1][0][2][1]
    pma = random.choice([True, False])
    layer.append([spa, edg, pma])
    return layer


def gen_params(path, spa, edg, pma):
    with open(path, 'r') as f:
        dic = json.load(f)
    depth = random.choice(dic['depth'])
    layers = []
    for _ in range(0, depth):
        layer = []
        hidden_in = random.choice(dic['hidden_in'])
        num_heads = random.choice(dic['num_heads'])
        att_size = random.choice(dic['att_size'])
        hidden_mid = random.choice(dic['hidden_mid'])
        ffn_size = random.choice(dic['ffn_size'])
        mask = random.choice(dic['mask'])
        layer.append((hidden_in, num_heads, att_size, hidden_mid, ffn_size, mask))
        cen = random.choice([True, False])
        eig = random.choice([True, False])
        svd = random.choice([True, False])
        layer.append((cen, eig, svd))
        spa = spa > 0
        edg = edg > 0
        pma = pma > 0
        layer.append((spa, edg, pma))
        layers.append(tuple(layer))
    return (depth, tuple(layers))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def initial_args():
    parent_parser = ArgumentParser()
    parser = parent_parser.add_argument_group("GraphTransformer")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--ffn_dim', type=int, default=32)
    parser.add_argument('--intput_dropout_rate', type=float, default=0.1)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--attention_dropout_rate',type=float, default=0.1)
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--warmup_updates', type=int, default=6)
    parser.add_argument('--tot_updates', type=int, default=50)
    parser.add_argument('--peak_lr', type=float, default=2e-4)
    parser.add_argument('--end_lr', type=float, default=1e-9)
    parser.add_argument('--edge_type', type=str, default='multi_hop')
    parser.add_argument('--validate', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--dataset_name', type=str, default='PROTEINS')
    parser.add_argument('--multi_hop_max_dist', type=int, default=5)
    parser.add_argument('--num_class', type=int, default=1)
    parser.add_argument('--path', type=str, default='json/PROTEINS.json')

    parser = parent_parser.add_argument_group("Dataset")
    parser.add_argument('--lap_enc_dim', type=int, default=10)
    parser.add_argument('--svd_enc_dim', type=int, default=16)
    parser.add_argument('--pma_dim', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_node', type=int, default=512)
    parser.add_argument('--data_split', type=int, default=0)

    parser = parent_parser.add_argument_group("Training")
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--split_epochs', type=int, default=2)
    parser.add_argument('--end_epochs', type=int, default=4)
    parser.add_argument('--retrain_epochs', type=int, default=4)

    parser = parent_parser.add_argument_group("Evolution")
    parser.add_argument('--evol_epochs', type=int, default=20)
    parser.add_argument('--population_num', type=int, default=100)
    parser.add_argument('--m_prob', type=float, default=0.3)
    parser.add_argument('--mutation_num', type=int, default=20)
    parser.add_argument('--hybridization_num', type=int, default=20)
    parser.add_argument('--retrain_num', type=int, default=4)

    return parent_parser.parse_args()


def retrain_graphormer(args, train_loader, valid_loader, test_loader_):
    start = time.time()
    model, optimizer, scheduler = get_model(args)
    model.to_graphormer()
    best_test_ = 0
    worst_test = 0
    best_valid = 0
    results = np.zeros((args.retrain_epochs, 3))
    for epoch in range(args.retrain_epochs):
        loss = train(model, train_loader, optimizer, scheduler)
        train_loss, train_acc = test(model, train_loader)
        valid_loss, valid_acc = test(model, valid_loader)
        test__loss, test_acc_ = test(model, test_loader_)
        print("Epoch {: >3}: Train Loss: {:.3f}, Train Acc: {:.2%}, Valid Loss: {:.3f}, Valid Acc: {:.2%}, Test Loss: {:.3f}, Test Acc: {:.2%}".format(epoch, loss, train_acc, valid_loss, valid_acc, test__loss, test_acc_))
        if valid_acc > best_valid:
            best_valid = valid_acc
            best_test_ = test_acc_
            worst_test = test_acc_
        elif valid_acc == best_valid:
            if test_acc_ > best_test_:
                best_test_ = test_acc_
            elif test_acc_ < worst_test:
                worst_test = test_acc_
        results[epoch, 0] = train_acc
        results[epoch, 1] = valid_acc
        results[epoch, 2] = test_acc_
    train_top = np.max(results[:, 0])
    valid_top = np.max(results[:, 1])
    test_top_ = np.max(results[:, 2])
    last_test_ = results[-1, 2]
    end = time.time()
    print(f"Graphormer Retrain Ended!")
    print(f"Use time: {end - start} s")
    print("Best Result: {:.2%}, Worst Result: {:.2%}, Last Result: {:.2%}".format(best_test_, worst_test, last_test_))
    print("Best Train: {:.2%}, Best Valid: {:.2%}, Best Test: {:.2%}".format(train_top, valid_top, test_top_))
    return best_test_, worst_test, last_test_, results


def retrain(args, train_loader, valid_loader, test_loader_, params=None):
    model, optimizer, scheduler = get_model(args)
    start = time.time()
    best_test_ = 0
    worst_test = 0
    best_valid = 0
    results = np.zeros((args.retrain_epochs, 3))
    for epoch in range(args.retrain_epochs):
        loss = train(model, train_loader, optimizer, scheduler, params)
        train_loss, train_acc = test(model, train_loader)
        valid_loss, valid_acc = test(model, valid_loader)
        test__loss, test_acc_ = test(model, test_loader_)
        print("Epoch {: >3}: Train Loss: {:.3f}, Train Acc: {:.2%}, Valid Loss: {:.3f}, Valid Acc: {:.2%}, Test Loss: {:.3f}, Test Acc: {:.2%}".format(epoch, loss, train_acc, valid_loss, valid_acc, test__loss, test_acc_))
        if valid_acc > best_valid:
            best_valid = valid_acc
            best_test_ = test_acc_
            worst_test = test_acc_
        elif valid_acc == best_valid:
            if test_acc_ > best_test_:
                best_test_ = test_acc_
            elif test_acc_ < worst_test:
                worst_test = test_acc_
        results[epoch, 0] = train_acc
        results[epoch, 1] = valid_acc
        results[epoch, 2] = test_acc_
    train_top = np.max(results[:, 0])
    valid_top = np.max(results[:, 1])
    test_top_ = np.max(results[:, 2])
    last_test_ = results[-1, 2]
    end = time.time()
    print(f"Graph Transformer Retrain For Params={params} Ended!")
    print(f"Use time: {end - start} s")
    print("Best Result: {:.2%}, Worst Result: {:.2%}, Last Result: {:.2%}".format(best_test_, worst_test, last_test_))
    print("Best Train: {:.2%}, Best Valid: {:.2%}, Best Test: {:.2%}".format(train_top, valid_top, test_top_))
    return best_test_, worst_test, last_test_, results


def graph_transformer(epochs, model, optimizer, scheduler, train_loader, valid_loader, test_loader_):
    start = time.time()
    best_test_ = 0
    worst_test = 0
    best_valid = 0
    results = np.zeros((epochs, 3))
    for epoch in range(epochs):
        loss = train(model, train_loader, optimizer, scheduler)
        train_loss, train_acc = test(model, train_loader)
        valid_loss, valid_acc = test(model, valid_loader)
        test__loss, test_acc_ = test(model, test_loader_)
        print("Epoch {: >3}: Train Loss: {:.3f}, Train Acc: {:.2%}, Valid Loss: {:.3f}, Valid Acc: {:.2%}, Test Loss: {:.3f}, Test Acc: {:.2%}".format(epoch, loss, train_acc, valid_loss, valid_acc, test__loss, test_acc_))
        if valid_acc > best_valid:
            best_valid = valid_acc
            best_test_ = test_acc_
            worst_test = test_acc_
        elif valid_acc == best_valid:
            if test_acc_ > best_test_:
                best_test_ = test_acc_
            elif test_acc_ < worst_test:
                worst_test = test_acc_
        results[epoch, 0] = train_acc
        results[epoch, 1] = valid_acc
        results[epoch, 2] = test_acc_
    train_top = np.max(results[:, 0])
    valid_top = np.max(results[:, 1])
    test_top_ = np.max(results[:, 2])
    end = time.time()
    print(f"Graph Transformer Training And Testing Ended! Use time: {end - start} s")
    print("Best Result: {:.2%}, Worst Result: {:.2%}".format(best_test_, worst_test))
    print("Best Train: {:.2%}, Best Valid: {:.2%}, Best Test: {:.2%}".format(train_top, valid_top, test_top_))
    return best_test_, worst_test


def train_supernet(model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    for batched_data in train_loader[1:]:
        optimizer.zero_grad()
        y_hat = model(batched_data, model.gen_params()).squeeze()
        y_gt = batched_data.y.view(-1)
        loss = F.binary_cross_entropy_with_logits(y_hat, y_gt.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.double() * batched_data.y.shape[0]
        scheduler.step()
    return total_loss / train_loader[0]


def train(model, train_loader, optimizer, scheduler, params=None):
    model.train()
    total_loss = 0
    for batched_data in train_loader[1:]:
        optimizer.zero_grad()
        y_hat = model(batched_data, params).squeeze()
        y_gt = batched_data.y.view(-1)
        loss = F.binary_cross_entropy_with_logits(y_hat, y_gt.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.double() * batched_data.y.shape[0]
        scheduler.step()
    return total_loss / train_loader[0]


@torch.no_grad()
def test(model, data_loader, params=None):
    model.eval()
    total_correct = 0
    total_loss = 0
    for batched_data in data_loader[1:]:
        out = model(batched_data, params).squeeze()
        total_correct += int(((out > 0.5) == batched_data.y).sum())
        loss = F.binary_cross_entropy_with_logits(out, batched_data.y.view(-1).float())
        total_loss += loss.double() * batched_data.y.shape[0]
    return total_loss / data_loader[0], total_correct / data_loader[0]


def cli_main():
    # start time
    start = time.time()
    print(datetime.datetime.fromtimestamp(int(time.time()), pytz.timezone('Asia/Shanghai')).strftime('Start time: %Y.%m.%d %H:%M:%S'))
    # args
    args = initial_args()
    print("==================== Arguments Start ====================")
    for key in args.__dict__:
        print(key + ": " + str(args.__dict__[key]))
    print("===================== Arguments End =====================")
    # set seed
    seed_everything(args.seed)
    # dataset
    train_loader, valid_loader, test_loader_ = get_dataset(args, args.data_split)
    # model
    model, optimizer, scheduler = get_model(args)
    print('Total Params:', sum(parameter.numel() for parameter in model.parameters()))
    # training and testing
    graph_transformer(10, model, optimizer, scheduler, train_loader, valid_loader, test_loader_)
    # end time
    end = time.time()
    print(datetime.datetime.fromtimestamp(int(time.time()), pytz.timezone('Asia/Shanghai')).strftime('End time: %Y.%m.%d %H:%M:%S'))
    print("Program Ended! Use time: {} s".format(end - start))


def few_shot(args):
    # start time
    start = time.time()
    print(datetime.datetime.fromtimestamp(int(time.time()), pytz.timezone('Asia/Shanghai')).strftime('Start time: %Y.%m.%d %H:%M:%S'))
    # args
    # args = initial_args()
    print("==================== Arguments Start ====================")
    for key in args.__dict__:
        print(key + ": " + str(args.__dict__[key]))
    print("===================== Arguments End =====================")
    # set seed
    seed_everything(args.seed)
    # dataset
    train_loader, valid_loader, test_loader_ = get_dataset(args, args.data_split)
    # model
    model, optimizer, scheduler = get_model(args)
    # few shot training
    start_ = time.time()
    for epoch in range(args.split_epochs):
        train_supernet(model, train_loader, optimizer, scheduler)
    end_ = time.time()
    print("Few Shot Training Ended! Use time: {} s".format(end_ - start_))
    # save model
    directory = get_directory(args)
    if not os.path.exists(directory):
        os.makedirs(directory)
    name = 'supernet.pt'
    save_model(model, optimizer, scheduler, directory + name)
    # load model and split
    for ord in range(8):
        spa = int((ord & 1) != 0)
        edg = int((ord & 2) != 0)
        pma = int((ord & 4) != 0)
        start_ = time.time()
        model, optimizer, scheduler = load_model(args, directory + name)
        for epoch in range(args.split_epochs, args.end_epochs):
            params = gen_params(args.path, spa, edg, pma)
            train(model, train_loader, optimizer, scheduler, params)
        sub_name = 'supernet_' + str(ord) + '.pt'
        save_model(model, optimizer, scheduler, directory + sub_name)
        end_ = time.time()
        print("Few Shot Training for {} Encoding Ended! Use time: {} s".format(ord, end_ - start_))
    # end time
    end = time.time()
    print(datetime.datetime.fromtimestamp(int(time.time()), pytz.timezone('Asia/Shanghai')).strftime('End time: %Y.%m.%d %H:%M:%S'))
    print("Program Ended! Use time: {} s".format(end - start))
    # start time
    start = time.time()
    print(datetime.datetime.fromtimestamp(int(time.time()), pytz.timezone('Asia/Shanghai')).strftime('Start time: %Y.%m.%d %H:%M:%S'))
    # evolution
    evolution(args, directory, train_loader, valid_loader, test_loader_)
    # end time
    end = time.time()
    print(datetime.datetime.fromtimestamp(int(time.time()), pytz.timezone('Asia/Shanghai')).strftime('End time: %Y.%m.%d %H:%M:%S'))
    print("Program Ended! Use time: {} s".format(end - start))


def get_ord(params):
    spa = int(params[1][0][2][0])
    edg = int(params[1][0][2][1])
    pma = int(params[1][0][2][2])
    cen = int(params[1][0][1][0])
    ord = spa + (edg << 1)# + (pma << 2) + (cen << 3)
    return ord


def evolution(args, directory, train_loader, valid_loader, test_loader_):
    start = time.time()
    with open(args.path, 'r') as f:
        dic = json.load(f)
    models = []
    for ord in range(4):
        sub_name = 'supernet_' + str(ord) + '.pt'
        models.append(load_model(args, directory + sub_name)[0])

    information = {}
    population = []
    candidates = []

    def is_legal(params):
        if params in information:
            return False
        info = {}
        ord = get_ord(params)
        _, valid_acc = test(models[ord], valid_loader, params)
        _, test_acc_ = test(models[ord], test_loader_, params)
        info['valid_acc'] = valid_acc
        info['test_acc_'] = test_acc_
        print('Top-1 Valid Accuracy = {}, Top-1 Test Accuracy = {}, Parameters = {}'.format(
            valid_acc, test_acc_, params))
        information[params] = info
        return True

    def get_mutation():
        start = time.time()
        print('Start Mutation!')
        result = []

        def random_function():
            params = random.choice(population)
            depth, layers = params
            layers = [list(list(item) for item in layer) for layer in layers]
            if random.random() < args.m_prob:
                new_depth = random.choice(dic['depth'])
                if new_depth > depth:
                    layers = layers + [gen_layer(dic, params) for _ in range(new_depth - depth)]
                else:
                    layers = layers[:new_depth]
                depth = new_depth
            for i in range(depth):
                if random.random() < args.m_prob:
                    layers[i][0][0] = random.choice(dic['hidden_in'])
                if random.random() < args.m_prob:
                    layers[i][0][1] = random.choice(dic['num_heads'])
                if random.random() < args.m_prob:
                    layers[i][0][2] = random.choice(dic['att_size'])
                if random.random() < args.m_prob:
                    layers[i][0][3] = random.choice(dic['hidden_mid'])
                if random.random() < args.m_prob:
                    layers[i][0][4] = random.choice(dic['ffn_size'])
                if random.random() < args.m_prob:
                    layers[i][0][5] = random.choice(dic['mask'])
                if random.random() < args.m_prob:
                    layers[i][1][0] = random.choice([True, False])
                if random.random() < args.m_prob:
                    layers[i][1][1] = random.choice([True, False])
                if random.random() < args.m_prob:
                    layers[i][1][2] = random.choice([True, False])
            if random.random() < args.m_prob:
                flag = random.choice([True, False])
                for i in range(depth):
                    layers[i][2][0] = flag
            if random.random() < args.m_prob:
                flag = random.choice([True, False])
                for i in range(depth):
                    layers[i][2][1] = flag
            if random.random() < args.m_prob:
                flag = random.choice([True, False])
                for i in range(depth):
                    layers[i][2][2] = flag

            layers = tuple([tuple([tuple(item) for item in layer]) for layer in layers])
            result = tuple([depth, layers])
            return result

        iters = args.mutation_num * 10
        while len(result) < args.mutation_num and iters > 0:
            iters -= 1
            params = random_function()
            if not is_legal(params):
                continue
            result.append(params)

        end = time.time()
        print("End Mutation! Use time: {} s".format(end - start))
        return result

    def get_hybridization():
        start = time.time()
        print('Start Hybridization!')
        result = []

        def random_function():
            params1 = random.choice(population)
            params2 = random.choice(population)
            iters = args.population_num
            while params1[0] != params2[0] and iters > 0:
                iters -= 1
                params1 = random.choice(population)
                params2 = random.choice(population)
            AT_choice = random.choice([params1[1][0][2], params2[1][0][2]])
            layers = []
            for i in range(params1[0]):
                shape_choice = []
                for j in range(6):
                    shape_choice.append(random.choice([params1[1][i][0][j], params2[1][i][0][j]]))
                PE_choice = []
                for j in range(3):
                    PE_choice.append(random.choice([params1[1][i][1][j], params2[1][i][1][j]]))
                layers.append(tuple([tuple(shape_choice), tuple(PE_choice), AT_choice]))
            params = (params1[0], tuple(layers))
            return params

        iters = 10 * args.hybridization_num
        while len(result) < args.hybridization_num and iters > 0:
            iters -= 1
            params = random_function()
            if not is_legal(params):
                continue
            result.append(params)

        end = time.time()
        print("End Hybridization! Use time: {} s".format(end - start))
        return result

    epoch = 0
    while epoch < args.evol_epochs:
        while len(candidates) < args.population_num:
            spa = random.choice([True, False])
            edg = random.choice([True, False])
            pma = random.choice([True, False])
            params = gen_params(args.path, spa, edg, pma)
            if not is_legal(params):
                continue
            candidates.append(params)

        population += candidates
        population.sort(key=lambda x: information[x]['valid_acc'], reverse=True)
        population = population[:args.population_num]

        print('epoch = {} : top {} result'.format(epoch, len(population)))
        for i, params in enumerate(population):
            print('No.{} Top-1 Valid Accuracy = {}, Top-1 Test Accuracy = {}, Parameters = {}'.format(
                i + 1, information[params]['valid_acc'], information[params]['test_acc_'], params))

        epoch += 1
        if epoch != args.evol_epochs:
            candidates = get_mutation() + get_hybridization()

    end = time.time()
    print("Evolution Ended! Use time: {} s".format(end - start))


if __name__ == '__main__':
    # cli_main()
    args = initial_args()
    few_shot(args)
