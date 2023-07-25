# coding=utf-8
import matplotlib.pyplot as plt

from prototypical_batch_sampler import PrototypicalBatchSampler
from prototypical_loss import prototypical_loss as loss_fn
from parser_util import get_parser
from TSD.code.tuh_dataset import get_data_loader

from tqdm import tqdm
import numpy as np
import torch
import os
from vit_pytorch.vit import ViT


def init_seed(opt):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)


def init_sampler(opt, labels, mode):
    if 'train' in mode:
        classes_per_it = opt.classes_per_it_tr
        num_samples = opt.num_support_tr + opt.num_query_tr
    else:
        classes_per_it = opt.classes_per_it_val
        num_samples = opt.num_support_val + opt.num_query_val

    return PrototypicalBatchSampler(labels=labels,
                                    classes_per_it=classes_per_it,
                                    num_samples=num_samples,
                                    iterations=opt.iterations)


def init_dataloader(opt):
    ret = get_data_loader(batch_size=2*(opt.num_support_tr + opt.num_query_tr), save_dir=opt.data_root,
                          return_dataset=True, masking=False)
    tr_dataset, val_dataset, test_dataset, tr_label, val_label, test_label = ret

    tr_sampler = init_sampler(opt, tr_label, mode="train")
    tr_dataloader = torch.utils.data.DataLoader(tr_dataset, batch_sampler=tr_sampler)
    val_sampler = init_sampler(opt, val_label, mode="val")
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_sampler=val_sampler)
    test_sampler = init_sampler(opt, test_label, mode="test")
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_sampler)

    return tr_dataloader, val_dataloader, test_dataloader


def init_vit(opt):
    """
    Initialize the ViT
    """
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    model = ViT(image_size=(3200, 15), patch_size=(80, 5), num_classes=16, dim=16, depth=4, heads=4, mlp_dim=4,
                pool='cls',
                channels=1, dim_head=4, dropout=0.2, emb_dropout=0.2).to(device)
    return model


def init_optim(opt, model):
    '''
    Initialize optimizer
    '''
    return torch.optim.Adam(params=model.parameters(),
                            lr=opt.learning_rate)


def init_lr_scheduler(opt, optim):
    '''
    Initialize the learning rate scheduler
    '''
    return torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                           gamma=opt.lr_scheduler_gamma,
                                           step_size=opt.lr_scheduler_step)


def save_list_to_file(path, thelist):
    with open(path, 'w') as f:
        for item in thelist:
            f.write("%s\n" % item)


def get_mask():
    MASK = np.ones(20, dtype=np.bool)

    # Create a list of indices
    indices = np.arange(20)

    # Randomly shuffle the indices
    indices = indices[np.random.permutation(20)]

    # Select the first 8 indices and assign 0 to the corresponding MASK elements
    MASK[indices[:8]] = 0

    return MASK


def train(opt, tr_dataloader, model, optim, lr_scheduler, val_dataloader=None):
    '''
    Train the model with the prototypical learning algorithm
    '''

    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'

    if val_dataloader is None:
        best_state = None
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_acc = 0

    best_model_path = os.path.join(opt.experiment_root, 'best_model.pth')
    last_model_path = os.path.join(opt.experiment_root, 'last_model.pth')

    for epoch in range(opt.epochs):
        print('=== Epoch: {} ==='.format(epoch))
        tr_iter = iter(tr_dataloader)
        model.train()
        for batch in tqdm(tr_iter):
            optim.zero_grad()
            x, y = batch
            x, y = x.to(device), y.to(device)

            mask = get_mask()
            x[:, mask, :, :] = -1  # mask the channels
            x = x.reshape((x.shape[0], 1, -1, x.shape[3]))

            model_output = model(x)
            loss, acc = loss_fn(model_output, target=y,
                                n_support=opt.num_support_tr)
            loss.backward()
            optim.step()
            train_loss.append(loss.item())
            train_acc.append(acc.item())
        avg_loss = np.mean(train_loss[-opt.iterations:])
        avg_acc = np.mean(train_acc[-opt.iterations:])
        print('Avg Train Loss: {}, Avg Train Acc: {}'.format(avg_loss, avg_acc))
        lr_scheduler.step()
        if val_dataloader is None:
            continue
        val_iter = iter(val_dataloader)
        model.eval()
        for batch in val_iter:
            x, y = batch
            x, y = x.to(device), y.to(device)

            mask = get_mask()
            x[:, mask, :, :] = -1  # mask the channels
            x = x.reshape((x.shape[0], 1, -1, x.shape[3]))

            model_output = model(x)
            loss, acc = loss_fn(model_output, target=y,
                                n_support=opt.num_support_val)
            val_loss.append(loss.item())
            val_acc.append(acc.item())
        avg_loss = np.mean(val_loss[-opt.iterations:])
        avg_acc = np.mean(val_acc[-opt.iterations:])
        postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(
            best_acc)
        print('Avg Val Loss: {}, Avg Val Acc: {}{}'.format(
            avg_loss, avg_acc, postfix))
        if avg_acc >= best_acc:
            torch.save(model.state_dict(), best_model_path)
            best_acc = avg_acc
            best_state = model.state_dict()

    torch.save(model.state_dict(), last_model_path)

    for name in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
        save_list_to_file(os.path.join(opt.experiment_root,
                                       name + '.txt'), locals()[name])

    return best_state, best_acc, train_loss, train_acc, val_loss, val_acc


def test(opt, test_dataloader, model):
    """
    Test the model trained with the prototypical learning algorithm
    """
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    avg_acc = list()
    for epoch in range(10):
        test_iter = iter(test_dataloader)
        for batch in test_iter:
            x, y = batch
            x, y = x.to(device), y.to(device)

            mask = get_mask()
            x[:, mask, :, :] = -1  # mask the channels
            x = x.reshape((x.shape[0], 1, -1, x.shape[3]))

            model_output = model(x)
            _, acc = loss_fn(model_output, target=y,
                             n_support=opt.num_support_val)
            avg_acc.append(acc.item())
    avg_acc = np.mean(avg_acc)
    print('Test Acc: {}'.format(avg_acc))

    return avg_acc


# def eval(opt):
#     '''
#     Initialize everything and train
#     '''
#     options = get_parser().parse_args()
#
#     if torch.cuda.is_available() and not options.cuda:
#         print("WARNING: You have a CUDA device, so you should probably run with --cuda")
#
#     init_seed(options)
#     test_dataloader = init_dataset(options)[-1]
#     model = init_protonet(options)
#     model_path = os.path.join(opt.experiment_root, 'best_model.pth')
#     model.load_state_dict(torch.load(model_path))
#
#     test(opt=options,
#          test_dataloader=test_dataloader,
#          model=model)


def main():
    """
    Initialize everything and train
    """
    options = get_parser().parse_args()
    if not os.path.exists(options.experiment_root):
        os.makedirs(options.experiment_root)

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_seed(options)

    tr_dataloader, val_dataloader, test_dataloader = init_dataloader(options)

    model = init_vit(options)
    optim = init_optim(options, model)
    lr_scheduler = init_lr_scheduler(options, optim)
    res = train(opt=options,
                tr_dataloader=tr_dataloader,
                val_dataloader=val_dataloader,
                model=model,
                optim=optim,
                lr_scheduler=lr_scheduler)
    best_state, best_acc, train_loss, train_acc, val_loss, val_acc = res
    print('Testing with last model..')
    test(opt=options,
         test_dataloader=test_dataloader,
         model=model)

    model.load_state_dict(best_state)
    print('Testing with best model..')
    test(opt=options,
         test_dataloader=test_dataloader,
         model=model)


if __name__ == '__main__':
    main()
