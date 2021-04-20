import os
import time
import random
import argparse
import logging
import shutil
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.resnet import resnet18

logger = logging.getLogger(__name__)


# Perturbation Network
class G(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0),
                nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


def train(train_loader, net, perturb, criterion, optimizer):
    net.train()
    train_loss = 0
    train_acc = 0
    gamma = 0.01
    softmax = nn.Softmax(dim=1)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        # authorized
        output = net(perturb(data) + data)
        # plain
        plain_output = net(data)

        # dissmilarity
        raw_loss = torch.sum(softmax(plain_output) * F.one_hot(target))
        # classification
        ce_loss = criterion(output, target)
        # similarity
        distance = gamma * torch.norm(perturb(data), 2)

        loss = ce_loss + raw_loss + distance

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # for name, param in perturb.named_parameters():
        #     print(name, param.grad)

        train_loss += loss.item() * target.size(0)
        train_acc += (output.max(1)[1] == target).sum().item()

    train_loss /= len(train_loader.dataset)
    train_acc /= len(train_loader.dataset)
    return train_loss, train_acc


def test(test_loader, net, perturb=None):
    global best_acc

    net.eval()
    perturb.eval()
    test_loss = 0
    test_acc = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            if perturb:
                output = net(perturb(data) + data)
            else:
                output = net(data)
            test_loss += F.cross_entropy(output, target,
                                         reduction="sum").item()
            test_acc += (output.max(1)[1] == target).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc /= len(test_loader.dataset)
    logger.info("== Test loss: {:.4f}, Test acc: {:.4f}"
                .format(test_loss, test_acc))

    is_best = test_acc > best_acc

    save_model(net.state_dict(), is_best, "protected")
    torch.save(perturb.state_dict(), "G.pth.tar")

    if is_best:
        best_acc = test_acc

    return test_loss, test_acc


def load_cifar10(worker_init_fn):
    batch_size = 128
    data_path = '~/.fastai/data'
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2471, 0.2435, 0.2616]  # before converting to tensors

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    num_workers = 2

    test_dataset = datasets.CIFAR10(
        data_path, train=False, transform=test_transform, download=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
    )

    train_dataset = datasets.CIFAR10(
        data_path, train=True, transform=train_transform, download=True
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
    )
    return train_loader, test_loader


def save_model(state, is_best, filename):
    torch.save(state, filename + ".pth.tar")
    if is_best:
        shutil.copyfile(filename + ".pth.tar", filename + "_best.pth.tar")


def load_model(path, model):
    model.load_state_dict(torch.load(path))


def main():
    global best_acc

    epochs = 200
    eval_freq = 50
    momentum = 0.9
    weight_decay = 0.0005

    logfile = "log.txt"
    if os.path.exists(logfile):
        os.remove(logfile)

    logging.basicConfig(
        format="[%(asctime)s] - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.FileHandler(logfile), logging.StreamHandler()],
    )

    # fix all seeds for reproducibility
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    def _init_fn(worker_id):
        np.random.seed(int(seed))

    train_loader, test_loader = load_cifar10(worker_init_fn=_init_fn)

    net = resnet18().cuda()
    perturb = G().cuda()

    optimizer = torch.optim.SGD(
        list(net.parameters()) + list(perturb.parameters()),
        lr=1e-1,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    criterion = nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40,
                                                gamma=0.1)

    logger.info("Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc")
    best_acc = 0.0
    # loss_history = []
    # val_loss_history = []
    # acc_history = []
    # val_acc_history = []

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        start = time.time()

        train_loss, train_acc = train(
            train_loader, net, perturb, criterion, optimizer
        )
        scheduler.step()
        end = time.time()
        lr = scheduler.get_lr()[0]
        logger.info(
            "%d \t %.1f \t \t %.4f \t %.4f \t %.4f",
            epoch,
            end - start,
            lr,
            train_loss,
            train_acc,
        )

        if epoch == 1 or epoch % eval_freq == 0 or epoch == epochs:
            test_loss, test_acc = test(test_loader, net, perturb)
            # val_loss_history.append(test_loss)
            # val_acc_history.append(test_acc)
    end_time = time.time()
    logger.info("== Training Finished. best_test_acc: {:.4f} =="
                .format(best_acc))
    logger.info(
        "== Total training time: {:.4f} minutes =="
        .format((end_time - start_time) / 60)
    )

    plain_loss, plain_acc = test(test_loader, net)
    logger.info("== plain_test_acc: {:.4f} =="
                .format(plain_acc))
    # history = {
    #         'loss': loss_history,
    #         'val_loss': val_loss_history,
    #         'acc': acc_history,
    #         'val_acc': val_acc_history
    #     }

    # torch.save({'history': history}, config.work_path + "/history.pth")


if __name__ == "__main__":
    main()
