import sys
sys.path.append('/home/zhouzhiheng/STGCN')
import os
import time
import datetime
import torch
import torch.nn as nn
import numpy as np
import wandb
import models_utils
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Subset
from constants import FREQUENCY
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, cohen_kappa_score

class CrossValidation:
    def __init__(self, args):
        self.args = args

    def cross_validation_kfold(self, args, DIR, SubDataLoader, Model, folds=10):
        all_set = np.load(DIR + 'all_set.npy')  # [subjects*trials, sfreq*time]
        all_label = np.load(DIR + 'all_label.npy')  # [subjects*trials, 4]
        subdataloader = SubDataLoader(args, all_set, all_label, is_train=False)
        # k 折交叉验证
        kf = KFold(n_splits=folds, shuffle=True)
        loss_list = {}
        acc_list = {}
        for f, fold in enumerate(kf.split(subdataloader)):
            train_idx, valid_idx = fold
            load_time = time.time()
            valid_subset = Subset(subdataloader, valid_idx)
            train_subset = Subset(subdataloader, train_idx)
            TrainLoader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, drop_last=True)
            TestLoader = DataLoader(valid_subset, batch_size=args.batch_size, shuffle=True, drop_last=True)
            print('data load success! | load time = %.4f HR' % ((time.time() - load_time) / 3600))
            
            now = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")
            model = Model(args).to(args.device)
            print(model)
            modelname = Model.__name__ + "_" + str(args.subjects) + '-subjects' + '_' + now
            print(f"modelname: {modelname}")
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            print(f"第 {f} 次交叉验证---------------------------------------")
            maxloss, maxacc = self.run_model(args, TrainLoader, TestLoader, optimizer, model, modelname)
            loss_list[str(f)] = maxloss
            acc_list[str(f)] = maxacc
        print("---------------------------------------------------------------")
        avg_loss = sum(loss_list.values()) / len(loss_list.values())
        avg_acc = sum(acc_list.values()) / len(acc_list.values())
        loss_list['avg_loss'] = avg_loss
        acc_list['avg_acc'] = avg_acc
        print(f'loss_list: {loss_list}')
        print(f'acc_list: {acc_list}')
        try:
            os.makedirs(f"./{modelname}/")
        except OSError:
            pass
        np.save(f'./{modelname}/loss_list.npy', loss_list)
        np.save(f'./{modelname}/acc_list.npy', acc_list)
        with open(f'./{modelname}/args.txt', 'w') as f:
            f.write(f'{modelname}\n')
            for key, value in vars(args).items():
                f.write(f'{key}: {value}\n')

    def adjust_learning_rate(self, optimizer, epoch):
        if epoch <= 300:
            lr = self.args.learning_rate
        else:
            lr = 1e-5
        print(f'learning_rate:{lr}')
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def run_model(self, args, TrainLoader, TestLoader, optimizer, model, modelname):
            if args.wandb: 
                wandb.init(
                    project = 'GCNs_Net',
                    config = args,
                    name = f'{modelname}',
                )
            # 初始化loss和acc列表
            max_acc = 0
            max_loss = 1e5
            max_epo = 0
            start_full_time = time.time()
            hook_handles = []
            train_loss_epoch = []
            train_acc_epoch = []
            test_loss_epoch = []
            test_acc_epoch = []

            for epoch in range(args.epochs):
                epoch_start_time = time.time()
                self.adjust_learning_rate(optimizer, epoch)
                # training
                train_loss_avg_epoch, train_acc_avg_epoch, eval = self.run_oneepoch(args, TrainLoader, model, optimizer, epoch, is_train=True)
                train_loss_epoch.append(train_loss_avg_epoch)
                train_acc_epoch.append(train_acc_avg_epoch)
                
                # test
                test_loss_avg_epoch, test_acc_avg_epoch, eval = self.run_oneepoch(args, TestLoader, model, optimizer, epoch, is_train=False)
                test_loss_epoch.append(test_loss_avg_epoch)
                test_acc_epoch.append(test_acc_avg_epoch)

                if test_acc_avg_epoch > max_acc:
                    max_acc = test_acc_avg_epoch
                    max_epo = epoch
                if test_loss_avg_epoch < max_loss:
                    max_loss = test_loss_avg_epoch
                print('%s: MAX %d epoch total test acc = %.5f' % (modelname, max_epo, max_acc))
                print('%d epoch time = %.5f HR' %
                    (epoch, (time.time() - epoch_start_time) / 3600))

                none_grads = {}
                grads = {}
                values = model.state_dict()
                for name, parms in model.named_parameters():
                    if parms.requires_grad:
                        grads[f'{name}.grads'] = parms.grad
                        if parms.grad is None:
                            none_grads[name] = parms.grad
                grads['epoch'] = epoch
                print(f'none grads: {none_grads}')
                # print(f"conv1.weight.grads: {grads['conv1.weight.grads']}")
                if args.wandb:
                    wandb.log(grads)
                    wandb.log({'epoch': epoch, 'train_loss': train_loss_avg_epoch, 
                            'train_acc': train_acc_avg_epoch, 'test_loss': test_loss_avg_epoch, 'test_acc': test_acc_avg_epoch})

                if epoch % 20 == 0:
                    try:
                        os.makedirs(f'./SAVE/{modelname}/{epoch}')
                    except OSError:
                        pass
                    torch.save(model.state_dict(), f'./SAVE/{modelname}//{epoch}/model_{epoch}.pth')
                    np.save(f'./SAVE/{modelname}/{modelname}_train_loss_epoch{epoch}.npy', train_loss_epoch)
                    np.save(f'./SAVE/{modelname}/{modelname}_train_acc_epoch{epoch}.npy', train_acc_epoch)
                    np.save(f'./SAVE/{modelname}/{modelname}_test_loss_epoch{epoch}.npy', test_loss_epoch)
                    np.save(f'./SAVE/{modelname}/{modelname}_test_acc_epoch{epoch}.npy', test_acc_epoch)

            print('full time = %.2f HR' % ((time.time() - start_full_time) / 3600))
            print('max_epo: %d' % max_epo)
            print('max_acc: %.3f' % max_acc)
            if args.close_wandb:
                os.environ["WANDB_DISABLIED"] = "true"
                wandb.finish()
            return max_loss.item(), max_acc

    def run_oneepoch(self, args, DataLoader, model, optimizer, epoch, is_train=True):
        total_loss = []
        total_acc = []
        for batch_idx, data in enumerate(DataLoader):
            if is_train:
                loss, eval = self.train(optimizer, data, model, args.device, max_grad_norm=args.max_grad_norm, data_augment=args.data_augment)
            else:
                loss, eval = self.test(data, model, args.device)
            total_acc.append(eval['accuracy'])
            total_loss.append(loss)
        loss_avg_epoch = sum(total_loss) / len(total_loss)
        acc_avg_epoch = sum(total_acc) / len(total_acc)
        if is_train:
            print('epoch %d total train loss = %.3f | total train acc = %.3f' % (epoch, loss_avg_epoch, acc_avg_epoch))
        else:
            print('epoch %d total test loss = %.3f | total test acc = %.3f' % (epoch, loss_avg_epoch, acc_avg_epoch))
        return loss_avg_epoch, acc_avg_epoch, eval

    def train(self, optimizer, data, model, device, max_grad_norm, data_augment=False):
        model.train()
        data_slice, label_train, L, _ = data
        # TODO:: data augment
        if data_augment and len(data_slice.shape) >= 4:
            # 应用data_slice augment
            data_slice, label_train = models_utils.interaug_slice(data_slice, label_train)
        elif data_augment and len(data_slice.shape) < 4:
            # 应用data_window augment
            data_slice, label_train = models_utils.interaug(data_slice, label_train)
        x_slice = torch.as_tensor(np.array(data_slice), dtype=torch.float32).to(device)
        target = torch.as_tensor(np.array(label_train), dtype=torch.float32).to(device)
        for i in range(len(L)):
            L[i] = L[i].to(device)
        
        optimizer.zero_grad()
        output = model(x_slice, L)
        eval = self.evaluation(y_true=torch.argmax(target, dim=1), y_pred=torch.argmax(output, dim=1))
        # acc = (torch.argmax(output, dim=1) == torch.argmax(target, dim=1)).float().sum()
        loss = self.loss_fn(output, target)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        # print(f"learning_rate: {optimizer.state_dict()['param_groups'][0]['lr']}")
        
        return loss.detach().cpu(), eval

    def test(self, data, model, device):
        model.eval()

        data_slice, label_test, L, _ = data
        x_slice = torch.as_tensor(np.array(data_slice), dtype=torch.float32).to(device)
        target = torch.as_tensor(np.array(label_test), dtype=torch.float32).to(device)
        for i in range(len(L)):
            L[i] = L[i].to(device)

        with torch.no_grad():
            pred = model(x_slice, L)
            eval = self.evaluation(y_true=torch.argmax(target, dim=1), y_pred=torch.argmax(pred, dim=1))
            # acc = (torch.argmax(pred, dim=1) == torch.argmax(target, dim=1)).float().sum()
            loss = self.loss_fn(pred, target)

        return loss.detach().cpu(), eval

    def loss_fn(self, output, target):
        b = 0.5
        criterion = CrossEntropyLoss()
        loss = criterion(output, target)
        flood = (loss - b).abs()+b
        return loss
    
    def evaluation(self, y_true, y_pred):
        y_true = y_true.detach().cpu()
        y_pred = y_pred.detach().cpu()

        cr = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)
        kp = cohen_kappa_score(y_true, y_pred)
        s = cr['weighted avg']
        eval_dir = {}
        eval_dir['accuracy'] = cr['accuracy']
        eval_dir['precision'] = s['precision']
        eval_dir['recall'] = s['recall']
        eval_dir['f1-score'] = s['f1-score']
        eval_dir['kappa'] = kp
        return eval_dir