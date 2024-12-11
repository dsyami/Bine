import torch
import pickle
import numpy as np
import importlib
import argparse
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from sklearn.metrics import classification_report, cohen_kappa_score


class ModelController:
    def __init__(self, network, adj):
        self.dir = 'D:/dsHit/thesis/MyProject/Saved_Files/myGConvLSTM_20-subjects_2024-12-09-13-46/'
        self.trials = 84

        args_dir = self.dir + "args.pkl"
        with open(args_dir, "rb") as f:
            args_dict = pickle.load(f)
        self.args = argparse.Namespace(**args_dict)
        # arg_name 对应的 class_name 和 module_name
        NetWorkDict = {
            'GConvLSTM' : 'myGConvLSTM',
        }
        NetWorkModuleDict = {
            'GConvLSTM' : 'Models.Network.GConvLSTM',
        }
        DataLoaderDict = {
            'GConvLSTM' : 'myDataLoaderGConvLSTM',
        }
        DataLoaderModuleDict = {
            'GConvLSTM' : 'Models.Network.lib_for_GCN.GCN_dataloader'
        }

        self.network = self.get_class_by_name(NetWorkModuleDict[network], NetWorkDict[network])
        self.myDataLoader = self.get_class_by_name(DataLoaderModuleDict[network], DataLoaderDict[network])

    
    def get_class_by_name(self, module_name, class_name):
        try:
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except ModuleNotFoundError:
            print(f"Module '{module_name}' not found.")
        except AttributeError:
            print(f"Class '{class_name}' not found in module '{module_name}'.")
            return None
    

    def test(self, dataLoader, model):
        for data in dataLoader:
            data_slice, label_test, L, _ = data
            x_slice = torch.as_tensor(np.array(data_slice), dtype=torch.float32).to(self.args.device)
            target = torch.as_tensor(np.array(label_test), dtype=torch.float32).to(self.args.device)
            for i in range(len(L)):
                L[i] = L[i].to(self.args.device)

            with torch.no_grad():
                pred = model(x_slice, L)
                y_true = torch.argmax(target, dim=1)
                y_pred = torch.argmax(pred, dim=1)
                eval = self.evaluation(y_true=y_true, y_pred=y_pred)
                # acc = (torch.argmax(pred, dim=1) == torch.argmax(target, dim=1)).float().sum()
                loss = self.loss_fn(pred, target)

        return loss.detach().cpu(), eval, y_pred.detach().cpu().numpy(), y_true.detach().cpu().numpy()


    def loss_fn(self, output, target):
            b = 0.5
            criterion = CrossEntropyLoss()
            loss = criterion(output, target)
            flood = (loss - b).abs()+b
            return loss


    def evaluation(self, y_true, y_pred):
        y_true = y_true.detach().cpu()
        y_pred = y_pred.detach().cpu()

        cr = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True, )
        kp = cohen_kappa_score(y_true, y_pred)
        s = cr['weighted avg']
        eval_dir = {}
        eval_dir['accuracy'] = cr['accuracy']
        eval_dir['precision'] = s['precision']
        eval_dir['recall'] = s['recall']
        eval_dir['f1_score'] = s['f1-score']
        eval_dir['kappa'] = kp
        return eval_dir


    def predict(self, set, label):
        model = self.network(self.args).to(self.args.device)
        model_dict_dir = 'D:/dsHit/thesis/MyProject/Saved_Files/myGConvLSTM_20-subjects_2024-12-09-13-46/0fold/20/model_20.pth'
        model.load_state_dict(torch.load(model_dict_dir, weights_only=True))
        model.eval()
        dataLoader = DataLoader(self.myDataLoader(self.args, set, label, is_train=False), batch_size=self.trials)
        loss, eval, pred, target = self.test(dataLoader, model)
        return loss, eval, pred, target