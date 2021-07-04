
""" CNN for architecture search """
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.modules.search_cells import SearchCell
import src.models.modules.genotypes as gt
import pytorch_lightning as pl
import wandb
from typing import Any, List
from torchmetrics.classification.accuracy import Accuracy
from hydra.utils import instantiate
import torchvision
import copy
from src.models.modules.head import mark_classifier
from src.models.modules.architect import Architect



class SearchCNN(nn.Module):
    """ Search CNN model """
    def __init__(self, C_in, C, n_classes, n_layers, criterion, n_nodes=4, stem_multiplier=3):
        """
        Args:
            C_in: # of input channels
            C: # of starting model channels
            n_classes: # of classes
            n_layers: # of layers
            n_nodes: # of intermediate nodes in Cell
            stem_multiplier
        """
        super().__init__()
        self.C_in = C_in
        self.C = C
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.criterion = criterion

        C_cur = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(C_in, C_cur, 3, 1, 1, bias=False),
            nn.BatchNorm2d(C_cur)
        )

        # for the first cell, stem is used for both s0 and s1
        # [!] C_pp and C_p is output channel size, but C_cur is input channel size.
        C_pp, C_p, C_cur = C_cur, C_cur, C

        self.cells = nn.ModuleList()
        reduction_p = False
        for i in range(n_layers):
            # Reduce featuremap size and double channels in 1/3 and 2/3 layer.
            if i in [n_layers//3, 2*n_layers//3]:
                C_cur *= 2
                reduction = True
            else:
                reduction = False

            cell = SearchCell(n_nodes, C_pp, C_p, C_cur, reduction_p, reduction)
            reduction_p = reduction
            self.cells.append(cell)
            C_cur_out = C_cur * n_nodes
            C_pp, C_p = C_p, C_cur_out

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(C_p, n_classes)

        # initialize architect parameters: alphas
        self._init_alphas()

    def _init_alphas(self):
        """
        initialize architect parameters: alphas
        """
        n_ops = len(gt.PRIMITIVES)

        self.alpha_normal = nn.ParameterList()
        self.alpha_reduce = nn.ParameterList()

        for i in range(self.n_nodes):
            self.alpha_normal.append(nn.Parameter(1e-3*torch.randn(i+2, n_ops)))
            self.alpha_reduce.append(nn.Parameter(1e-3*torch.randn(i+2, n_ops)))

    def forward(self, x):
        s0 = s1 = self.stem(x)

        weights_normal = [F.softmax(alpha, dim=-1) for alpha in self.alpha_normal]
        weights_reduce = [F.softmax(alpha, dim=-1) for alpha in self.alpha_reduce]

        for cell in self.cells:
            weights = weights_reduce if cell.reduction else weights_normal
            s0, s1 = s1, cell(s0, s1, weights)

        out = self.gap(s1)
        out = out.view(out.size(0), -1) # flatten
        logits = self.linear(out)
        return logits

    def loss(self, X, y):
        logits = self(X)
        return self.criterion(logits, y)

    def print_alphas(self):
        # TODO: change it to return str
        print("####### ALPHA #######")
        print("# Alpha - normal")
        for alpha in self.alpha_normal:
            print(F.softmax(alpha, dim=-1))

        print("\n# Alpha - reduce")
        for alpha in self.alpha_reduce:
            print(F.softmax(alpha, dim=-1))
        print("#####################")

    def genotype(self):
        gene_normal = gt.parse(self.alpha_normal, k=2)
        gene_reduce = gt.parse(self.alpha_reduce, k=2)
        concat = range(2, 2+self.n_nodes) # concat all intermediate nodes

        return gt.Genotype(normal=gene_normal, normal_concat=concat,
                           reduce=gene_reduce, reduce_concat=concat)

    def weights(self):
        for k, v in self.named_parameters():
            if 'alpha' not in k:
                yield v

    def named_weights(self):
        for k, v in self.named_parameters():
            if 'alpha' not in k:
                yield k, v

    def alphas(self):
        for k, v in self.named_parameters():
            if 'alpha' in k:
                yield v

    def named_alphas(self):
        for k, v in self.named_parameters():
            if 'alpha'in k:
                yield k, v


class DARTS(pl.LightningModule):
    def __init__(self, C_in, C, n_classes, n_layers=8, n_nodes=4, stem_multiplier=3, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        print(self.hparams)
        self.automatic_optimization = False # important
        criterion = nn.CrossEntropyLoss()
        self.net = SearchCNN(
            self.hparams.C_in, self.hparams.C, self.hparams.n_classes, self.hparams.n_layers, criterion,
            n_nodes=self.hparams.n_nodes, stem_multiplier=self.hparams.stem_multiplier
        )
        self.v_net = copy.deepcopy(self.net)
        self.architect = Architect(
            self.net, self.v_net,
            self.hparams.optim.items()[0][1].momentum,
            self.hparams.optim.items()[0][1].weight_decay,
        )

        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()
        self.train_accuracy5 = Accuracy(top_k=5)
        self.val_accuracy5 = Accuracy(top_k=5)
        self.test_accuracy5 = Accuracy(top_k=5)

        self.table = wandb.Table(columns=["GenoType", "Current Score", "Epoch", "Step"])


    def on_train_epoch_start(self) -> None:
        # do scheduler step
        sch = self.lr_schedulers()
        sch.step()
        self.lr = sch.get_lr()[0]
        # log alphas

    def training_step(self, batch: Any, batch_idx: int, optimizer_idx: int):

        weights_optim, alpha_optim = self.optimizers()

        train_x, train_y = batch['train']
        val_x, val_y = batch['val']

        # phase 2. architect step (alpha)
        alpha_optim.zero_grad()
        self.architect.unrolled_backward(train_x, train_y, val_x, val_y, self.lr, weights_optim)
        alpha_optim.step()

        # phase 1. child network step (w)
        weights_optim.zero_grad()
        logits = self.net(train_x)
        loss = self.net.criterion(logits, train_y)
        self.manual_backward(loss)
        # gradient clipping
        nn.utils.clip_grad_norm_(self.net.weights(), self.hparams.w_grad_clip)
        weights_optim.step()

        logits = F.softmax(logits)
        acc = self.train_accuracy(logits, train_y)
        acc5 = self.train_accuracy5(logits, train_y)

        self.log(f"train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log(f"train/acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"train/acc5", acc5, on_step=True, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        x, y = batch
        logits = self.net(x)
        loss = self.net.criterion(logits, y)

        logits = F.softmax(logits)
        # log val metrics
        acc = self.val_accuracy(logits, y)
        acc5 = self.val_accuracy5(logits, y)
        self.log(f"val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"val/acc5", acc5, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        x, y = batch
        logits = self.net(x)
        loss = self.net.criterion(logits, y)

        logits = F.softmax(logits)
        # log test metrics
        acc = self.test_accuracy(logits, y)
        acc5 = self.test_accuracy5(logits, y)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)
        self.log("test/acc5", acc5, on_step=False, on_epoch=True)

        return {"loss": loss}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self) -> None:
        #         self.net.print_alphas()
        if self.trainer.sanity_checking:
            return
        current_score = self.trainer.checkpoint_callback.current_score.item() if self.trainer.checkpoint_callback.current_score is not None else "Not evaluated"
        self.table.add_data(
            str(self.net.genotype()),
            str(current_score),
            str(self.current_epoch), str(self.global_step),
        )
        expr = self.logger.experiment[0]
        expr.summary["best_score"] = self.trainer.checkpoint_callback.best_model_score
        # log genotype
        # log the best val acc yet

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        weights_optim_name, weights_optim_config = self.hparams.optim.items()[0]
        weights_optim = instantiate(
            weights_optim_config, params=self.net.weights(), _convert_="partial"
        )
        # Architecture optimizer
        alpha_optim_name, alpha_optim_config = self.hparams.optim.items()[1]
        alpha_optim = instantiate(
            alpha_optim_config, params=self.net.alphas(), _convert_="partial"
        )
        weights_optim_sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            weights_optim, self.trainer.max_epochs, eta_min=0.001
        )
        return [weights_optim, alpha_optim], [weights_optim_sch]

    def on_fit_end(self) -> None:
        expr = self.logger.experiment[0]
        expr.log({"Table": self.table})