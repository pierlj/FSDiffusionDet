import logging
import torch
import weakref
import time
import os

from collections import OrderedDict


import detectron2.utils.comm as comm
from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_train_loader
from detectron2.engine import DefaultTrainer, create_ddp_model, \
    AMPTrainer, SimpleTrainer, hooks, TrainerBase
from detectron2.evaluation import DatasetEvaluator, print_csv_format
from detectron2.modeling import build_model
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter
from detectron2.utils.file_io import PathManager

from diffusiondet import DiffusionDetDatasetMapper
from diffusiondet.util.model_ema import may_build_model_ema, may_get_ema_checkpointer, EMAHook, \
    apply_model_ema_and_restore

from .trainer import DiffusionTrainer
from ..data import DiffusionDetDatasetMapper, ClassMapper, ClassSampler, FilteredDataLoader
from ..data.registration import get_datasets
from ..data.task_sampling import TaskSampler
from ..eval.fs_evaluator import FSEvaluator
from ..eval.hooks import FSValidationHook, FSTestHook


class FineTuningTrainer(DiffusionTrainer):
    def __init__(self, cfg):
        super(DefaultTrainer, self).__init__()
        # self.logger = logging.getLogger('base.training')    
        # if not self.logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
        self.logger = logging.getLogger(__name__) 
        setup_logger(name='diffusiondet', abbrev_name='diffdet')
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        model_size = pytorch_total_params * 4 / 1024 / 1024
        self.logger.info('Number of total parameters: {} \nModel size: {} Mb'.format(pytorch_total_params, model_size))
        optimizer = self.build_optimizer(cfg, model)

        self.build_dataset(cfg)
        selected_classes = self.task_sampler.c_train

        torch.manual_seed(6565)
        data_loader = self.build_train_loader(cfg, selected_classes)

        model = create_ddp_model(model, broadcast_buffers=False)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, self.optimizer)

        ########## EMA ############
        kwargs = {
            'trainer': weakref.proxy(self),
        }
        kwargs.update(may_get_ema_checkpointer(cfg, model))
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            **kwargs,
            # trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

    
        self.register_hooks(self.build_hooks())

    def launch_training(self):
        
        if self.iter < self.cfg.SOLVER.MAX_ITER:
            self.logger.info('Start base training on classes: {}'.format(self.data_loader.draw_images_from_classes))
            self.logger.info('With annotation from classes: {}'.format(self.data_loader.keep_annotations_from_classes))
            # self.freeze_model(freeze_modules=['backbone'], backbone_freeze_at=2) #freeze early bb layers 
            self.train()

            self.prepare_for_finetuning()
        else:
            self.prepare_for_finetuning(resume_ft=True)

        self.logger.info('Start fine tuning on classes: {}'.format(self.data_loader.draw_images_from_classes))
        self.logger.info('With annotation from classes: {}'.format(self.data_loader.keep_annotations_from_classes))
        # Use great grand father class (TrainerBase) train function which is actually available 
        # in _trainer which is a SimpleTrainer(TrainerBase)
        # self._trainer.train(self.iter, self.cfg.FINETUNE.MAX_ITER)
        # TrainerBase.train(self, self.iter, self.cfg.FINETUNE.MAX_ITER)
        self.train()
        

    def run_step(self):
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()

        data = next(iter(self.data_loader))
        data_time = time.perf_counter() - start
       
        loss_dict = self.model(data)
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())


        self.optimizer.zero_grad()
        losses.backward()

        self._write_metrics(loss_dict, data_time)

        self.optimizer.step()
        return None 
    
    def prepare_for_finetuning(self, resume_ft=False):
        # self.logger = logging.getLogger('finetuning')  
        # if not self.logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
        #     setup_logger(name='finetuning')

        # freeze net
        self.freeze_model(freeze_modules=['backbone', 'head'], 
                            backbone_freeze_at=0, # all backbone
                            cls_reg_module=True) #last layer of each head only is trained
                            
        # update allowed classes
        self.data_loader = self.build_train_loader(self.cfg, self.task_sampler.c_test)

        # Update cfg params 
        self.cfg.merge_from_list(['SOLVER.MAX_ITER', self.cfg.FINETUNE.MAX_ITER])
        self.scheduler = self.build_lr_scheduler(self.cfg, self.optimizer)

        # when restarting finetuning iter should be incremented from value in checkpoint
        self.start_iter = self.iter + 1 if resume_ft else self.iter 
        self.max_iter = self.cfg.FINETUNE.MAX_ITER

        # update hooks for finetuning
        self._hooks = []
        self.register_hooks(self.build_hooks(base_training=False)) 

    def build_dataset(self, cfg):
        self.dataset, self.dataset_metadata = get_datasets(cfg.DATASETS.TRAIN, cfg)
        self.task_sampler = TaskSampler(cfg, self.dataset_metadata, torch.Generator())

    def build_train_loader(self, cfg, selected_classes):
        sampler = ClassSampler(cfg, self.dataset_metadata, selected_classes, n_query=100, is_train=True)
        mapper = ClassMapper(selected_classes, 
                            self.dataset_metadata.thing_dataset_id_to_contiguous_id,
                            cfg, 
                            is_train=True)

        dataloader = FilteredDataLoader(cfg, self.dataset, mapper, sampler, self.dataset_metadata)
        return dataloader
    
    @classmethod
    def build_test_loader(cls, trainer, cfg, selected_classes, validation=True):
        dataset, dataset_metadata = get_datasets(cfg.DATASETS.VAL 
                                                    if validation else cfg.DATASETS.TEST, 
                                                cfg)
        sampler = ClassSampler(cfg, dataset_metadata, selected_classes, n_query=100, is_train=False)
        mapper = ClassMapper(selected_classes, 
                            dataset_metadata.thing_dataset_id_to_contiguous_id,
                            cfg, 
                            is_train=True)

        dataloader = FilteredDataLoader(cfg, 
                                        dataset, 
                                        mapper, 
                                        sampler, 
                                        dataset_metadata, 
                                        is_eval=True)
        return dataloader.dataloader

    def _write_metrics(
        self,
        loss_dict,
        data_time: float,
        prefix: str = "",
    ) -> None:
        SimpleTrainer.write_metrics(loss_dict, data_time, prefix)
    
    def build_hooks(self, base_training=True):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            EMAHook(self.cfg, self.model) if cfg.MODEL_EMA.ENABLED else None,  # EMA hook
            hooks.LRScheduler(),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            prefix = 'model_base' if base_training else 'model_finetuned'
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, file_prefix=prefix))

        def test_and_save_results(validation=True):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            dataset_name = cfg.DATASETS.VAL[0] if validation else cfg.DATASETS.TEST[0]
            evaluators = [
                FSEvaluator(self.task_sampler.c_train, dataset_name, cfg, True, output_folder, name='Base classes evaluation'),
                FSEvaluator(self.task_sampler.c_test, dataset_name, cfg, True, output_folder, name='Novel classes evaluation')
            ]
            self._last_eval_results = self.eval(self, self.cfg, self.model, evaluators, validation=validation)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(FSValidationHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))
        ret.append(FSTestHook(cfg.TEST.EVAL_PERIOD, lambda: test_and_save_results(validation=False)))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    @classmethod
    def eval(cls, trainer, cfg, model, evaluators, validation=False):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.

        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        

        results = OrderedDict()
        for idx, evaluator in enumerate(evaluators):
            # eval/validate only on one dataset at a time in FS mode 
            dataset_name = cfg.DATASETS.VAL[0] if validation else cfg.DATASETS.TEST[0]
            evaluator_name = evaluator.name
            dataloader = cls.build_test_loader(trainer, cfg, evaluator.selected_classes, validation=validation)
            results_i = evaluator.inference_on_dataset(model, dataloader, validation=validation)
            results[evaluator_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results
    
    def freeze_model(self, freeze_modules=['backbone'], backbone_freeze_at=0, cls_reg_module=False, time_mlp=True):
        '''
        Freeze model parameters for various modules
        When backbone_freeze == 0, freeze all backbone parameters
        Otherwise freeze up to res_#backbone_freeze_at layer.

        '''
        if 'backbone' in freeze_modules:
            if backbone_freeze_at > 0:
                self.model.backbone.bottom_up.freeze(backbone_freeze_at)
            else:
                for param in self.model.backbone.parameters():
                    param.requires_grad = False
        if 'head' in freeze_modules:
            hot_modules = ['cls_module', 'reg_module', 'class_logits', 'bboxes_delta']
            for module in self.model.head.head_series:
                for name, param in module.named_parameters():
                    print(name)
                    if all([hm not in name for hm in hot_modules]):
                        param.requires_grad = False
                    elif ('cls_module' in name or 'reg_module' in name) and cls_reg_module:
                        param.requires_grad = False

            if time_mlp:        
                for param in self.model.head.time_mlp.parameters():
                    param.requires_grad = False
    



        

    
    

