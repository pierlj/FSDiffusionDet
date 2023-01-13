from detectron2.engine import HookBase

class SupportExtractionHook(HookBase):
    def __init__(self, cfg, hook_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cfg = cfg
        self.hook_fn = hook_fn

    def before_step(self):
        next_iter = self.trainer.iter
        if next_iter % self.cfg.FEWSHOT.ATTENTION.EXTRACT_EVERY == 0:
            self.hook_fn()