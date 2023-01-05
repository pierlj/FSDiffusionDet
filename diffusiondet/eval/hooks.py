from detectron2.engine import EvalHook

class FSValidationHook(EvalHook):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def after_train(self):
        del self._func

class FSTestHook(EvalHook):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def after_step(self):
        pass



