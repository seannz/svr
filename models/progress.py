import sys
from pytorch_lightning.callbacks import TQDMProgressBar

class ProgressBar(TQDMProgressBar):
    def get_metrics(self, trainer, model):
        # don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        items.pop("loss", None)
        
        return items

    def on_validation_end(self, *args, **kwargs): 
        super().on_validation_end(*args, **kwargs)
        print('', file=sys.stderr)
