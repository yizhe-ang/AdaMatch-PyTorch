from dassl.data import DataManager
from dassl.engine import TRAINER_REGISTRY, TrainerXU
from dassl.data.transforms import build_transform


@TRAINER_REGISTRY.register()
class AdaMatch(TrainerXU):
    """AdaMatch: A Unified Approach to Semi-Supervised Learning
    and Domain Adaptation

    https://arxiv.org/abs/2106.04732
    """

    def __init__(self, cfg):
        super().__init__(cfg)

    def build_data_loader(self):
        """Create essential data-related attributes.

        What must be done in the re-implementation
        of this method:
        1) initialize data manager
        2) assign as attributes the data loaders
        3) assign as attribute the number of classes
        """
        cfg = self.cfg

        # Initialize training transformations
        # Initialize default / weak transforms
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        # Initialize strong transforms
        choices = cfg.TRAINER.ADAMATCH.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)

        custom_tfm_train += [tfm_train_strong]

        self.dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)

        self.train_loader_x = self.dm.train_loader_x
        self.train_loader_u = self.dm.train_loader_u
        self.val_loader = self.dm.val_loader
        self.test_loader = self.dm.test_loader

        self.num_classes = self.dm.num_classes
