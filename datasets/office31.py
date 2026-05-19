import os.path as osp

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden


@DATASET_REGISTRY.register()
class Office31Flex(DatasetBase):
    """Office-31 with flexible directory layout support for DA."""

    dataset_dir = "office31"
    domains = ["amazon", "dslr", "webcam"]

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )

        train_x = self._read_data(cfg.DATASET.SOURCE_DOMAINS)
        train_u = self._read_data(cfg.DATASET.TARGET_DOMAINS)
        test = self._read_data(cfg.DATASET.TARGET_DOMAINS)

        super().__init__(train_x=train_x, train_u=train_u, test=test)

    def _read_data(self, input_domains):
        items = []

        for domain_idx, domain_name in enumerate(input_domains):
            domain_dir = osp.join(self.dataset_dir, domain_name)
            image_root = osp.join(domain_dir, "images")
            class_root = image_root if osp.isdir(image_root) else domain_dir

            if not osp.isdir(class_root):
                raise FileNotFoundError(
                    "Office-31 domain folder not found: "
                    f"{class_root}. Expected either '<root>/office31/{domain_name}/images/*' "
                    "or '<root>/office31/{domain_name}/*'."
                )

            class_names = listdir_nohidden(class_root)
            class_names.sort()

            for label, class_name in enumerate(class_names):
                class_dir = osp.join(class_root, class_name)
                if not osp.isdir(class_dir):
                    continue

                for image_name in listdir_nohidden(class_dir):
                    impath = osp.join(class_dir, image_name)
                    if osp.isdir(impath):
                        continue

                    items.append(
                        Datum(
                            impath=impath,
                            label=label,
                            domain=domain_idx,
                            classname=class_name,
                        )
                    )

        return items
