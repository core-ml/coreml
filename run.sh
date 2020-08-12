# python competitions/2020/melanoma-classification/evaluation/eval.py -v competitions/2020/melanoma-classification/configs/efficientnet/best-ns-colorjitter-b3.yml -m test --wandb -b --n-tta 15

# python competitions/2020/melanoma-classification/evaluation/eval.py -v competitions/2020/melanoma-classification/configs/efficientnet-cv/best-ns-colorjitter-b3-fold2.yml -m test --wandb -b --n-tta 15

# python competitions/2020/melanoma-classification/evaluation/eval.py -v competitions/2020/melanoma-classification/configs/efficientnet-cv/best-ns-colorjitter-b3-fold3.yml -m test --wandb -b --n-tta 15

# python competitions/2020/melanoma-classification/evaluation/eval.py -v competitions/2020/melanoma-classification/configs/efficientnet-cv/best-ns-colorjitter-b3-fold4.yml -m test --wandb -b --n-tta 15

python competitions/2020/melanoma-classification/evaluation/eval.py -v competitions/2020/melanoma-classification/configs/effb5/best-1cycle-wd4e-1-384/fold3.yml -m val --wandb -b --n-tta 15
