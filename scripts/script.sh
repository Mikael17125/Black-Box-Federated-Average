CUDA_VISIBLE_DEVICES=1 python main.py --non_iid 0 --frac 0.1 --log_dir /data/nias/black-box-fed-avg/logs/iid_frac_0.1 --save_dir /data/nias/black-box-fed-avg/results/iid_frac_0.1/

CUDA_VISIBLE_DEVICES=2 python main.py --non_iid 0 --frac 0.2 --log_dir /data/nias/black-box-fed-avg/logs/full_shot_iid_frac_0.2 --save_dir /data/nias/black-box-fed-avg/results/full_shot_iid_frac_0.2/

CUDA_VISIBLE_DEVICES=0 python train.py --non_iid 0 --frac 0.1 --log_dir /home/mikael/Code/black-box-fed-avg/logs/analyze_few_0.1 --save_dir /home/mikael/Code/black-box-fed-avg/results/analyze_few_0.1
CUDA_VISIBLE_DEVICES=1 python train.py --non_iid 0 --frac 0.2 --log_dir /home/mikael/Code/black-box-fed-avg/logs/analyze_few_0.2 --save_dir /home/mikael/Code/black-box-fed-avg/results/analyze_few_0.2
CUDA_VISIBLE_DEVICES=1 python train.py --non_iid 0 --frac 0.5 --log_dir /home/mikael/Code/black-box-fed-avg/logs/analyze_few_0.5 --save_dir /home/mikael/Code/black-box-fed-avg/results/analyze_few_0.5
CUDA_VISIBLE_DEVICES=1 python train.py --non_iid 0 --frac 1.0 --log_dir /home/mikael/Code/black-box-fed-avg/logs/analyze_few_1.0 --save_dir /home/mikael/Code/black-box-fed-avg/results/analyze_few_1.0

CUDA_VISIBLE_DEVICES=0 python inference.py --non_iid 0
