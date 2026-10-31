out_dir = "out-tinystories-v2"
dataset = "tinystories_v2"
init_from = "scratch"

batch_size = 8
gradient_accumulation_steps = 4
block_size = 256

n_layer = 6
n_head = 6
n_embd = 384

dropout = 0.0
bias = False

learning_rate = 6e-4
max_iters = 2500
warmup_iters = 100
lr_decay_iters = 2500
min_lr = 6e-5

eval_interval = 200
eval_iters = 20
log_interval = 10

always_save_checkpoint = True
wandb_log = False
compile = False
device = "cuda"
