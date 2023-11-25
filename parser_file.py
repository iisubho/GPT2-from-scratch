
def initialize(parser):

  parser.add_argument('--eval_iters', default = 200, type = int, help = 'How many iterations we want for the evaluation.')
  parser.add_argument('--log_interval', default = 200, type = int, help = 'After how many iterations we want to save the log file.')
  parser.add_argument('--save_every', default = 100, type = int, help = 'The frequency of saving our model.')
  parser.add_argument('--data', default = 'openwebtext', type = str, help = 'Which data we will be training on.')
  parser.add_argument('--batch_size', default = 24, type = int, help = 'The batch size we want to use')
  parser.add_argument('--model_type', default = 'gpt2', type = str, help = 'gpt2, gpt2-medium, gpt2-large, gpt2-excel')
  parser.add_argument('--dropout', default = 0.0, type = float, help = 'Its 0.0 during training')
  parser.add_argument('--bias', default = False, type = int, help = 'We do not induce any bias during training.')
  parser.add_argument('--lr', default = 6e-4, type = float, help = 'The training initial learning rate.')
  parser.add_argument('--max_iter', default = 60000, type = int, help = 'Maximum iterations for the training')
  parser.add_argument('--wd', default = 1e-1, type = float, help = 'Weight decay for the regularization.')
  parser.add_argument('--beta1', default = 0.90, type = float, help = 'beta1 for optimizer')
  parser.add_argument('--beta2', default = 0.95, type = float, help = 'beta2 for optimizer')
  parser.add_argument('--grad_clip', default = 1.0, type = float, help = 'The maximum gradient we want for our parameters.')
  parser.add_argument('--decay_lr', default = True, type = int, help = 'Learning rate decay during training')
  parser.add_argument('--warmup_iters', default = 2000, type = int, help = 'Warmup iterations')
  parser.add_argument('--lr_decay_iters', default = 600000, type = int, help = 'After what iteration the lr should decay.')
  parser.add_argument('--min_lr', default = 6e-5, type = float, help = 'Minimum training learning rate')
  parser.add_argument('--time_step', default = 1024, type = int, help = 'It is the time step or block size of data we want to sample.')
  parser.add_argument('--num_embed', default = 50304, type = int, help = '50257 for gpt2, but rounded up for efficiency.')
  parser.add_argument('--data_dir', default = '/data/lab/doppa/Subhankar/Subhankar/Codes/LLM/data/', type = str, help = 'Data path.')

  return parser

