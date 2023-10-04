########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

print('Loading...')
# from model import RWKV
from src.model_run import RWKV_RNN
import numpy as np
import os, copy, types, gc, sys
import torch
# from rwkv.utils import PIPELINE
# from src.utils import TOKENIZER

from transformers import AutoTokenizer
from torch.nn import functional as F
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
except:
    pass
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
np.set_printoptions(precision=4, suppress=True, linewidth=200)


def sample_logits(out, x, ctx_len, temperature=1.0, top_p_usual=None, top_p_newline=None):
    # out[self.UNKNOWN_CHAR] = -float('Inf')
    # lastChar = int(x[-1])

    probs = F.softmax(out, dim=-1)

    top_p = top_p_usual

    if os.environ["RWKV_RUN_DEVICE"] == "cpu":
        probs = probs.numpy()
        sorted_probs = np.sort(probs)[::-1]
        cumulative_probs = np.cumsum(sorted_probs)
        cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
        probs[probs < cutoff] = 0
        if temperature != 1.0:
            probs = probs.pow(1.0 / temperature)
        probs = probs / np.sum(probs)
        out = np.random.choice(a=len(probs), p=probs)
        return out
    else:
        sorted_probs = torch.sort(probs, descending=True)[0]
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
        cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
        probs[probs < cutoff] = 0
        if temperature != 1.0:
            probs = probs.pow(1.0 / temperature)
        out = torch.multinomial(probs, num_samples=1)[0]
        return out

CHAT_LANG = 'English'  # English Chinese


user = 'User'
bot = 'Bot'
interface = ':'


# WORD_NAME = [
#     "20B_tokenizer.json",
#     "20B_tokenizer.json",
# ]  # [vocab, vocab] for Pile model
# UNKNOWN_CHAR = None
# tokenizer = TOKENIZER(WORD_NAME, UNKNOWN_CHAR=UNKNOWN_CHAR)

tokenizer = AutoTokenizer.from_pretrained("../checkpoint/rwkv4c/")

args = types.SimpleNamespace()
args.RUN_DEVICE = "cuda"  # 'cpu' (already very fast) // 'cuda'
args.FLOAT_MODE = "fp32"  # fp32 (good for CPU) // fp16 (recommended for GPU) // bf16 (less accurate)
args.vocab_size = 25000
args.head_qk = 0
args.pre_ffn = 0
args.grad_cp = 0
args.my_pos_emb = 0

# args.MODEL_NAME = '../checkpoint/rwkv4_169m_vi_20230923/rwkv-12'
# args.MODEL_NAME = '../checkpoint/rwkv4_169m_vi_20230924_ft_chat/rwkv-4'
# args.MODEL_NAME = '../checkpoint/rwkv4_169m_vi_20230924_ft_chat/rwkv-10'
# args.MODEL_NAME = '../checkpoint/rwkv4_169m_vi_20230926/rwkv-7'
args.MODEL_NAME = '../checkpoint/rwkv4_the_thao/rwkv-29'
args.n_layer = 12
args.n_embd = 768
args.ctx_len = 1024

# args.MODEL_NAME = './out/rwkv-step_0'
# args.n_layer = 4
# args.n_embd = 256
# args.ctx_len = 1024

# args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-7b/RWKV-4-Pile-7B-20221115-8047'
# args.n_layer = 32
# args.n_embd = 4096
# args.ctx_len = 1024

# args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-3b/RWKV-4-Pile-3B-20221008-8023'
# args.n_layer = 32
# args.n_embd = 2560
# args.ctx_len = 1024

# Load Model

os.environ["RWKV_RUN_DEVICE"] = args.RUN_DEVICE
MODEL_NAME = args.MODEL_NAME

print(f'loading... {MODEL_NAME}')
model = RWKV_RNN(args)

model_tokens = []

current_state = None


########################################################################################################

def run_rnn(tokens, newline_adj=0):
    global model_tokens, current_state
    for i in range(len(tokens)):
        model_tokens += [int(tokens[i])]
        if i == len(tokens) - 1:
            out, current_state = model.forward(model_tokens, current_state)
        else:
            current_state = model.forward(model_tokens, current_state, preprocess_only=True)

    # print(f'### model ###\n[{tokenizer.decode(model_tokens)}]')

    out[0] = -999999999  # disable <|endoftext|>
    out[200] += newline_adj
    if newline_adj > 0:
        out[15] += newline_adj / 2 # '.'
    return out


all_state = {}


def save_all_stat(srv, name, last_out):
    n = f'{name}_{srv}'
    all_state[n] = {}
    all_state[n]['out'] = last_out
    all_state[n]['rnn'] = copy.deepcopy(current_state)
    all_state[n]['token'] = copy.deepcopy(model_tokens)


def load_all_stat(srv, name):
    global model_tokens, current_state
    n = f'{name}_{srv}'
    current_state = copy.deepcopy(all_state[n]['rnn'])
    model_tokens = copy.deepcopy(all_state[n]['token'])
    return all_state[n]['out']


########################################################################################################

# Run inference
print(f'\nRun prompt...')

# out = run_rnn(tokenizer.encode(init_prompt))
gc.collect()
torch.cuda.empty_cache()
out = None
save_all_stat('', 'chat_init', out)

srv_list = ['dummy_server']
for s in srv_list:
    save_all_stat(s, 'chat', out)

# print(f'### prompt ###\n[{tokenizer.decode(model_tokens)}]\n')


def reply_msg(msg):
    print(f'{bot}{interface} {msg}\n')


def on_message(message):
    global model_tokens, current_state

    srv = 'dummy_server'

    msg = message.replace('\\n', '\n').strip()
    if len(msg) > 1000:
        reply_msg('your message is too long (max 1000 tokens)')
        return

    x_temp = 1.0
    x_top_p = 0.95
    if ("-temp=" in msg):
        x_temp = float(msg.split("-temp=")[1].split(" ")[0])
        msg = msg.replace("-temp=" + f'{x_temp:g}', "")
        # print(f"temp: {x_temp}")
    if ("-top_p=" in msg):
        x_top_p = float(msg.split("-top_p=")[1].split(" ")[0])
        msg = msg.replace("-top_p=" + f'{x_top_p:g}', "")
        # print(f"top_p: {x_top_p}")
    if x_temp <= 0.2:
        x_temp = 0.2
    if x_temp >= 5:
        x_temp = 5
    if x_top_p <= 0:
        x_top_p = 0

    if msg == '+reset':
        out = load_all_stat('', 'chat_init')
        save_all_stat(srv, 'chat', out)
        reply_msg("Chat reset.")
        return

    elif msg[:5].lower() == '+gen ' or msg[:4].lower() == '+qa ' or msg.lower() == '+more' or msg.lower() == '+retry':

        if msg[:5].lower() == '+gen ':
            new = '\n' + msg[5:].strip()
            # print(f'### prompt ###\n[{new}]')
            current_state = None
            out = run_rnn(tokenizer.encode(new))
            save_all_stat(srv, 'gen_0', out)

        elif msg[:4].lower() == '+qa ':
            out = load_all_stat('', 'chat_init')

            real_msg = msg[4:].strip()
            new = f"{user}{interface} {real_msg}\n\n{bot}{interface}"
            # print(f'### qa ###\n[{new}]')

            out = run_rnn(tokenizer.encode(new))
            save_all_stat(srv, 'gen_0', out)

            # new = f"\nThe following is an excellent Q&A session consists of detailed and factual information.\n\nQ: What is 3+5?\nA: The answer is 8.\n\nQ: {msg[9:].strip()}\nA:"
            # print(f'### prompt ###\n[{new}]')
            # current_state = None
            # out = run_rnn(tokenizer.encode(new))
            # save_all_stat(srv, 'gen_0', out)

        elif msg.lower() == '+more':
            try:
                out = load_all_stat(srv, 'gen_1')
                save_all_stat(srv, 'gen_0', out)
            except:
                return

        elif msg.lower() == '+retry':
            try:
                out = load_all_stat(srv, 'gen_0')
            except:
                return

        begin = len(model_tokens)
        out_last = begin
        for i in range(1024):
            token = sample_logits(
                out,
                model_tokens,
                args.ctx_len,
                temperature=x_temp,
                top_p_usual=x_top_p,
                top_p_newline=x_top_p,
            )
            if token == 0:
                break
            if msg[:4].lower() == '+qa ':
                out = run_rnn([token], newline_adj=-1)
            else:
                out = run_rnn([token])

            xxx = tokenizer.decode(model_tokens[out_last:])
            if '\ufffd' not in xxx:
                print(xxx, end='', flush=True)
                out_last = begin + i + 1
        print('\n')
        # send_msg = tokenizer.decode(model_tokens[begin:]).strip()
        # print(f'### send ###\n[{send_msg}]')
        # reply_msg(send_msg)
        save_all_stat(srv, 'gen_1', out)

    else:
        if msg.lower() == '+alt':
            try:
                out = load_all_stat(srv, 'chat_pre')
            except:
                return
        else:
            out = load_all_stat(srv, 'chat')
            if out == None:
                new = f"{user}{interface} {msg}<|endoftext|>\n\n{bot}{interface} "
            else:
                new = f"\n\n{user}{interface} {msg}<|endoftext|>\n\n{bot}{interface} "
            # new = f"{msg}"
            # print(f'### add ###\n[{new}]')
            out = run_rnn(tokenizer.encode(new), newline_adj=-999999999)
            save_all_stat(srv, 'chat_pre', out)

        begin = len(model_tokens)
        out_last = begin
        print(f'{bot}{interface}', end='', flush=True)
        # print(f'', end='', flush=True)
        for i in range(999):
            # if i <= 0:
            #     newline_adj = -999999999
            # elif i <= 30:
            #     newline_adj = (i - 30) / 10
            # elif i <= 130:
            #     newline_adj = 0
            # else:
            #     newline_adj = (i - 130) * 0.25  # MUST END THE GENERATION
            token = sample_logits(
                out,
                model_tokens,
                args.ctx_len,
                temperature=x_temp,
                top_p_usual=x_top_p,
                top_p_newline=x_top_p,
            )

            if token == 0:
                break

            out = run_rnn([token])

            xxx = tokenizer.decode(model_tokens[out_last:])
            # if '\ufffd' not in xxx:
            print(xxx, end='', flush=True)
            out_last = begin + i + 1

            # send_msg = tokenizer.decode(model_tokens[begin:])
            # if '\n\n' in send_msg:
            #     send_msg = send_msg.strip()
            #     break

            # send_msg = tokenizer.decode(model_tokens[begin:]).strip()
            # if send_msg.endswith(f'{user}{interface}'): # warning: needs to fix state too !!!
            #     send_msg = send_msg[:-len(f'{user}{interface}')].strip()
            #     break
            # if send_msg.endswith(f'{bot}{interface}'):
            #     send_msg = send_msg[:-len(f'{bot}{interface}')].strip()
            #     break

        # print(f'{model_tokens}')
        # print(f'[{tokenizer.decode(model_tokens)}]')

        # print(f'### send ###\n[{send_msg}]')
        # reply_msg(send_msg)
        save_all_stat(srv, 'chat', out)


# print(HELP_MSG)

while True:
    # msg = input(f'{user}{interface} ')
    msg = input(f'\nCâu mới: ')
    if len(msg.strip()) > 0:
        on_message(msg)
    else:
        print('Erorr: please say something')
    print("")