########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

print('Loading...')
from src.model_run import RWKV_RNN
import numpy as np
import os, copy, types, gc, sys
import torch
from src.utils import TOKENIZER

try:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
except:
    pass
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
np.set_printoptions(precision=4, suppress=True, linewidth=200)

CHAT_LANG = 'Vietnamese'  # English Chinese
# CHAT_LANG = 'English'  # English Chinese

# tokenizer = TOKENIZER("20B_tokenizer.json")
tokenizer = TOKENIZER("../checkpoint/vitok20k/tokenizer.json")
# tokenizer = TOKENIZER("rwkv_vocab_v20230424.txt")

args = types.SimpleNamespace()
args.RUN_DEVICE = "cuda"  # 'cpu' (already very fast) // 'cuda'
args.FLOAT_MODE = "bf16"  # fp32 (good for CPU) // fp16 (recommended for GPU) // bf16 (less accurate)
# args.vocab_size = len(tokenizer.tokenizer)
# args.vocab_size = 50277
args.vocab_size = 20000
args.head_qk = 0
args.pre_ffn = 0
args.grad_cp = 0
args.my_pos_emb = 0

# args.MODEL_NAME = '../checkpoint/RWKV-4-World-0.1B-v1-20230520-ctx4096'
# args.MODEL_NAME = '../checkpoint/rwkv4_169m_ft_chat_20231024/rwkv-30'
args.MODEL_NAME = '../checkpoint/rwkv4_vitok20k_l12_768_128/rwkv-20-qa'
# args.MODEL_NAME = '../checkpoint/rwkv4_pileplus_ft_20231007/rwkv-12'
# args.MODEL_NAME = '../checkpoint/rwkv4_the_thao/rwkv-40'
# args.MODEL_NAME = '../checkpoint/rwkv4_the_thao_chat/rwkv-42'
args.n_layer = 12
args.n_embd = 768
# args.ctx_len = 1024
# args.ctx_len = 4096
args.ctx_len = 128

# tokenizer = AutoTokenizer.from_pretrained("../checkpoint/vitok20k/")

args.MODEL_NAME = '../checkpoint/rwkv4_vitok20k_L24_2048_ctx1024_20231029/rwkv-4'
args.n_layer = 24
args.n_embd = 2048
args.ctx_len = 1024

# args.MODEL_NAME = '../checkpoint/rwkv4_1.5B_lora_8_16_13B_20231029/RWKV-4-World-1.5B-ft-13B_merge_lora_1'
# args.n_layer = 24
# args.n_embd = 2048
# args.ctx_len = 4096

# args.MODEL_NAME = '../checkpoint/rwkv4_3B_lora_8_16_13B_20231030/RWKV-4-World-3B-ft-13B_merge_lora_0'
# args.n_layer = 32
# args.n_embd = 2560
# args.ctx_len = 4096

default_stop = [
    "\n\nUser",
    "\nUser",
    "\n\nQuestion",
    "\nQuestion",
    "\n\nQ",
    "\nQ",
    "\n\nHuman",
    "\nHuman",
    "\n\nBob",
    "\nBob",
    "\n\nBot",
    "\nBot"
]

if CHAT_LANG == 'English':
    user = "User"
    bot = "Assistant"
    interface = ":"
    end_of_message = "\n"

    # The following is a verbose and detailed conversation between an AI assistant called {bot}, and a human user called {user}. {bot} is intelligent, knowledgeable, wise and polite.
    # The following is a conversation between a highly knowledgeable and intelligent AI called {bot}, and a human called {user}. In the following interactions, {user} and {bot} converse in natural language, and {bot} do its best to answer {user}'s questions. {bot} is respectful, polite and inclusive. {bot} knows a lot, and always tells the truth.

    init_prompt = f'''
The following is a verbose and detailed conversation between an AI assistant called {bot}, and a human user called {user}. {bot} is intelligent, knowledgeable, wise and polite.

{user}{interface} french revolution what year{end_of_message}
{bot}{interface} The French Revolution started in 1789, and lasted 10 years until 1799.{end_of_message}
{user}{interface} 3+5=?{end_of_message}
{bot}{interface} The answer is 8.{end_of_message}
{user}{interface} guess i marry who ?{end_of_message}
{bot}{interface} Only if you tell me more about yourself - what are your interests?{end_of_message}
{user}{interface} solve for a: 9-a=2{end_of_message}
{bot}{interface} The answer is a = 7, because 9 - 7 = 2.{end_of_message}
{user}{interface} wat is lhc{end_of_message}
{bot}{interface} LHC is a high-energy particle collider, built by CERN, and completed in 2008. They used it to confirm the existence of the Higgs boson in 2012.{end_of_message}

'''
    HELP_MSG = '''Commands:
say something --> chat with bot. use \\n for new line.
+alt --> alternate chat reply
+reset --> reset chat

+gen YOUR PROMPT --> free generation with any prompt. use \\n for new line.
+qa YOUR QUESTION --> free generation - ask any question (just ask the question). use \\n for new line.
+more --> continue last free generation (only for +gen / +qa)
+retry --> retry last free generation (only for +gen / +qa)

Now talk with the bot and enjoy. Remember to +reset periodically to clean up the bot's memory. Use RWKV-4 14B for best results.
This is not instruct-tuned for conversation yet, so don't expect good quality. Better use +gen for free generation.
'''
elif CHAT_LANG == 'Vietnamese':
    # user = "Question"
    user = "User"
    bot = "Bot"
    # bot = "Assistant"
    # bot = "Answer"
    # bot = "AI"
    interface = ":"
    # end_of_message = "<|endoftext|>\n"
    # end_of_message = "<|endoftext|>"
    end_of_message = "\n"
    # end_of_message = ""

    # init_prompt = f'''Sau đây là cuộc trò chuyện dài và chi tiết giữa trợ lý AI có tên là {bot} và một người dùng tên là {user}. {bot} thông minh, hiểu biết, khôn ngoan và lịch sự.{end_of_message}\n'''
    init_prompt = f''''''
    init_prompt1 = f'''
    Sau đây là cuộc trò chuyện dài và chi tiết giữa trợ lý AI có tên là {bot} và một người dùng tên là {user}. {bot} thông minh, hiểu biết, khôn ngoan và lịch sự.

    {user}{interface} Cách mạng Pháp năm nào{end_of_message}
    {bot}{interface} Cách mạng Pháp bắt đầu từ năm 1789 và kéo dài 10 năm cho đến năm 1799.{end_of_message}
    {user}{interface} 3+5=?{end_of_message}
    {bot}{interface} Câu trả lời là 8.{end_of_message}
    {user}{interface} đoán xem tôi cưới ai ?{end_of_message}
    {bot}{interface} Chỉ khi bạn cho tôi biết thêm về bản thân - sở thích của bạn là gì?{end_of_message}
    {user}{interface} Giả tìm a: 9-a=2{end_of_message}
    {bot}{interface} Đáp án là a = 7, vì 9 - 7 = 2.{end_of_message}
    {user}{interface} lhc là gì{end_of_message}
    {bot}{interface} LHC là máy va chạm hạt năng lượng cao, do CERN chế tạo và hoàn thành vào năm 2008. Họ đã sử dụng nó để xác nhận sự tồn tại của boson Higgs vào năm 2012.'''
    HELP_MSG = '''Commands:
    say something --> chat with bot. use \\n for new line.
    +alt --> alternate chat reply
    +reset --> reset chat

    +gen YOUR PROMPT --> free generation with any prompt. use \\n for new line.
    +qa YOUR QUESTION --> free generation - ask any question (just ask the question). use \\n for new line.
    +more --> continue last free generation (only for +gen / +qa)
    +retry --> retry last free generation (only for +gen / +qa)

    Now talk with the bot and enjoy. Remember to +reset periodically to clean up the bot's memory. Use RWKV-4 14B for best results.
    This is not instruct-tuned for conversation yet, so don't expect good quality. Better use +gen for free generation.
    '''

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

    # print(f'### model ###\n[{tokenizer.tokenizer.decode(model_tokens)}]')

    # out[0] = -999999999  # disable <|endoftext|>
    try:
        out[tokenizer.tokenizer.non_decode] = -999999999  # disable <|endoftext|>
    # out[187] += newline_adj
    #     out[261] += newline_adj
    except:
        pass
    # if newline_adj > 0:
    #     out[15] += newline_adj / 2 # '.'
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
if init_prompt:
    out = run_rnn(tokenizer.encode(init_prompt))
else:
    out = None

gc.collect()
torch.cuda.empty_cache()

save_all_stat('', 'chat_init', out)

srv_list = ['dummy_server']
for s in srv_list:
    save_all_stat(s, 'chat', out)

print(f'### prompt ###\n[{tokenizer.tokenizer.decode(model_tokens)}]\n')


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
    x_top_p = 0.85
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
            # out = run_rnn(tokenizer.tokenizer.encode(new))
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
        for i in range(150):
            token = tokenizer.sample_logits(
                out,
                temperature=x_temp,
                top_p=0.5,
                top_k=50,
            )
            if msg[:4].lower() == '+qa ':
                out = run_rnn([token], newline_adj=-1)
            else:
                out = run_rnn([token])

            xxx = tokenizer.tokenizer.decode(model_tokens[out_last:])
            if '\ufffd' not in xxx:
                print(xxx, end='', flush=True)
                out_last = begin + i + 1
        print('\n')
        # send_msg = tokenizer.tokenizer.decode(model_tokens[begin:]).strip()
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
            msg = msg.replace("\n\n", '\n')
            new = f"{user}{interface} {msg}{end_of_message}\n{bot}{interface}"
            # new = f"{user}{interface} {msg}\n\n{bot}{interface}"
            # new = f"{user}{interface} {msg}\n{bot}{interface}"
            print(f'### add ###\n[{new}]')
            out = run_rnn(tokenizer.encode(new), newline_adj=-999999999)
            save_all_stat(srv, 'chat_pre', out)

        begin = len(model_tokens)
        out_last = begin
        print(f'{bot}{interface}', end='', flush=True)
        for i in range(999):
            if i <= 0:
                newline_adj = -999999999
            elif i <= 30:
                newline_adj = (i - 30) / 10
            elif i <= 130:
                newline_adj = 0
            else:
                newline_adj = (i - 130) * 0.25  # MUST END THE GENERATION
            token = tokenizer.sample_logits(
                out,
                temperature=x_temp,
                top_p=0.5,
                top_k=50,
            )
            out = run_rnn([token], newline_adj=newline_adj)
            # print(model_tokens[out_last:])
            xxx = tokenizer.decode(model_tokens[out_last:])
            if '\ufffd' not in xxx:
                print(xxx, end='', flush=True)
                out_last = begin + i + 1

            send_msg = tokenizer.decode(model_tokens[begin:])
            if '\n\n' in send_msg or any(i in send_msg for i in default_stop):
                # send_msg = send_msg.strip()
                print("")
                break

            # send_msg = tokenizer.tokenizer.decode(model_tokens[begin:]).strip()
            # if send_msg.endswith(f'{user}{interface}'): # warning: needs to fix state too !!!
            #     send_msg = send_msg[:-len(f'{user}{interface}')].strip()
            #     break
            # if send_msg.endswith(f'{bot}{interface}'):
            #     send_msg = send_msg[:-len(f'{bot}{interface}')].strip()
            #     break

        # print(f'{model_tokens}')
        # print(f'[{tokenizer.tokenizer.decode(model_tokens)}]')

        # print(f'### send ###\n[{send_msg}]')
        # reply_msg(send_msg)
        save_all_stat(srv, 'chat', out)


print(HELP_MSG)

while True:
    msg = input(f'{user}{interface} ')
    if len(msg.strip()) > 0:
        on_message(msg)
    else:
        print('Erorr: please say something')