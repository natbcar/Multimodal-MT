import subprocess
import sys
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
install("sacrebleu==2.0.0")

import argparse
from sacrebleu.metrics import BLEU, CHRF, TER
from Model import *
from DataPrep import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

# Random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def run_epoch(data, model, loss_compute, epoch, multi_modal=False):
    start = time.time()
    total_tokens = 0.
    total_loss = 0.
    tokens = 0.
    img_feats = None
    for i, batch in enumerate(data):
        if multi_modal:
            img_feats = torch.from_numpy(batch.img_feats).to(DEVICE)
            out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask, img_feats)
        else:
            out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        
        # delete image features to free up memory
        if multi_modal: 
            del img_feats

        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch {:d} Batch: {:d} Loss: {:.4f} Tokens per Sec: {:.2f}s".format(
                epoch, i - 1, loss / batch.ntokens, (tokens.float() / elapsed / 1000.)))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


def train(n_epochs, data, model, criterion, optimizer, save_file, mm=False):
    """
    Train and Save the model.
    """
    # init loss as a large value
    best_dev_loss = 1e5

    for epoch in range(n_epochs):
        # Train model
        model.train()
        run_epoch(data.train_data, model, SimpleLossCompute(
            model.generator, criterion, optimizer), epoch, mm)
        model.eval()

        # validate model on dev dataset
        print('>>>>> Evaluate')
        dev_loss = run_epoch(data.dev_data, model, SimpleLossCompute(
            model.generator, criterion, None), epoch, mm)
        print('<<<<< Evaluate loss: {:.2f}'.format(dev_loss))

        # save the model with best-dev-loss
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            # SAVE_FILE = 'save/model.pt'
            torch.save(model.state_dict(), save_file)

        print(f">>>>> current best loss: {best_dev_loss}")


def greedy_decode(model, src, src_mask, max_len, start_symbol, img=None):
    """
    Translate src with model
    """
    # decode the src
    memory = model.encode(src, src_mask)
    # init 1×1 tensor as prediction，fill in ('BOS')id, type: (LongTensor)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    #  run 遍历输出的长度下标
    for i in range(max_len-1):
        # decode one by one
        out = model.decode(memory,
                           src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1)).type_as(src.data)),
                           img)
        #  out to log_softmax
        prob = model.generator(out[:, -1])
        #  get the max-prob id
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        #  concatnate with early predictions
        ys = torch.cat([ys, torch.ones(1, 1).type_as(
            src.data).fill_(next_word)], dim=1)
    return ys


def evaluate(data, model, max_length, multimodal=False):
    """
    Make prediction with trained model, and print results.
    """
    translations = []
    source_sents = []
    refs = []
    model.eval()
    cnt = 0
    with torch.no_grad():
        for batch in data.test_data:
            for i in range(batch.src.size(0)):
                en_sent = " ".join([data.en_index_dict[w.item()] for w in batch.src[i]])
                source_sents.append(en_sent)

                # cn_sent = " ".join([data.cn_index_dict[w.item()] for w in batch.trg[i]])
                cn_sent = [data.cn_index_dict[w.item()] for w in batch.trg[i]]
                cleaned_cn_sent = []
                for j in range(1, len(cn_sent)):
                    if cn_sent[j] == "EOS":
                        break
                    cleaned_cn_sent.append(cn_sent[j])

                refs.append(" ".join(cleaned_cn_sent))

                # conver English to tensor
                #src = torch.from_numpy(np.array(batch.src[i])).long().to(DEVICE)
                src = batch.src[i].unsqueeze(0)

                # set attention mask
                src_mask = (src != 0).unsqueeze(-2)

                if multimodal:
                    # img = data.dev_feats[i]
                    img = torch.from_numpy(batch.img_feats[i]).to(DEVICE)
                    # img = torch.reshape(img, (49, 2048))
                    out = greedy_decode(
                        model, src, src_mask, max_len=max_length, start_symbol=data.cn_word_dict["BOS"], img=img.unsqueeze(0))
                else:
                    out = greedy_decode(
                        model, src, src_mask, max_len=max_length, start_symbol=data.cn_word_dict["BOS"])

                # save all in the translation list
                translation = []
                # convert id to Chinese, skip 'BOS' 0.
                # 遍历翻译输出字符的下标（注意：跳过开始符"BOS"的索引 0）
                for j in range(1, out.size(1)):
                    sym = data.cn_index_dict[out[0, j].item()]
                    if sym != 'EOS':
                        translation.append(sym)
                    else:
                        break
                translations.append(" ".join(translation))

            if cnt % 10 == 0:
                    print(cnt)
                    print("source: ", source_sents[-1])
                    print("target: ", refs[-1])
                    print("translation: ", translations[-1])
            cnt += 1

    print("{} total reference sentences".format(len(refs)))
    refs = [refs]

    # score translations
    bleu = BLEU()
    chrf = CHRF()
    ter = TER()

    bleu_score = bleu.corpus_score(translations, refs).score
    chrf_score = chrf.corpus_score(translations, refs).score
    ter_score = ter.corpus_score(translation, refs).score

    print("bleu : {}".format(bleu_score))
    print("chrf : {}".format(chrf_score))
    print("ter : {}".format(ter_score))

    return bleu_score, chrf_score, ter_score, translations, refs[0]

def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h = 8, dropout=0.1, enc_mm=False, dec_mm=False, use_nonlinear_projection=False):
    c = copy.deepcopy
    #  Attention 
    attn = MultiHeadedAttention(h, d_model).to(DEVICE)
    enc_mm_attn = None
    img_proj_enc = None
    dec_mm_attn = None
    img_proj_dec = None
    if enc_mm:
        enc_mm_attn = MultiHeadedAttention(h, d_model).to(DEVICE)
        img_proj_enc = ImageLocalFeaturesProjector(1, 2048, d_model, 0.1, use_nonlinear_projection).to(DEVICE)
    if dec_mm:
        dec_mm_attn = MultiHeadedAttention(h, d_model).to(DEVICE)
        img_proj_dec = ImageLocalFeaturesProjector(1, 2048, d_model, 0.1, use_nonlinear_projection).to(DEVICE)
        
    #  FeedForward 
    ff = PositionwiseFeedForward(d_model, d_ff, dropout).to(DEVICE)
    #  Positional Encoding
    position = PositionalEncoding(d_model, dropout).to(DEVICE)
    #  Transformer 
    model = Transformer(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout, enc_mm_attn).to(DEVICE), N).to(DEVICE),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout, dec_mm_attn).to(DEVICE), N).to(DEVICE),
        nn.Sequential(Embeddings(d_model, src_vocab).to(DEVICE), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab).to(DEVICE), c(position)),
        Generator(d_model, tgt_vocab), img_proj_enc, img_proj_dec).to(DEVICE)
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    # Paper title: Understanding the difficulty of training deep feedforward neural networks Xavier
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model.to(DEVICE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-epochs", type=int, help="number of training epochs")
    parser.add_argument("--n-layers", type=int, help="number of encoder and decoder layers")
    parser.add_argument("--batch-size", type=int, help="batch size for training")
    parser.add_argument("--h-num", type=int, help="number of attention heads")
    parser.add_argument("--d-model", type=int, help="dimension of model hiden layer")
    parser.add_argument("--d-ff", type=int, help="dimension of feed forward layer")
    parser.add_argument("--dropout", type=float, help="dropout probability")
    parser.add_argument("--max-len", type=int, help="max sequence length")
    parser.add_argument("--enc-mm", type=int, help="use multimodal attention in encoder")
    parser.add_argument("--dec-mm", type=int, help="use multimodal attention in decodder")
    parser.add_argument("--non-linear-proj", type=int, help="1 to use 0 to not use")
    parser.add_argument("--degrade-source", type=int, help="mask random words in source sentences")
    parser.add_argument("--train-file", type=str, help="path to training file")
    parser.add_argument("--val-file", type=str, help="path to validation file")
    parser.add_argument("--test-file", type=str, help="path to test file")
    parser.add_argument("--img-path", type=str, help="path to img directory")
    parser.add_argument("--save-file", type=str, help="path to save model to")
    parser.add_argument("--trans-file", type=str, help="path to write translations to")
    parser.add_argument("--scores-file", type=str, help="path to write results to")
    args = parser.parse_args()

    # Step 1: Data Preprocessing
    print("preparing data")
    if args.enc_mm or args.dec_mm:
        data = PrepareData(args.train_file, args.val_file, args.test_file, args.batch_size, args.img_path, degrade_source=args.degrade_source)
    else:
        data = PrepareData(args.train_file, args.val_file, args.test_file, args.batch_size, None, degrade_source=args.degrade_source)
    src_vocab = len(data.en_word_dict)
    tgt_vocab = len(data.cn_word_dict)
    print(f"src_vocab {src_vocab}")
    print(f"tgt_vocab {tgt_vocab}")

    # Step 2: Init model
    model = make_model(
        src_vocab,
        tgt_vocab,
        args.n_layers,
        args.d_model,
        args.d_ff,
        args.h_num,
        args.dropout,
        enc_mm=args.enc_mm, 
        dec_mm=args.dec_mm, 
        use_nonlinear_projection=args.non_linear_proj
    )

    print(">>>>>>> start train")
    train_start = time.time()
    criterion = LabelSmoothing(tgt_vocab, padding_idx=0, smoothing=0.0)
    optimizer = NoamOpt(args.d_model, 1, 2000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    mm = args.enc_mm or args.dec_mm

    train(args.n_epochs, data, model, criterion, optimizer, args.save_file, mm)
    print(f"<<<<<<< finished train, cost {time.time()-train_start:.4f} seconds")

    model.load_state_dict(torch.load(args.model_path))
    print(">>>>>>> start evaluate")
    evaluate_start = time.time()
    bleu_score, chrf_score, ter_score, translations, refs = evaluate(data, model, args.max_len, mm)
    print(f"<<<<<<< finished evaluate, cost {time.time()-evaluate_start:.4f} seconds")

    results_dict = {"bleu": bleu_score,
                    "chrf": chrf_score,
                    "ter": ter_score}
                    
    # Save results
    with open(args.scores_file, "w") as f:
        for key, value in results_dict.items(): 
            f.write('%s:%s\n' % (key, value))

    with open(args.trans_file, "w") as f:
        for i in range(len(translations)):
            f.write(translations[i] + "\t" + refs[i] + "\n")
