import os


def calc_bleu_score(predictions, references, log_dir=None, multi_ref=False):
    pred_file = os.path.join(log_dir, 'pred.txt')
    ref_file = os.path.join(log_dir, 'ref.txt')

    if multi_ref:
        num_sents = len(references)
        num_refs = len(references[0])
        for ref_idx in range(num_refs):
            with open(ref_file + str(ref_idx), 'w', encoding='utf-8') as f:
                for sent_idx in range(num_sents):
                    print(references[sent_idx][ref_idx], file=f, )
    else:
        with open(ref_file, 'w', encoding='utf-8') as f:
            for s in references:
                print(s, file=f, )
    with open(pred_file, 'w', encoding='utf-8') as f:
        for s in predictions:
            print(s, file=f, )

    temp = os.path.join(log_dir, "result.txt")

    command = "perl utils/multi-bleu.perl " + ref_file + "<" + pred_file + "> " + temp
    os.system(command)
    with open(temp) as ft:
        result = ft.read()
    os.remove(temp)
    score = float(result.split()[2][:-1])

    return score