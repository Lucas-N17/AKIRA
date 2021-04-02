import yaml
import numpy as np

data = yaml.safe_load(open('nlu\\train.yml', 'r', encoding='utf=8').read())

Inputs, outputs = [], []

for comand in data['comands']:
    Inputs.append(comand['input'].lower())
    outputs.append('{}\{}'.format(comand['entity'], comand['action']))


# Processar texto: palavras, caracteres, bytes, sub-palavras

chars = set()

for input in Inputs + outputs:
    for ch in input:
        if ch not in chars:
            chars.add(ch)

# Mapear char-idx

chr2idx = {}
idx2chr = {}

for i, ch in enumerate(chars):
    chr2idx[ch]: i
    idx2chr[i]: ch



max_seq = max(len(x) for x in Inputs)

print('Número de chars:', len(chars))
print('Maior seq:', max_seq)

# Criar o dataset one-hot(número de exemplos, tamanho da seq, num caracteres) 

input_data = np.zeros((len(Inputs), max_seq, len(chars)), dtype='int32')

for i, input in enumerate(Inputs):
    for k, ch in enumerate(input):
        input_data[i, k, chr2idx[ch]] = 1.0



print(input_data[0])



'''
print(Inputs)
print(outputs)
'''