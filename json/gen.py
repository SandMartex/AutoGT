import json

def write_json(path):
    dic = {}
    dic['depth']      = [2, 3, 4]
    dic['hidden_in']  = [24, 28, 32]
    dic['num_heads']  = [2, 3, 4]
    dic['att_size']   = [6, 8]
    dic['hidden_mid'] = [24, 28, 32]
    dic['ffn_size']   = [24, 28, 32]
    dic['mask']       = [1, 2, 3, 4, 5]
    with open(path, 'w') as f:
        json.dump(dic, f, indent=1)
    return

if __name__ == '__main__':
    path = 'PROTEINS.json'
    write_json(path)
