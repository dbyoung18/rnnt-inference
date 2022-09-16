import collections

def res2order_dict(file_name):
    with open(file_name) as f:
        res = f.readlines()
    d = {}
    for line in res:
        if '::' in line:
            l_split = line.split('::')
            idx = int(l_split[0])
            seq = l_split[1]
            d[idx] = seq
    order_d = collections.OrderedDict(sorted(d.items()))
    return order_d

def cmp_res(order_d1, order_d2):
    count = 0
    error_idx = []
    for k in order_d1.keys():
        if order_d1[k] != order_d2[k]:
            error_idx.append(k)
            print(k)
            print(order_d1[k])
            print(order_d2[k])
            print('-'*30)
            count += 1
    print('diff samples num:', count)
    print('error index list:', error_idx)

if __name__ == "__main__":
    f1_name = "../logs/hypotheses.log"
    f2_name = "../hypotheses.log"
    order_d1 = res2order_dict(f1_name)
    order_d2 = res2order_dict(f2_name)
    cmp_res(order_d1, order_d2)
    print('done')
