import random

def split_huge_file(file,out1,out2,percentage=0.90,seed=123):
    random.seed(seed)
    with open('../data/%s' % file, 'r',encoding = "ISO-8859-1") as fin, \
         open('../data/%s' % out1, 'w') as foutBig, \
         open('../data/%s' % out2, 'w') as foutSmall:

        for line in fin:
            r = random.random() 
            if r < percentage:
                foutBig.write(line)
            else:
                foutSmall.write(line)

if __name__== "__main__":
   split_huge_file('train_5500.label.txt','train_label.txt','test_label.txt')







