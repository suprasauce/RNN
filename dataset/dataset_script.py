from os.path import isfile, join, splitext
from os import listdir

dataset = ''
files = 0
root_dir = '/home/suprasauce/twitter_code_dataset/the-algorithm'

def rec(curr_path):
    global files, dataset
    # base case
    if isfile(curr_path) == True:
        if splitext(curr_path)[-1] == '.scala':
            f = open(curr_path)
            content = f.read().strip()
            content += '\n\n'
            dataset += content
            f.close()
            files += 1
        return
    
    curr_direcs = listdir(curr_path)
    for i in curr_direcs:
        rec(curr_path + '/' + i)
    

if __name__ == '__main__':
    rec(root_dir)
    f = open('dataset.txt', 'w')
    f.write(dataset)
    f.close()
    print(files)