import os
#from skimage import data
import numpy as np

def read_file(dir, dataset):
    dir_path = os.path.join(dir,dataset)
    if dataset=='train':
        for(root, dirs, files) in os.walk(dir_path):
            print root
            for file in files:
                if not '.txt' in file:
                    label = file.split("_")[0]
                    yield label, file
                
    elif dataset=='val':
        for(root, dirs, files) in os.walk(dir_path):
            for file in files:
                if '.txt' in file:
                    # this is val_annotaions.txt
                    f = open(os.path.join(root,file), 'r')
                    while 1:
                        line = f.readline()
                        if not line: break
                        line_seg = line.split()
                        img_filepath = os.path.join(root,'images',line_seg[0])
                        label = line_seg[1]
                        #yield label, data.imread(img_filepath)
                        yield label, img_filepath
                        
                        print(line_seg[0]) # image name
                        print(line_seg[1]) # label
                        print(line_seg[2]) # position x1
                        print(line_seg[3]) # position y1
                        print(line_seg[4]) # position x2
                        print(line_seg[5]) # position y2
                    f.close()

def load_data(path):
    for idx, (label, img) in enumerate(read_file(path,'train')):
        print label
    for idx, (label, img) in enumerate(read_file(path,'val')):
        print idx

if __name__ == '__main__':
    load_data('/Users/hunjae/Downloads/tiny-imagenet-200/')
