import numpy as np

FLICKR_DATA_PATH= "/shared/flickr_style/"
STYLE_NAME_PATH = FLICKR_DATA_PATH + "style_names.txt"
OBJS_BY_FILENAME_PATH = FLICKR_DATA_PATH + "flickr_obj_label.txt"
NUM_OF_STYLES = 20
LIMIT_RETURN_IMAGES = 20
TOP_N = 5
OBJ_WEIGHT = [0.9, 0.7, 0.5, 0.3, 0.1]
STYLE_WEIGHT = [0.9, 0.7, 0.5, 0.3, 0.1]

flickr_test_set = np.loadtxt(FLICKR_DATA_PATH+'test.txt', str, delimiter='\t')
flickr_test_filename = [readline.split()[0].split('/')[-1] for readline in flickr_test_set]
flickr_test_label = [int(readline.split()[1]) for readline in flickr_test_set]
flickr_test_dict = dict(zip(flickr_test_filename, flickr_test_label))

flickr_train_set = np.loadtxt(FLICKR_DATA_PATH+'train.txt', str, delimiter='\t')
flickr_train_filename = [readline.split()[0].split('/')[-1] for readline in flickr_train_set]
flickr_train_label = [int(readline.split()[1]) for readline in flickr_train_set]
flickr_train_dict = dict(zip(flickr_train_filename, flickr_train_label))

def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z

def getStyleList():
    f = open(STYLE_NAME_PATH, "r")
    style_list = []
    while 1:
        line = f.readline()
        if not line: break
        line = line.split('\n')
        style_list.append(line[0])

    f.close()
    return style_list

# Function: getStyleClassifier
# Input:    none
# Output:   style classifier
# Desc:     It returns each bunch of files and its predicted objects by index
#           of style.
def getStyleClassifier():
    # file open
    f = open(OBJS_BY_FILENAME_PATH, "r")

    # initialize classifier
    styleClassifier = [None]*NUM_OF_STYLES
    for i in xrange(0, NUM_OF_STYLES):
        styleClassifier[i] = {}

    # get predicted object and its score from file
    while 1:
        # Each line by line, there are file name and its top-5 predicted
        # objects and scores
        line = f.readline()
        if not line: break

        # get file name
        lineEle = line.split(',')
        filePath = lineEle[0]
        fileName = filePath.split('/')[-1]

        # get predicted objects and scores
        fileObjs = []
        for j in xrange(1, len(lineEle), 2):
            # we expect 5 predicted objects
            if len(lineEle) != 11:
                continue

            try:
                # get objects label and its score
                labelCode = lineEle[j]
                labelScore = lineEle[j+1].split('\n')[0]

                # make dictionary with them and append to the list we defined
                tmp = {}
                tmp[labelCode] = labelScore
                fileObjs.append(tmp)

                # attach it to the classifier's dictionary
                styleClassifier[flickr_all_dict[fileName]][fileName] = fileObjs

            except:
                # to detect some error
                print "error!"
                print fileName
                print flickr_all_dict[fileName]
                print fileObjs

    f.close()
    return styleClassifier


flickr_all_dict = merge_two_dicts(flickr_train_dict, flickr_test_dict)
style_list = getStyleList()
styleClassifier = getStyleClassifier()

# Function: recommender
# Input:    the results of uploaded image analysis (predicted objects and predited
#           style
# Output:   recommanded images
# Desc:     It
def recommender(res):
    # predicted results by machine learning algorithm
    pred_obj = res["predict_obj"]
    pred_sty = res["predict_style"]


    # make matrix and normalize it to find good answers
    pred_objcol = np.ones((5,1))
    pred_styrow = np.ones((1,5))
    for i in range(0, len(pred_sty)):
        pred_objcol[i][0] = float(pred_obj[i]["score"])*OBJ_WEIGHT[i]
        pred_styrow[0][i] = float(pred_sty[i]["score"])*STYLE_WEIGHT[i]

    mat = np.dot(pred_objcol, pred_styrow)  # 5 by 5 matrix
    mat = mat/mat.flatten().sum()           # normalize it
    mat_flat = mat.flatten()                # flatten it
    idxs = np.argsort(mat_flat)             # 0 to 24

    # normailize again for top N features
    sum = 0
    for i in range(1, TOP_N+1):
        sum = sum + mat_flat[idxs[-i]]
    top_mat = mat/sum

    recom = []
    for i in range(1, TOP_N+1):
        row_n = idxs[-i]/5
        col_n = idxs[-i]%5

        # high scored predicted style
        print pred_sty[col_n]["name"]
        # high scored predicted object
        print pred_obj[row_n]["name"],"(", pred_obj[row_n]["label"],")"
        # its score
        propo = top_mat[row_n][col_n]
        print propo

        # count the number of images that will return
        num_return_imgs = 0

        # index of predicted style
        rec_idx = style_list.index(pred_sty[col_n]["name"])
        for fileName, objs in styleClassifier[rec_idx].iteritems():
            # now we have same style images, but we have to consider objects
            # from now. each image have ranked objects list.
            for j in range(5):
                if pred_obj[row_n]["label"] in objs[j]:
                    # if there is predicted label in this image, get filename
                    # and score
                    tmp = {"filename": fileName, "style":pred_sty[col_n]["name"], "object":pred_obj[row_n]["name"], "style_obj_score": propo, "obj_score":float(objs[j][pred_obj[row_n]["label"]])}
                    if num_return_imgs > int(propo*LIMIT_RETURN_IMAGES):
                        continue
                    recom.append(tmp)
                    num_return_imgs = num_return_imgs + 1
        print "---------------"

    return recom

#res = {'predict_obj': [{'score': '0.121914', 'name': 'gown', 'label': 'n03450230'}, {'score': '0.119266', 'name': 'jersey,T-shirt,teeshirt', 'label': 'n03595614'}, {'score': '0.0851326', 'name': 'maillot', 'label': 'n03710637'}, {'score': '0.0685063', 'name': 'website,website,internetsite,site', 'label': 'n06359193'}, {'score': '0.0598228', 'name': 'sunglass', 'label': 'n04355933'}], 'predict_style': [{'score': '0.252059', 'name': 'Bright'}, {'score': '0.212549', 'name': 'Pastel'}, {'score': '0.110541', 'name': 'Vintage'}, {'score': '0.0962888', 'name': 'Romantic'}, {'score': '0.0546268', 'name': 'Depth of Field'}]}
#recommender(res)
