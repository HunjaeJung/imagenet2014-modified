# -*- coding: utf-8 -*-
from datetime import datetime
import numpy as np

result_path = "/home/hawk/imagenet2014/keras/results/"

def write_log(data_config, exp_result, network_config, history, labels):
    # run_date, data_type, n_data, test_set, n_test_set, score, running_time, n_conv_layer, n_max_pooling, n_batch, n_weight, epoch
    now = datetime.now().strftime("%Y-%m-%d-%H%M%S") # date for log file name
    run_datetime = datetime.now().strftime("%m/%d/%Y %H:%M:%S") # date in excel form

    merged = data_config + exp_result + network_config
    log_str = run_datetime + "\t" + "\t".join(str(log) for log in merged)
    log_str += "\n"
    log_str += "\n".join("epoch : " + str(history['epoch'][i]) + ", loss : " + str(history['loss'][i]) for i in range(0,len(history["epoch"])))

    label_str = "\n".join(str(log) for log in labels)

    try:
        file_name = result_path + now + "_" + data_config[0]

        f = open(file_name + ".txt", "w")
        f.write(log_str)
        f.close()

        f_label = open(file_name + "_labels.txt", "w")
        f_label.write(label_str)
        f_label.close()
    except Exception as e:
        print e
        pass

def main():
    d_conf = ['tiny imageNet', 10000, 'tiny-imagenet-200/val', 200]
    net_conf = ['relu', 4, 2, 32, 10000000, 10]
    exp = [10, 0.2, 123123123]
    hist = {"epoch" : [0,1,2,3,4,5], "loss" : [0.0,0.1,0.2,0.3,0.4,0.5]}
    labels = np.array(["lb1", "lb2", "lb3", "lb4"])
    write_log(d_conf, net_conf, exp, hist, labels)

if __name__ == '__main__':
    main()

