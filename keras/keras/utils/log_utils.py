# -*- coding: utf-8 -*-
from datetime import datetime
result_path = "/home/hawk/imagenet2014/keras/results/"

def write_log(data_config, exp_result, network_config, internal_log):
    # run_date, data_type, n_data, test_set, n_test_set, score, running_time, n_conv_layer, n_max_pooling, n_batch, n_weight, epoch
    run_datetime = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
    merged = data_config + network_config + exp_result
    log_str = run_datetime + "\t" + "\t".join(str(log) for log in merged)
    log_str = "\n" + log_str + "\n".join(str(log) for log in internal_log)

    try:
        f = open(result_path +
                 datetime.now().strftime("%Y-%m-%d-%H%M%S") +
                 "_" + data_config[0] + ".txt", 'w'
                 )
        f.write(log_str)

        f.close()
    except Exception as e:
        print e
        pass

def main():
    write_log(['e', 12], ['wfwe', 34], ['wfwef', 56, 78])

if __name__ == '__main__':
    main()

