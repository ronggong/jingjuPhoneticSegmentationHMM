import matplotlib.pyplot as plt

def parsePdnnLog(filename):
    """
    read pdnn output log file
    :param filename:
    :return:
    """
    with open(filename,'rb') as f:
        lines = f.readlines()
    return lines

def trainingValidationError(lines_log):
    """
    parse log file to training error and validation error
    :param lines_log:
    :return:
    """
    errors_training = []
    errors_validation = []
    for line in lines_log:
        if 'training error' in line:
            index_start = line.index('training error')+len('training error')+1
            index_end   = index_start + 9
            error_training = float(line[index_start:index_end])
            errors_training.append(error_training)
        elif 'validation error' in line:
            index_start = line.index('validation error') + len('validation error') + 1
            index_end = index_start + 9
            error_validation = float(line[index_start:index_end])
            errors_validation.append(error_validation)
    return errors_training,errors_validation

def errorPlot(errors_training,errors_validation,title):
    """
    plot the errors training and validation
    :param errors_training:
    :param errors_validation:
    :return:
    """
    plt.figure()
    plt.plot(errors_training,label='training error')
    plt.plot(errors_validation,label='validation error')
    plt.legend(loc='best')
    plt.ylabel('error percentage %')
    plt.xlabel('epoch')
    plt.title(title)
    plt.show()

if __name__ == '__main__':
    filename_log = 'pdnnLog/dnn.training_phonemeSeg_2_256C01M5MB80DO5_plus_validation_72.log'
    title = '2 layers, 256 nodes, l rate 0.01, Momentum 0.5, Dropout 0.5, plus validation, 72 epochs'
    lines_log = parsePdnnLog(filename_log)
    errors_training,errors_validation = trainingValidationError(lines_log)
    errorPlot(errors_training,errors_validation,title)