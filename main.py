from py_code import *

def main():
    # sample
    sample, sampleLabel = train_data.sample(BATCH_SIZE)

    if DEBUG:
        print('-----hsic test-----')
        hsic.selfTest()
        print('\n\n-----U test----')
        
        _ = update_U(sample,debug=True)

    # init weight
    init_weight(INIT_EPOCH, INIT_EBCODER_LR, INIT_DECODER_LR)
    if DEBUG:
        print('-----network test----')
        test_network()

    # check U mode
    if CHECK_U:
        check_U_mode()
    
    # train model
    train(EPOCH, ITER, ENCODER_LR, DECODER_LR, SCHEDULER_STEP, SCHEDULER_GAMMA)

    # evel mode
    encoder.eval()
    decoder.eval()

    # plot
    plot_result(U = update_U(sample))
    if DEBUG:
        plot_distribution(axis = 0)

    # eval accuracy
    acc = accuracy()

    # save model
    torch.save(encoder.state_dict(), MODEL_PATH)

    # write log
    with open('log.txt','w') as log:
        with open('py_code/config.py','r') as f:
            all = f.read()
            log.write(all)
        log.write('\n\n------------')
        log.write(f'\ndataset: {train_data._type}, k = {train_data.label_num}')
        log.write(f'\nacc: {acc}')
        log.write(f'\nmodel: {encoder}')
        
if __name__ == '__main__':
    main()