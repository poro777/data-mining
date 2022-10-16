from config import *
from utils import *
from train import *
from network import *
from dataset import *

def main():
    sample = train_data.sample(BATCH_SIZE)

    if DEBUG:
        print('-----hsic test-----')
        hsic.selfTest()
        print('\n\n-----U test----')
        
        _ = update_U(sample,debug=True)

    init_weight(INIT_EPOCH, INIT_EBCODER_LR, INIT_DECODER_LR)
    if DEBUG:
        print('-----network test----')
        test_network()

    train(EPOCH, ITER, ENCODER_LR, DECODER_LR, SCHEDULER_STEP, SCHEDULER_GAMMA,
            LAMBDA, LAMBDA_STEP, LAMBDA_MAX, LAMBDA_GAMMA)

    encoder.eval()
    decoder.eval()
    # plot
    plot_result(U = update_U(sample))
    if DEBUG:
        plot_distribution(axis = 0)

    # eval
    acc = accuracy()

    torch.save(encoder.state_dict(), MODEL_PATH)
    
    with open('log.txt','w') as log:
        with open('config.py','r') as f:
            all = f.read()
            log.write(all)
        log.write('\n\n------------')
        log.write(f'\ndataset: {train_data._type}, k = {train_data.label_num}')
        log.write(f'\nacc: {acc}')
        
if __name__ == '__main__':
    main()