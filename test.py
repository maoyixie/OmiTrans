### test.py
"""
The testing part for OmiTrans (generate fake_A only)
"""
import time
from util import util
from params.test_params import TestParams
from datasets import create_single_dataloader
from models import create_model
from util.visualizer import Visualizer

if __name__ == '__main__':
    # Get testing parameter
    param = TestParams().parse()
    if param.deterministic:
        util.setup_seed(param.seed)

    # Dataset related
    dataloader, param.sample_list = create_single_dataloader(param, shuffle=False)
    print('The size of testing set is {}'.format(len(dataloader)))
    sample_list = dataloader.get_sample_list()
    feature_list_A = ["fake_feature_{}".format(i) for i in range(dataloader.get_A_dim())]  # Dummy feature names
    param.A_dim = dataloader.get_A_dim()
    param.B_dim = dataloader.get_B_dim()
    print('The dimension of omics type A is %d' % param.A_dim)
    print('The dimension of omics type B is ' % param.B_dim)
    if param.zo_norm:
        param.target_min = dataloader.get_values_min()
        param.target_max = dataloader.get_values_max()

    model = create_model(param)
    model.set_eval()
    visualizer = Visualizer(param)

    model.setup(param)
    if param.save_fake:
        fake_dict = model.init_fake_dict()

    test_start_time = time.time()

    for i, data in enumerate(dataloader):
        model.set_input(data)
        model.test()
        if param.save_fake:
            fake_dict = model.update_fake_dict(fake_dict)

    if param.save_fake:
        visualizer.save_fake_omics(fake_dict, sample_list, feature_list_A)

    print("Done generating fake A")
