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
    param = TestParams().parse()
    if param.deterministic:
        util.setup_seed(param.seed)

    # Load dataset
    dataloader, param.sample_list = create_single_dataloader(param, shuffle=False)
    print('The size of testing set is {}'.format(len(dataloader)))
    sample_list = dataloader.get_sample_list()
    # feature_list_A = ["fake_feature_{}".format(i) for i in range(dataloader.get_A_dim())]
    feature_list_A = ["fake_feature_{}".format(i) for i in range(param.output_chan_num)]
    param.A_dim = dataloader.get_A_dim()
    param.B_dim = dataloader.get_B_dim()
    print('The dimension of omics type A is %d' % param.A_dim)
    print('The dimension of omics type B is ', param.B_dim)
    print(f"A_dim: {param.A_dim}")
    print(f"B_dim: {param.B_dim}")
    print(f"Output channels: {param.output_chan_num}")

    if param.zo_norm:
        param.target_min = dataloader.get_values_min()
        param.target_max = dataloader.get_values_max()

    # Init model
    model = create_model(param)
    model.set_eval()
    visualizer = Visualizer(param)
    model.setup(param)

    # Init fake_dict
    if param.save_fake:
        fake_dict = model.init_fake_dict()
        print("[DEBUG] Initialized fake_dict:", fake_dict.keys())

    test_start_time = time.time()

    for i, data in enumerate(dataloader):
        model.set_input(data)

        # ✅ Ensure model has data_index for update_fake_dict
        if 'index' in data:
            model.data_index = data['index']
        else:
            raise KeyError("Missing 'index' in input data. Please check your dataset __getitem__ method.")

        model.test()

        if param.save_fake:
            fake_dict = model.update_fake_dict(fake_dict)
            print(f"[DEBUG] Batch {i}, fake_dict['index'] shape: {fake_dict['index'].shape}")
            print(f"[DEBUG] Batch {i}, fake_dict['fake'] shape: {fake_dict['fake'].shape}")

    if param.save_fake:
        # Fallback if update_fake_dict failed
        if 'index' not in fake_dict or len(fake_dict['index']) == 0:
            print("[WARN] 'index' is missing or empty. Fallback to sequential index.")
            fake_dict['index'] = np.arange(len(fake_dict['fake']))
        visualizer.save_fake_omics(fake_dict, sample_list, feature_list_A)

    print("Done generating fake A")
