class japan_params:
    def __init__(self):
        self.sum_factor = 10 # 20
        self.vid_length = 500 # 150
        self.crop_loc = [192, 50] # [y, x]
        self.pixel_size = 160
        self.DS_sum = 20
        self.img_size = 64
        self.path = r'./data/japan'
        self.filename = r'min_m012_data.tif'
        self.model_name = 'LSTM_model'

class japan_scale8_params:
    def __init__(self):
        self.sum_factor = 20
        self.vid_length = 100
        self.crop_loc = [192, 50] # [y, x]
        self.pixel_size = 160
        self.DS_sum = 100
        self.img_size = 64
        self.path = r'./data/japan'
        self.filename = r'min_m012_data.tif'
        self.model_name = 'LSTM_model_scale8'

class japan_big_fov_params:
    def __init__(self):
        self.sum_factor = 20
        self.vid_length = 200
        self.crop_loc = [156, 50] #[y, x]
        self.pixel_size = 160
        self.DS_sum = 100
        self.img_size = 100
        self.path = r'./data/japan'
        self.filename = r'min_m012_data.tif'
        self.model_name = 'LSTM_model'

class drift_params:
    def __init__(self):
        self.sum_factor = 100
        self.vid_length = 100
        self.crop_loc = [6, 50] #[y, x]
        self.pixel_size = 160
        self.DS_sum = 200
        self.img_size = 64
        self.path = r'./data/microtubules/drift_correction'
        self.filename = r'before_drift_correction.tif'
        self.model_name = 'LSTM_model'

class mito_params:
    def __init__(self):
        self.sum_factor = 20
        self.vid_length = 200
        self.crop_loc = [32, 32] #[y, x]
        self.pixel_size = 233
        self.DS_sum = 10
        self.img_size = 64
        self.path = r'./data/mikes/new_mitochondria/exp3'
        self.filename = r'min_filtered.tif'
        self.model_name = 'LSTM_model_mito'

class mito2_params:
    def __init__(self):
        self.sum_factor = 40
        self.vid_length = 100
        self.crop_loc = [20, 0] #[y, x]
        self.pixel_size = 233
        self.DS_sum = 100
        self.img_size = 64
        self.path = r'./data/mikes/0107_mito/exp1'
        self.filename = r'exp1.tif'
        self.model_name = 'LSTM_model_mito'

class mito3_params:
    def __init__(self):
        self.sum_factor = 40
        self.vid_length = 100
        self.crop_loc = [20, 0] #[y, x]
        self.pixel_size = 233
        self.DS_sum = 100
        self.img_size = 64
        self.path = r'./data/mikes/0107_mito/exp1'
        self.filename = r'exp1_thunderSTORM.tif'
        self.model_name = 'LSTM_model_mito'

class mito4_params:
    def __init__(self):
        self.sum_factor = 25
        self.vid_length = 600
        self.crop_loc = [0, 0] #[y, x]
        self.pixel_size = 233
        self.DS_sum = 64
        self.img_size = 100
        self.path = r'./data/mikes/0107_mito/exp5'
        self.filename = r'exp5.tif'
        self.model_name = 'LSTM_model_mito'

class mito3_big_fov_params:
    def __init__(self):
        self.sum_factor = 40
        self.vid_length = 200
        self.crop_loc = [200, 60] #[y, x]
        self.pixel_size = 233
        self.DS_sum = 60
        self.img_size = 100
        self.path = r'./data/mikes/new_mitochondria/exp3'
        self.filename = r'COX8_JK073_10nM_20ms_20k_1_MMStack_Pos0_1.ome.tif'
        self.model_name = 'LSTM_model_mito'

class rotation_params:
    def __init__(self):
        self.sum_factor = 40
        self.vid_length = 100
        self.crop_loc = [75, 128] #[y, x]
        self.pixel_size = 160
        self.DS_sum = 40
        self.img_size = 64
        self.path = r'./data/microtubules/rotation_correction/exp009'
        self.filename = r'exp009.tif'
        self.model_name = 'LSTM_model'

class rotation2_params:
    def __init__(self):
        self.sum_factor = 20
        self.vid_length = 150
        self.crop_loc = [140, 65] #[y, x]
        self.pixel_size = 160
        self.DS_sum = 200
        self.img_size = 64
        self.path = r'./data/microtubules/rotation_correction/exp010'
        self.filename = r'exp010.tif'
        self.model_name = 'LSTM_model'

class rotation_big_fov_params:
    def __init__(self):
        self.sum_factor = 40
        self.vid_length = 125
        self.crop_loc = [32, 92] #[y, x]
        self.pixel_size = 160
        self.DS_sum = 200
        self.img_size = 128
        self.path = r'./data/microtubules/rotation_correction/exp009'
        self.filename = r'exp009.tif'
        self.model_name = 'LSTM_model'

class rotation2_big_fov_params:
    def __init__(self):
        self.sum_factor = 40
        self.vid_length = 125
        self.crop_loc = [32, 128] #[y, x]
        self.pixel_size = 160
        self.DS_sum = 200
        self.img_size = 128
        self.path = r'./data/microtubules/rotation_correction/exp010'
        self.filename = r'exp010.tif'
        self.model_name = 'LSTM_model'

class static_scale8_params:
    def __init__(self):
        self.sum_factor = 100
        self.vid_length = 100
        self.crop_loc = [32, 32] # [y, x]
        self.pixel_size = 160
        self.DS_sum = 100
        self.img_size = 32
        self.path = r'./data/microtubules/long_exp_static_mt'
        self.filename = r'Noam_100_exp_15_msec_003.tif'
        self.model_name = 'LSTM_model_scale8'

class mito_for_mike_params():
    def __init__(self):
        self.sum_factor = 40
        self.vid_length = 100
        self.crop_loc = [64, 96] #[y, x]
        self.pixel_size = 233
        self.DS_sum = 50
        self.img_size = 64
        self.path = r'./data/mikes/2706_mito/500pM/exp2'
        self.filename = r'JK114_500pM_20ms_20mW_2_MMImages.ome.tif'
        self.model_name = 'LSTM_model_mito'

class ER_decode_params():
    def __init__(self):
        self.sum_factor = 40
        self.vid_length = 132
        self.crop_loc = [0, 0] #[y, x]
        self.pixel_size = 127
        self.DS_sum = 40
        self.img_size = 200
        self.path = r'./data/DECODE_data/ER'
        self.filename = r'13_Calnexin-mEos32_561i100_UVi20_x15_1_MMStack_Default.ome.tif'
        self.model_name = 'LSTM_model'