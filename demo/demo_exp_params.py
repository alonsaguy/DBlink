class demo_params():
    def __init__(self):
        self.sum_factor = 10
        self.vid_length = 100
        self.crop_loc = [0, 0] #[y, x]
        self.pixel_size = 160
        self.DS_sum = 10
        self.img_size = 32
        self.path = r'./demo/data'
        self.filename = r'demo.csv'
        self.model_name = 'LSTM_model'
        self.window_size = 10