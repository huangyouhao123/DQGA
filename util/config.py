class Config(object):
    def __init__(self):
        self.base_directory = r'output'
        self.model_path = self.base_directory + '\\' + 'model.pkl'
        self.opt_path = self.base_directory + '\\' + 'optimizer.pkl'
        self.model_path_final = self.base_directory + '\\' + 'model_final.pkl'
        self.opt_path_final = self.base_directory + '\\' + 'optimizer_final.pkl'
        self.reTrain_model_path = self.base_directory + '\\' + 'retrain' + '\\' + 're_model.pkl'
        self.reBest_model_path = self.base_directory + '\\' + 'retrain' + '\\' + 're_best.pkl'
        self.patient_choice_num = 18
        self.cnn_net = False
        self.with_concrete_info=True
        self.find_gap=False
        self.hybrid_jump=3

config = Config()
