from src.trainer import train_by_config, test_by_config
import os

train_by_config(os.path.join('settings', 'training_settings_2.ini'))
#
# test_by_config(os.path.join('test_settings', 'test_settings.ini'))
