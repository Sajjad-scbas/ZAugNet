from config import config
import ast


def predict(cfg):
    if cfg.model_name == 'zaugnet':
        from predict_zaugnet import predict
        predict(cfg, cfg.dataset)
    elif cfg.model_name == 'zaugnet+':
        from predict_zaugnet_plus import predict
        predict(cfg, cfg.dataset, cfg.factor, cfg.DPM)
    else :
        raise ValueError("You have to choose between ['zaugnet', 'zaugnet+'] for the prediction")


if __name__ == "__main__":
    cfg = config()
    cfg.device_ids = ast.literal_eval(cfg.device_ids)
    predict(cfg)
    print('Done!')
        